use std::{borrow::Cow, sync::OnceLock};

use anyhow::{Context, Result};
use naga::{proc::GlobalCtx, Handle};
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};
use regex::Regex;

const DECORATION_PRE: &str = "X_naga_oil_mod_X";
const DECORATION_POST: &str = "X";

// https://github.com/bevyengine/naga_oil/blob/master/src/compose/mod.rs#L417-L419
fn decode(from: &str) -> String {
    String::from_utf8(data_encoding::BASE32_NOPAD.decode(from.as_bytes()).unwrap()).unwrap()
}

/// Converts
///   * "\"../types\"::RtsStruct" => "types::RtsStruct"
///   * "../more-shader-files/reachme" => "reachme"
pub fn make_valid_rust_import(value: &str) -> String {
    let v = value.replace("\"../", "").replace('"', "");
    std::path::Path::new(&v)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or(&v)
        .to_string()
}

fn undecorate_regex() -> &'static Regex {
    static MEM: OnceLock<Regex> = OnceLock::new();

    MEM.get_or_init(|| {
        // https://github.com/bevyengine/naga_oil/blob/master/src/compose/mod.rs#L355-L363
        Regex::new(
            format!(
                r"(\x1B\[\d+\w)?([\w\d_]+){}([A-Z0-9]*){}",
                regex_syntax::escape(DECORATION_PRE),
                regex_syntax::escape(DECORATION_POST)
            )
            .as_str(),
        )
        .unwrap()
    })
}

// https://github.com/bevyengine/naga_oil/blob/master/src/compose/mod.rs#L421-L431
pub fn demangle_str(string: &str) -> Cow<str> {
    undecorate_regex().replace_all(string, |caps: &regex::Captures| {
        format!(
            "{}{}::{}",
            caps.get(1).map(|cc| cc.as_str()).unwrap_or(""),
            make_valid_rust_import(&decode(caps.get(3).unwrap().as_str())),
            caps.get(2).unwrap().as_str()
        )
    })
}

fn name_from_mangled(string: &str) -> String {
    let demangled = demangle_str(string);
    let mut parts = demangled.as_ref().split("::").collect::<Vec<&str>>();
    parts.pop().unwrap().to_owned()
}

fn rust_type_name(ty: Handle<naga::Type>, ctx: &GlobalCtx) -> Option<String> {
    let wgsl_name = ty.to_wgsl(ctx);

    Some(match wgsl_name.as_str() {
        "i32" | "u32" | "f32" | "bool" => wgsl_name,
        "vec2f" => "glam::Vec2".to_owned(),
        "vec3f" => "glam::Vec3".to_owned(),
        "vec4f" => "glam::Vec4".to_owned(),
        _ => return None,
    })
}

pub fn build_modules(
    paths: &[&str],
    includes: &[&str],
    base_path: &str,
    output_path: &str,
) -> Result<()> {
    println!("Building modules");

    let mut total = "".to_owned();

    let mut composer = Composer::default();

    for include in includes {
        let helper_source = &std::fs::read_to_string(include)?;
        let helper_name = std::path::Path::new(include)
            .file_stem()
            .context("file name")?
            .to_str()
            .context("Path name")?;
        println!("Adding helper {}", helper_name);
        composer.add_composable_module(ComposableModuleDescriptor {
            source: helper_source,
            file_path: &include.replace(base_path, ""),
            as_name: Some(helper_name.to_string()),
            ..Default::default()
        })?;
    }

    for path in paths {
        let source = &std::fs::read_to_string(path)?;
        let module = composer.make_naga_module(NagaModuleDescriptor {
            source,
            file_path: path,
            ..Default::default()
        })?;

        // get file name as module name
        let entries = &module.entry_points;
        assert!(entries.len() == 1, "Must have 1 entry per file");

        let entry = &entries[0];
        let [wg_x, wg_y, wg_z] = entry.workgroup_size;

        let mod_name = std::path::Path::new(path)
            .file_stem()
            .context("file name")?
            .to_str()
            .context("Path name")?;

        total += &format!("pub(crate) mod {mod_name} {{\n");
        total +=
            &format!("    pub(crate) const WORKGROUP_SIZE: [u32; 3] = [{wg_x}, {wg_y}, {wg_z}];\n");

        let ctx = &module.to_ctx();

        for t in module.constants.iter() {
            let type_and_value = match module.global_expressions[t.1.init] {
                naga::Expression::Literal(literal) => match literal {
                    naga::Literal::F64(v) => Some(format!("f32 = {v}")),
                    naga::Literal::F32(v) => Some(format!("f32 = {v}")),
                    naga::Literal::U32(v) => Some(format!("u32 = {v}")),
                    naga::Literal::I32(v) => Some(format!("i32 = {v}")),
                    naga::Literal::Bool(v) => Some(format!("bool = {v}")),
                    naga::Literal::I64(v) => Some(format!("i64 = {v}")),
                    naga::Literal::U64(v) => Some(format!("u64 = {v}")),
                    naga::Literal::AbstractInt(v) => Some(format!("i64 = {v}")),
                    naga::Literal::AbstractFloat(v) => Some(format!("f64 = {v}")),
                },
                _ => continue,
            };

            if let Some(type_and_value) = type_and_value {
                total += &format!(
                    "   pub(crate) const {}: {type_and_value};\n",
                    name_from_mangled(t.1.name.as_ref().unwrap()),
                );
            }
        }

        for t in module.types.iter() {
            match &t.1.inner {
                naga::TypeInner::Struct { members, span: _ }
                    if t.1.name.as_ref().unwrap() == "Uniforms" =>
                {
                    total += "#[repr(C, align(16))]\n";
                    total += "#[derive(bytemuck::NoUninit, Debug, PartialEq, Clone, Copy)]\n";
                    total += "pub(crate) struct Uniforms {\n";
                    for member in members {
                        let rust_name = rust_type_name(member.ty, ctx);
                        if let Some(rust_name) = rust_name {
                            total +=
                                &format!("    {}: {},\n", member.name.as_ref().unwrap(), rust_name);
                        }
                    }
                    total += "}\n";
                }
                _ => continue,
            }
        }

        total += "pub(crate) fn create_shader_source() -> String {\n";
        total += &format!("include_str!(\"{path}\").to_owned()\n");
        total += "}\n";
        total += "}\n";
    }

    // TODO: Rerun-if-changed.
    // TODO: Proper error handling.
    // TODO: Shader define map.

    std::fs::write(output_path, total)?;

    Ok(())
}

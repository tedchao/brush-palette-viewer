//! Differential test: the native Rust optimizer vs. the reference Python
//! implementation in `python/constraint_optimizer.py`.
//!
//! For a battery of cases it runs `brush_palette::optimizer::run_optimizer` and
//! the Python `alternating_optimize` (via `python/compare_driver.py`) on
//! bit-identical inputs, then asserts the `dP` and `L` outputs agree.
//!
//! The two implementations should agree to ~f32 precision: all internal math
//! is `f64` in both, the algorithm is deterministic (no RNG), and a square
//! linear solve `A\b` has a unique answer regardless of solver (faer LU vs.
//! scipy `spsolve`). The only real error source is the final `f64 -> f32`
//! conversion of the Rust result, so the tolerance is set well above that.
//!
//! The reference is run with `uv` — the driver carries PEP 723 inline
//! dependencies, so `uv` fetches numpy/scipy on first run and no manual
//! environment setup is needed. If `uv` is unavailable the test SKIPS rather
//! than fails; override the `uv` binary with the `BRUSH_UV` env var. To see
//! the per-case report:
//!
//!     cargo test -p brush-palette --test compare_python -- --nocapture

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use brush_palette::optimizer::{
    run_optimizer, CurveConstraint, PaletteConstraint, PixelConstraint,
};
use serde_json::{json, Value};

/// Max allowed absolute difference between Rust and Python outputs.
const TOL: f64 = 1e-3;

// ── Deterministic input generation ──────────────────────────────────────────

/// Tiny LCG so cases are reproducible without pulling in `rand`. The exact
/// values it produces are sent to *both* implementations, so the generator
/// only needs to be deterministic within a single run.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// Next value in [0, 1).
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32)
    }
}

struct Case {
    name: &'static str,
    palette: Vec<f32>, // K * 3, row-major
    k: usize,
    n: usize,
    pixel: Vec<PixelConstraint>,
    palette_cons: Vec<PaletteConstraint>,
    curve_cons: Vec<CurveConstraint>,
}

/// Palette with black at index 0, white at index 1, random chromatic colors
/// afterwards (matching the optimizer's index convention).
fn gen_palette(rng: &mut Lcg, k: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; k * 3];
    p[3] = 1.0;
    p[4] = 1.0;
    p[5] = 1.0;
    for i in 2..k {
        for ch in 0..3 {
            p[i * 3 + ch] = rng.unit();
        }
    }
    p
}

fn gen_pixel(rng: &mut Lcg, k: usize) -> PixelConstraint {
    PixelConstraint {
        w_at_pixel: (0..k).map(|_| rng.unit()).collect(),
        target_rgb: [rng.unit(), rng.unit(), rng.unit()],
    }
}

fn make_cases() -> Vec<Case> {
    let mut cases = Vec::new();

    // 1. No constraints — should return identity (dP ≈ 0).
    {
        let mut rng = Lcg::new(1);
        let k = 4;
        cases.push(Case {
            name: "no_constraints",
            palette: gen_palette(&mut rng, k),
            k,
            n: 100,
            pixel: vec![],
            palette_cons: vec![],
            curve_cons: vec![],
        });
    }

    // 2. Single chromatic palette pin.
    {
        let mut rng = Lcg::new(2);
        let k = 4;
        cases.push(Case {
            name: "palette_pin_chromatic",
            palette: gen_palette(&mut rng, k),
            k,
            n: 100,
            pixel: vec![],
            palette_cons: vec![PaletteConstraint {
                idx: 2,
                target: [0.1, 0.8, 0.2],
            }],
            curve_cons: vec![],
        });
    }

    // 3. Achromatic pins (black + white) — the hard-pin path.
    {
        let mut rng = Lcg::new(3);
        let k = 5;
        cases.push(Case {
            name: "palette_pin_achromatic",
            palette: gen_palette(&mut rng, k),
            k,
            n: 100,
            pixel: vec![],
            palette_cons: vec![
                PaletteConstraint {
                    idx: 0,
                    target: [0.12, 0.10, 0.14],
                },
                PaletteConstraint {
                    idx: 1,
                    target: [0.93, 0.95, 0.90],
                },
            ],
            curve_cons: vec![],
        });
    }

    // 4. Image-space (pixel) constraints only.
    {
        let mut rng = Lcg::new(4);
        let k = 5;
        let palette = gen_palette(&mut rng, k);
        let pixel = (0..3).map(|_| gen_pixel(&mut rng, k)).collect();
        cases.push(Case {
            name: "pixel_only",
            palette,
            k,
            n: 100,
            pixel,
            palette_cons: vec![],
            curve_cons: vec![],
        });
    }

    // 5. Direct curve-point constraints only. l_x values avoid 0.5 so that
    //    round(l_x*(N-1)) is unambiguous across Python/Rust rounding modes.
    {
        let mut rng = Lcg::new(5);
        let k = 4;
        cases.push(Case {
            name: "curve_only",
            palette: gen_palette(&mut rng, k),
            k,
            n: 100,
            pixel: vec![],
            palette_cons: vec![],
            curve_cons: vec![
                CurveConstraint {
                    idx: 2,
                    l_x: 0.3,
                    l_y: 0.45,
                },
                CurveConstraint {
                    idx: 3,
                    l_x: 0.7,
                    l_y: 0.62,
                },
            ],
        });
    }

    // 6. Everything at once.
    {
        let mut rng = Lcg::new(6);
        let k = 6;
        let palette = gen_palette(&mut rng, k);
        let pixel = (0..2).map(|_| gen_pixel(&mut rng, k)).collect();
        cases.push(Case {
            name: "mixed",
            palette,
            k,
            n: 100,
            pixel,
            palette_cons: vec![PaletteConstraint {
                idx: 3,
                target: [0.7, 0.25, 0.4],
            }],
            curve_cons: vec![
                CurveConstraint {
                    idx: 2,
                    l_x: 0.25,
                    l_y: 0.35,
                },
                CurveConstraint {
                    idx: 4,
                    l_x: 0.8,
                    l_y: 0.7,
                },
            ],
        });
    }

    // 7. Non-default curve sample count.
    {
        let mut rng = Lcg::new(7);
        let k = 5;
        let palette = gen_palette(&mut rng, k);
        let pixel = (0..2).map(|_| gen_pixel(&mut rng, k)).collect();
        cases.push(Case {
            name: "mixed_n64",
            palette,
            k,
            n: 64,
            pixel,
            palette_cons: vec![PaletteConstraint {
                idx: 2,
                target: [0.3, 0.6, 0.5],
            }],
            curve_cons: vec![],
        });
    }

    cases
}

// ── JSON marshalling ────────────────────────────────────────────────────────

/// Build the JSON for one case. f32 inputs are widened to f64 so Python
/// receives bit-identical values (Rust then converts the same f32 -> f64).
fn case_to_json(case: &Case) -> Value {
    let palette: Vec<Vec<f64>> = case
        .palette
        .chunks(3)
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect();
    let w: Vec<Vec<f64>> = case
        .pixel
        .iter()
        .map(|pc| pc.w_at_pixel.iter().map(|&x| x as f64).collect())
        .collect();
    let t: Vec<Vec<f64>> = case
        .pixel
        .iter()
        .map(|pc| pc.target_rgb.iter().map(|&x| x as f64).collect())
        .collect();
    let palette_cons: Vec<Value> = case
        .palette_cons
        .iter()
        .map(|p| {
            json!([
                p.idx,
                [p.target[0] as f64, p.target[1] as f64, p.target[2] as f64]
            ])
        })
        .collect();
    let curve_cons: Vec<Value> = case
        .curve_cons
        .iter()
        .map(|c| json!([c.idx, c.l_x as f64, c.l_y as f64]))
        .collect();

    json!({
        "palette": palette,
        "c": case.pixel.len(),
        "w_at_cons": w,
        "target_colors": t,
        "palette_cons": palette_cons,
        "curve_cons": curve_cons,
        "N": case.n,
    })
}

fn json_f64_array(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("python result missing array '{key}'"))
        .iter()
        .map(|x| x.as_f64().expect("python result entry not a number"))
        .collect()
}

// ── Python invocation ───────────────────────────────────────────────────────

fn uv_exe() -> String {
    std::env::var("BRUSH_UV").unwrap_or_else(|_| "uv".to_string())
}

fn driver_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("compare_driver.py")
}

/// `uv run --script <driver>` — uv resolves the driver's PEP 723 deps.
fn driver_command(driver: &PathBuf) -> Command {
    let mut cmd = Command::new(uv_exe());
    cmd.arg("run").arg("--script").arg(driver);
    cmd
}

/// True if `uv` can run the driver (deps resolved, module importable).
fn python_available(driver: &PathBuf) -> bool {
    driver_command(driver)
        .arg("--check")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run all cases through the Python driver in one subprocess (amortizes the
/// scipy import) and return the parsed `results` array.
fn run_python(driver: &PathBuf, input_json: &str) -> Vec<Value> {
    let mut child = driver_command(driver)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn python driver");
    child
        .stdin
        .take()
        .expect("driver stdin")
        .write_all(input_json.as_bytes())
        .expect("failed to write to python driver");
    let out = child
        .wait_with_output()
        .expect("failed to wait for python driver");
    if !out.status.success() {
        panic!(
            "python driver exited with {}:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let parsed: Value =
        serde_json::from_slice(&out.stdout).expect("python driver produced invalid JSON");
    parsed["results"]
        .as_array()
        .expect("python result missing 'results'")
        .clone()
}

/// (max abs diff, flat index where it occurs).
fn max_abs_diff(rust: &[f32], python: &[f64]) -> (f64, usize) {
    assert_eq!(
        rust.len(),
        python.len(),
        "output length mismatch: rust={} python={}",
        rust.len(),
        python.len()
    );
    let mut worst = 0.0;
    let mut at = 0;
    for (i, (&r, &p)) in rust.iter().zip(python).enumerate() {
        let d = (r as f64 - p).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

#[test]
fn compare_with_python() {
    let driver = driver_path();
    if !python_available(&driver) {
        eprintln!(
            "SKIP compare_with_python: '{}' could not run the reference driver.\n\
             Install uv (or set BRUSH_UV) to enable this test.",
            uv_exe()
        );
        return;
    }

    let cases = make_cases();

    // Run the whole battery through Python in a single subprocess.
    let payload = json!({
        "cases": cases.iter().map(case_to_json).collect::<Vec<_>>(),
    });
    let py_results = run_python(&driver, &payload.to_string());
    assert_eq!(
        py_results.len(),
        cases.len(),
        "python returned {} results for {} cases",
        py_results.len(),
        cases.len()
    );

    println!(
        "\n{:<24} {:>9}  {:>11}  {:>11}",
        "case", "n_iter", "max|ΔdP|", "max|ΔL|"
    );
    println!("{}", "-".repeat(60));

    let mut failures = Vec::new();
    for (case, py) in cases.iter().zip(&py_results) {
        let result = run_optimizer(
            &case.palette,
            case.k,
            &case.pixel,
            &case.palette_cons,
            &case.curve_cons,
            case.n,
        )
        .unwrap_or_else(|e| panic!("rust optimizer failed for '{}': {e}", case.name));

        let py_dp = json_f64_array(py, "dP");
        let py_l = json_f64_array(py, "L");
        let py_iter = py["n_iter"].as_u64().expect("python n_iter") as u32;

        let (dp_err, dp_at) = max_abs_diff(&result.delta_palette, &py_dp);
        let (l_err, l_at) = max_abs_diff(&result.l_curves, &py_l);

        let iter_col = format!("{}/{}", result.n_iter, py_iter);
        println!(
            "{:<24} {:>9}  {:>11.3e}  {:>11.3e}",
            case.name, iter_col, dp_err, l_err
        );

        if dp_err > TOL || l_err > TOL {
            failures.push(format!(
                "'{}': max|ΔdP|={dp_err:.3e} (idx {dp_at}), max|ΔL|={l_err:.3e} (idx {l_at})",
                case.name
            ));
        }
    }
    println!();

    assert!(
        failures.is_empty(),
        "Rust optimizer diverged from Python reference (tol {TOL:.0e}):\n  {}",
        failures.join("\n  ")
    );
}

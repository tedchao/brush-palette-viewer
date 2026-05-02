//! Python bridge to constraint_optimizer.alternating_optimize.

use anyhow::{anyhow, Result};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::path::PathBuf;
use std::sync::Once;

#[derive(Debug, Clone)]
pub struct PixelConstraint {
    pub w_at_pixel: Vec<f32>,
    pub target_rgb: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct PaletteConstraint {
    pub idx: usize,
    pub target: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct CurveConstraint {
    pub idx: usize,
    pub l_x: f32,
    pub l_y: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizerResult {
    pub delta_palette: Vec<f32>,
    pub l_curves: Vec<f32>,
    pub n_iter: u32,
    pub runtime_ms: f64,
}

static INIT_PATH: Once = Once::new();

fn ensure_python_path(py: Python) -> PyResult<()> {
    let python_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    let sys = py.import_bound("sys")?;
    let path = sys.getattr("path")?;
    let path_list = path.downcast::<PyList>()?;
    let dir_str = python_dir.to_string_lossy().to_string();

    let mut already = false;
    for item in path_list.iter() {
        if let Ok(s) = item.extract::<String>() {
            if s == dir_str {
                already = true;
                break;
            }
        }
    }
    if !already {
        path_list.insert(0, dir_str)?;
    }
    Ok(())
}

fn run_optimizer_inner(
    py: Python,
    palette: &[f32],
    k_full: usize,
    pixel_cons: &[PixelConstraint],
    palette_cons: &[PaletteConstraint],
    curve_cons: &[CurveConstraint],
    n_curve_samples: usize,
) -> PyResult<OptimizerResult> {
    INIT_PATH.call_once(|| {
        if let Err(e) = ensure_python_path(py) {
            log::error!("Failed to set Python sys.path: {:?}", e);
        }
    });

    let module = py.import_bound("constraint_optimizer")?;
    let func = module.getattr("alternating_optimize")?;

    // palette: (K, 3)
    let palette_2d: Vec<Vec<f32>> = palette.chunks(3).map(|c| c.to_vec()).collect();
    let palette_np = PyArray2::<f32>::from_vec2_bound(py, &palette_2d)?;

    // W_at_cons: (c, K), target_colors: (c, 3)
    let c = pixel_cons.len();
    let w_np = if c == 0 {
        PyArray2::<f32>::zeros_bound(py, [0, k_full], false)
    } else {
        let w_data: Vec<Vec<f32>> = pixel_cons.iter().map(|pc| pc.w_at_pixel.clone()).collect();
        PyArray2::<f32>::from_vec2_bound(py, &w_data)?
    };
    let t_np = if c == 0 {
        PyArray2::<f32>::zeros_bound(py, [0, 3], false)
    } else {
        let t_data: Vec<Vec<f32>> = pixel_cons.iter().map(|pc| pc.target_rgb.to_vec()).collect();
        PyArray2::<f32>::from_vec2_bound(py, &t_data)?
    };

    // palette_cons: list of (i, np.array([r,g,b]))
    let pc_list = PyList::empty_bound(py);
    for pc in palette_cons {
        let arr = PyArray1::<f32>::from_slice_bound(py, &pc.target);
        let tup = PyTuple::new_bound(py, &[pc.idx.into_py(py), arr.into_py(py)]);
        pc_list.append(tup)?;
    }

    // curve_cons: list of (i, L_x, L_y)
    let cc_list = PyList::empty_bound(py);
    for cc in curve_cons {
        let tup = PyTuple::new_bound(
            py,
            &[cc.idx.into_py(py), cc.l_x.into_py(py), cc.l_y.into_py(py)],
        );
        cc_list.append(tup)?;
    }

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("N", n_curve_samples)?;
    kwargs.set_item("verbose", false)?;

    let args = PyTuple::new_bound(
        py,
        &[
            palette_np.into_py(py),
            w_np.into_py(py),
            t_np.into_py(py),
            pc_list.into_py(py),
            cc_list.into_py(py),
        ],
    );

    let result = func.call(args, Some(&kwargs))?;
    let dict = result.downcast::<PyDict>()?;

    // dP: (K, 3) → flat row-major
    let dp_obj = dict
        .get_item("dP")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing dP"))?;
    let dp_arr = dp_obj.downcast::<PyArray2<f64>>()?;
    let dp_vec: Vec<f32> = dp_arr
        .readonly()
        .as_array()
        .iter()
        .map(|&x| x as f32)
        .collect();

    // L: (N, K) → flat col-major (L_curves[k*N + n] in shader)
    let l_obj = dict
        .get_item("L")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing L"))?;
    let l_arr = l_obj.downcast::<PyArray2<f64>>()?;
    let l_view = l_arr.readonly();
    let l_array = l_view.as_array();
    let mut l_vec = Vec::with_capacity(n_curve_samples * k_full);
    for k in 0..k_full {
        for n in 0..n_curve_samples {
            l_vec.push(l_array[[n, k]] as f32);
        }
    }

    let n_iter = dict
        .get_item("n_iter")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing n_iter"))?
        .extract::<u32>()?;
    let runtime = dict
        .get_item("runtime")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing runtime"))?
        .extract::<f64>()?;

    Ok(OptimizerResult {
        delta_palette: dp_vec,
        l_curves: l_vec,
        n_iter,
        runtime_ms: runtime * 1000.0,
    })
}

pub fn run_optimizer(
    palette: &[f32],
    k_full: usize,
    pixel_cons: &[PixelConstraint],
    palette_cons: &[PaletteConstraint],
    curve_cons: &[CurveConstraint],
    n_curve_samples: usize,
) -> Result<OptimizerResult> {
    assert_eq!(palette.len(), k_full * 3);

    Python::with_gil(|py| {
        run_optimizer_inner(
            py,
            palette,
            k_full,
            pixel_cons,
            palette_cons,
            curve_cons,
            n_curve_samples,
        )
        .map_err(|e| anyhow!("Python optimizer failed: {}", e))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_no_constraints() {
        let palette: Vec<f32> = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let result = run_optimizer(&palette, 4, &[], &[], &[], 100).unwrap();
        assert_eq!(result.delta_palette.len(), 4 * 3);
        assert_eq!(result.l_curves.len(), 100 * 4);
        for v in &result.delta_palette {
            assert!(v.abs() < 1e-5);
        }
        println!(
            "Optimizer ran in {:.1}ms, n_iter={}",
            result.runtime_ms, result.n_iter
        );
    }
    
    #[test]
    fn palette_constraint_shifts_dp() {
        // K=4, pin chromatic palette [2] (red) → green
        let palette: Vec<f32> = vec![
            0.0, 0.0, 0.0,  // black
            1.0, 1.0, 1.0,  // white
            1.0, 0.0, 0.0,  // red (chromatic)
            0.0, 0.0, 1.0,  // blue (chromatic)
        ];
        let palette_cons = vec![PaletteConstraint {
            idx: 2,
            target: [0.0, 1.0, 0.0],
        }];
        let result = run_optimizer(&palette, 4, &[], &palette_cons, &[], 100).unwrap();
        
        // ΔP for index 2 should push red toward green
        let dp_idx2 = &result.delta_palette[6..9];
        println!("dP[2] = {:?}", dp_idx2);
        // Expected roughly: [-1, +1, 0]
        assert!(dp_idx2[0] < -0.1, "R component should decrease");
        assert!(dp_idx2[1] > 0.1, "G component should increase");
        
        // ΔP for other chromatic index should be near zero (sparsity)
        let dp_idx3 = &result.delta_palette[9..12];
        println!("dP[3] = {:?}", dp_idx3);
        let norm3: f32 = dp_idx3.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm3 < 0.05, "dP[3] should be near zero (sparsity)");
        
        println!("Optimizer ran in {:.1}ms", result.runtime_ms);
    }
}
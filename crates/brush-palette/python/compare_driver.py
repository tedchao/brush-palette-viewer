# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.4.6",
#     "scipy>=1.17.1",
# ]
# ///
"""
compare_driver.py
=================
Reference driver for `tests/compare_python.rs`. It runs the *Python*
`constraint_optimizer.alternating_optimize` so the Rust port can be diffed
against it.

Protocol (all numeric, no strings in the payload):
  * `--check`            -> import numpy/scipy/constraint_optimizer, print "ok".
  * otherwise            -> read one JSON object from stdin of the form

        { "cases": [ {
            "palette":        [[r,g,b], ...],   # K rows
            "c":              <int>,            # number of pixel constraints
            "w_at_cons":      [[..K..], ...],   # c rows  (raw weights)
            "target_colors":  [[r,g,b], ...],   # c rows
            "palette_cons":   [[idx,[r,g,b]], ...],
            "curve_cons":     [[idx,lx,ly], ...],
            "N":              <int>
          }, ... ] }

    and write `{ "results": [ {L, dP, n_iter}, ... ] }` to stdout, where
        L  is L.flatten(order='F')  -> col-major  l_curves[k*N + n]
        dP is dP.flatten(order='C') -> row-major  (K, 3)

  The flatten orders are chosen to match exactly what Rust's `run_optimizer`
  returns (`OptimizerResult::l_curves` / `::delta_palette`).
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if "--check" in sys.argv:
        import numpy  # noqa: F401
        import scipy.sparse  # noqa: F401
        import scipy.sparse.linalg  # noqa: F401
        import constraint_optimizer  # noqa: F401
        print("ok")
        return

    import numpy as np
    from constraint_optimizer import alternating_optimize

    payload = json.load(sys.stdin)
    results = []

    for case in payload["cases"]:
        palette = np.array(case["palette"], dtype=np.float64)        # (K, 3)
        k = palette.shape[0]
        c = int(case["c"])

        if c > 0:
            w = np.array(case["w_at_cons"], dtype=np.float64).reshape(c, k)
            t = np.array(case["target_colors"], dtype=np.float64).reshape(c, 3)
        else:
            w = np.zeros((0, k), dtype=np.float64)
            t = np.zeros((0, 3), dtype=np.float64)

        palette_cons = [
            (int(idx), np.array(rgb, dtype=np.float64))
            for (idx, rgb) in case["palette_cons"]
        ]
        curve_cons = [
            (int(idx), float(lx), float(ly))
            for (idx, lx, ly) in case["curve_cons"]
        ]
        n = int(case["N"])

        out = alternating_optimize(palette, w, t, palette_cons, curve_cons, N=n)

        l = np.asarray(out["L"], dtype=np.float64)    # (N, K)
        dp = np.asarray(out["dP"], dtype=np.float64)  # (K, 3)

        results.append({
            "L": l.flatten(order="F").tolist(),
            "dP": dp.flatten(order="C").tolist(),
            "n_iter": int(out["n_iter"]),
        })

    json.dump({"results": results}, sys.stdout)


if __name__ == "__main__":
    main()

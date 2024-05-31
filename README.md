# Brush 0.1

## Overview

Brush is a Gaussian Splatting engine using the [Burn](https://github.com/tracel-ai/burn) framework, and custom WGSL kernels. This allows for a highly portable implementation.

## Getting strated

Be sure to have rust 1.78+ installed.

Download some example datasets [here](https://drive.google.com/file/d/15w5NGHUs5prPDyIpENVvNXjRjFt5CB9T/view?usp=sharing), and place them in a brush_data folder in the project root.

When training, most of the data is visualized with [Rerun](https://rerun.io/). To install Rerun, follow the [instructions](https://rerun.io/docs/getting-started/installing-viewer).

Run the app with `cargo run` or `cargo run --release`. Release mode isn't much faster as though as everything is GPU bound.

You can run tests with `cargo test`. Currently these tests do not pass on Metal however, TBD.
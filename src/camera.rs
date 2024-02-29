use ndarray::{self, Dim};

pub(crate) struct Camera {
    transform: ndarray::Array<f32, Dim<[usize; 2]>>,
}

impl Camera {
    pub(crate) fn new() -> Self {
        Camera {
            transform: ndarray::Array::eye(4),
        }
    }
}

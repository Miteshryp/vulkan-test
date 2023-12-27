pub struct Vertex {
    x: f32,
    y: f32,
    z: f32
}

pub struct Color4 {
    r: usize,
    g: usize,
    b: usize,
    a: usize
}


pub struct Matrix {
    matrix: Vec<Vec<f32>>,
    dim_x: usize,
    dim_y: usize
}


pub trait VectorTransforms {
    pub fn mulMat(&mut self, m: &Matrix);
    pub fn translateVector(&mut self, m: &Matrix);
}
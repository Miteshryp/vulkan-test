use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]

pub struct Vec2 {
    #[format(R32_SFLOAT)] // single f32 value
    pub x: f32,

    #[format(R32_SFLOAT)] // single f32 value
    pub y: f32,
}

#[derive(BufferContents, Clone, Vertex, Debug)]
#[repr(C)]
pub struct VertexPoint {
    #[format(R32G32B32_SFLOAT)]
    pub position: Vec3,

    #[format(R32G32B32_SFLOAT)]
    pub color: Vec3
}

impl VertexPoint {
}

#[derive(BufferContents, Clone, Debug)]
// #[derive(Clone)]
#[repr(C)]
pub struct Vec3 {
    // #[format(R32_SFLOAT)] // single f32 value
    pub x: f32,

    // #[format(R32_SFLOAT)] // single f32 value
    pub y: f32,

    // #[format(R32_SFLOAT)] // single f32 value
    pub z: f32,
}

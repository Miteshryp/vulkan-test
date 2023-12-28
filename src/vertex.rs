use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]

pub struct Vec2 {
    #[format(R32_SFLOAT)] // single f32 value
    pub x: f32,

    #[format(R32_SFLOAT)] // single f32 value
    pub y: f32,
}

fn test() {
    let v = Vec2 { x: 0.0_f32, y: 0.0 };
}


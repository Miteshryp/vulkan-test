// use glm::translate;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};
// extern crate nalgebra_glm as glm;


#[derive(BufferContents, Clone, Vertex, Debug)]
#[repr(C)]
pub struct VertexPoint {
    #[format(R32G32B32_SFLOAT)]
    pub position: Vec3,

    #[format(R32G32B32_SFLOAT)]
    pub color: Vec3
}


// fn test() {
//     let pos = glm::vec4(1.0, 1.0, 1.0, 1.0);

//     let mut model: glm::Mat4 = glm::identity(); // object position
//     model = glm::translate(&model, &glm::vec3(1.0, 1.0, 1.0));

//     let mut view: glm::Mat4 = glm::identity(); // camera position
//     view = glm::look_at(&glm::vec3(0.0,0.0,0.0), &glm::vec3(0.0,0.0,1.0), &glm::vec3(0.0,1.0,0.0));
    
//     let projection: glm::Mat4 = glm::perspective(16.0 / 9.0, std::f32::consts::PI / 4.0, 0.1, 1000.0);

//     let final_pos: glm::Vec4 = (model * view * projection * pos).into();
// }

#[derive(BufferContents, Clone, Debug)]
#[repr(C)]
pub struct Vec3 {
    // #[format(R32_SFLOAT)] // single f32 value
    pub x: f32,

    // #[format(R32_SFLOAT)] // single f32 value
    pub y: f32,

    // #[format(R32_SFLOAT)] // single f32 value
    pub z: f32,
}

use std::sync::Arc;
use vulkano::{
    device::Device,
    shader::ShaderModule,
};

pub mod vs {
    vulkano_shaders::shader!(
        ty: "vertex",
        src: r"
            #version 460

            layout(set = 0, binding = 0) uniform MvpMatrix {
                mat4 model;
                mat4 view;
                mat4 projection;
            } mvp;

            // Vertex Attibutes
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec3 color;
            layout(location = 3) in vec2 tex_coord;

            // Instance attributes
            layout(location = 4) in vec3 global_position;
            layout(location = 5) in float local_scale;
            layout(location = 6) in uint tex_index;
    
            
            // Vertex inputs
            // layout(location = 0) in vec3 position; // What is this position? - Shader only runs on the fragments covered by the vertices, so we will have to pass this vertex into the rasterizer to obtain the correct fragments to process.
            
            void main() {
                vec4 player_position = mvp.model * vec4(position * local_scale + global_position, 1.0);
                gl_Position = mvp.projection * mvp.view * player_position;
            }
            ",
    );
}

pub mod fs {
    vulkano_shaders::shader!(
        ty: "fragment",
        src: "
            #version 460

            layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput attachment_color;
            layout(input_attachment_index = 1, set = 1, binding = 1) uniform subpassInput attachment_normals;

            layout(set = 0, binding = 0) uniform MvpMatrix {
                mat4 model;
                mat4 view;
                mat4 projection;
            } mvp;

            layout(location = 0) out vec4 f_color;
            
            void main() {
                vec4 new_color = vec4(subpassLoad(attachment_color).rgb, 1.0);
                f_color = vec4(0.0,1.0,0.5,1.0);
            }
        ",
    );
}

pub fn load_vertex_shader(device: Arc<Device>) -> Arc<ShaderModule> {
    vs::load(device).expect("Failed to load vertex shader")
}

pub fn load_fragment_shader(device: Arc<Device>) -> Arc<ShaderModule> {
    fs::load(device).expect("Failed to load vertex shader")
}

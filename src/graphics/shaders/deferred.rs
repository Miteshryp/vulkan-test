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

            layout(location = 0) out vec3 out_color;
            layout(location = 1) out vec2 out_textureMapping;
            layout(location = 2) out uint out_texture_index;
            layout(location = 3) out vec3 out_normal;
            
            void main() {                
                gl_Position = mvp.projection * mvp.view * mvp.model * vec4(position.xyz * local_scale + global_position, 1.0);
                
                out_color = color;
                out_textureMapping = tex_coord;
                out_texture_index = tex_index;
                out_normal = mat3(mvp.model) * normal;
            }
            ",
    );
}

// flat keyword signifies that the attribute is pulled only once from the \"provoking vertex\", and not from every fragment in the rendering zone.


pub mod fs {
    vulkano_shaders::shader!(
        ty: "fragment",
        src: "
            #version 460

            layout(location = 0) in vec3 color;
            layout(location = 1) in vec2 tex_coord;
            layout(location = 2) flat in uint tex_index;
            layout(location = 3) in vec3 in_normal;
            

            layout(set = 0, binding = 1) uniform sampler s; 
            // layout(set = 0, binding = 1) uniform texture2D tex;
            layout(set = 0, binding = 2) uniform texture2DArray tex;

            // For next pass (lighting), we need to pass the color and normal for the fragment
            layout(location = 0) out vec4 f_color;
            layout(location = 1) out vec3 f_normal;
            
            
            void main() {            
                f_color = texture(sampler2DArray(tex, s), vec3(tex_coord, tex_index));
                f_normal = in_normal;
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

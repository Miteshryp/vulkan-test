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

            layout(push_constant) uniform PushConstantData {
                float view;
            } pc;
            
            layout(set = 1, binding = 2) uniform MvpMatrix {
                mat4 model;
                mat4 view;
                mat4 projection;
            } mvp;
    
            
            // Vertex inputs
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;
            layout(location = 2) in vec2 tex_coord;

            // Instance inputs
            layout(location = 3) in vec3 global_position;
            layout(location = 4) in float local_scale;
            layout(location = 5) in uint tex_index;


            // Vertex Outputs
            layout(location = 0) out vec3 out_color;
            layout(location = 1) out float v;
            layout(location = 2) out vec2 textureMapping;
            layout(location = 3) out uint texture_index;
            
            void main() {
                vec4 player_position = mvp.model * vec4(position.xyz * local_scale + global_position, 1.0);
                gl_Position = mvp.projection * mvp.view * player_position;
                out_color = color;
                
                v = pc.view;
                textureMapping = tex_coord;
                texture_index = tex_index;
            }
            ",
    );
}

pub mod fs {
    vulkano_shaders::shader!(
        ty: "fragment",
        src: "
            #version 460

            layout(location = 0) in vec3 color;
            layout(location = 1) in float v;
            layout(location = 2) in vec2 tex_coord;
            layout(location = 3) flat in uint tex_index;

            // flat keyword signifies that the attribute is pulled only once from the \"provoking vertex\", and not from every fragment in the rendering zone.
            
            
            layout(push_constant) uniform PushConstantData {
                float view;
            } pc;

            layout(set = 1, binding = 0) uniform sampler s; 
            // layout(set = 1, binding = 1) uniform texture2D tex;
            layout(set = 1, binding = 1) uniform texture2DArray tex;


            layout(set = 1, binding = 2) uniform MvpMatrix {
                mat4 model;
                mat4 view;
                mat4 projection;
            } mvp;

            layout(location = 0) out vec4 f_color;
    
            void main() {
                // float dist = normalize(gl_FragCoord.xy).y;
                // f_color = texture(sampler2D(tex, s), tex_coord) + vec4(color.x + sin(pc.view), color.y, color.z + cos(pc.view), dist);
                f_color = texture(sampler2DArray(tex, s), vec3(tex_coord, tex_index));
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


use std::sync::Arc;
use vulkano::{
    device::Device,
    shader::ShaderModule
};

struct VertexShader {
    src: String
}

struct FragmentShader {
    src: String
}

pub mod vs {
    vulkano_shaders::shader!(
        ty: "vertex",
        src: r"
            #version 460
    
            // layout(location = 0) in vec2 position;
            layout(location = 0) in float x;
            layout(location = 1) in float y;

            layout(set = 0, binding = 0) uniform Data {
                float view;
            } uniforms;

            layout(location = 0) out float v;
            
            
            void main() {
                gl_Position = vec4(x, y, 0.0, 1.0);
                v = uniforms.view;
            }
            ",
    );
}


pub mod fs {
    vulkano_shaders::shader!(
        ty: "fragment",
        src: "
            #version 460
    
            layout(location = 0) in float v;
            layout(location = 0) out vec4 f_color;
    
            void main() {
                float dist = normalize(gl_FragCoord.xy).y;
                // f_color = vec4(1.0, dist, 0.0, 0.2);
                f_color = vec4(sin(v), dist, 0.0, 1);
            }
        ",
    );
}



pub fn get_vertex_shader (device: Arc<Device>,) ->  Arc<ShaderModule> {
    vs::load(device).expect("Failed to load vertex shader")
}

pub fn get_fragment_shader (device: Arc<Device>) -> Arc<ShaderModule> {
    fs::load(device).expect("Failed to load vertex shader")
}
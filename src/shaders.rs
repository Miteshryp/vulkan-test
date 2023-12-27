
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

mod vs {
    vulkano_shaders::shader!(
        ty: "vertex",
        src: r"
            #version 460
    
            // layout(location = 0) in vec2 position;
            layout(location = 0) in float x;
            layout(location = 1) in float y;
            

            
            void main() {
                gl_Position = vec4(x, y, 0.0, 1.0);
            }
            ",
    );
}


mod fs {
    vulkano_shaders::shader!(
        ty: "fragment",
        src: "
            #version 460
    
            layout(location = 0) out vec4 f_color;
    
            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
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
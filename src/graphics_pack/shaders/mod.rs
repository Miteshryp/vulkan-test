
use std::sync::Arc;
use vulkano::{
    device::Device,
    shader::ShaderModule, memory::allocator::MemoryTypeFilter, buffer::{Buffer, BufferUsage, BufferContents},
};

use crate::{graphics_pack::buffers, GenericBufferAllocator};

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

            layout(set = 0, binding = 0) uniform Data {
                float view;
            } uniforms;
    
    
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;

            layout(location = 0) out vec3 out_color;
            layout(location = 1) out float v;
            
            void main() {
                gl_Position = vec4(position.xyz, 1.0);
                out_color = color;
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

            layout(location = 0) in vec3 color;
            layout(location = 1) in float v;

            layout(set = 0, binding = 0) uniform Data {
                float view;
            } uniforms;

            layout(location = 0) out vec4 f_color;
    
            void main() {
                float dist = normalize(gl_FragCoord.xy).y;
                // f_color = vec4(1.0, dist, 0.0, 0.2);
                // f_color = vec4(sin(v), dist, 0.0, 1);
                f_color = vec4(color.x + sin(uniforms.view), color.y, color.z + cos(uniforms.view), dist);
            }
        ",
    );
}

// fn create_buffer_from_iter<T, I>(
//     allocator: GenericBufferAllocator,
//     iter: I,
//     buffer_usage: BufferUsage,
//     memory_type_filter: MemoryTypeFilter,
// ) -> Subbuffer<[T]>
// where
//     T: BufferContents,
//     I: IntoIterator<Item = T>,
//     I::IntoIter: ExactSizeIterator,
// {
//     Buffer::from_iter(
//         allocator.clone(),
//         BufferCreateInfo {
//             usage: buffer_usage,
//             ..Default::default()
//         },
//         AllocationCreateInfo {
//             memory_type_filter: memory_type_filter,
//             ..Default::default()
//         },
//         iter,
//     )
//     .unwrap()
// }


pub fn get_vertex_shader (device: Arc<Device>,) ->  Arc<ShaderModule> {
    vs::load(device).expect("Failed to load vertex shader")
}

pub fn get_fragment_shader (device: Arc<Device>) -> Arc<ShaderModule> {
    fs::load(device).expect("Failed to load vertex shader")
}
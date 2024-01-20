use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferUsage},
    device::Device,
    memory::allocator::MemoryTypeFilter,
    shader::ShaderModule,
};

use crate::{graphics_pack::buffers, GenericBufferAllocator};

struct VertexShader {
    src: String,
}

struct FragmentShader {
    src: String,
}

pub mod vs {
    vulkano_shaders::shader!(
        ty: "vertex",
        src: r"
            #version 460


            layout(push_constant) uniform PushConstantData {
                float view;
            } pc;

            // layout(set = 1, binding = 0) uniform Data {
            //     float view;
            // } uniforms;

            
            layout(set = 1, binding = 2) uniform MvpMatrix {
                mat4 model;
                mat4 view;
                mat4 projection;
            } mvp;
    
    
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;
            layout(location = 2) in vec2 tex_coord;

            layout(location = 3) in vec3 global_position;
            layout(location = 4) in float local_scale;

            layout(location = 0) out vec3 out_color;
            layout(location = 1) out float v;
            layout(location = 2) out vec2 textureMapping;
            
            void main() {
                vec4 player_position = mvp.model * vec4(position.xyz * local_scale + global_position, 1.0);
                // player_position +=;

                // gl_Position =  mvp.projection * mvp.view * mvp.model * vec4(position.xyz, 1.0);

                gl_Position = mvp.projection * mvp.view * player_position;
                out_color = color;
                
                // v = uniforms.view;
                v = pc.view;
                textureMapping = tex_coord;
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
            
            layout(push_constant) uniform PushConstantData {
                float view;
            } pc;

            layout(set = 1, binding = 0) uniform sampler s; 
            layout(set = 1, binding = 1) uniform texture2D tex;

            layout(set = 1, binding = 2) uniform MvpMatrix {
                mat4 model;
                mat4 view;
                mat4 projection;
            } mvp;

            layout(location = 0) out vec4 f_color;
    
            void main() {
                float dist = normalize(gl_FragCoord.xy).y;
                // f_color = vec4(1.0, dist, 0.0, 0.2);
                // f_color = vec4(sin(v), dist, 0.0, 1);
                f_color = texture(sampler2D(tex, s), tex_coord) + vec4(color.x + sin(pc.view), color.y, color.z + cos(pc.view), dist);
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

pub fn get_vertex_shader(device: Arc<Device>) -> Arc<ShaderModule> {
    vs::load(device).expect("Failed to load vertex shader")
}

pub fn get_fragment_shader(device: Arc<Device>) -> Arc<ShaderModule> {
    fs::load(device).expect("Failed to load vertex shader")
}

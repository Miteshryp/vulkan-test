use std::sync::Arc;

use vulkano::{buffer::BufferContents, command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer}, descriptor_set::allocator::StandardDescriptorSetAllocator, memory::allocator::GenericMemoryAllocator, pipeline::graphics::vertex_input::Vertex};


// Buffer type declarations
pub type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
pub type DescriptorSetAllocator = Arc<StandardDescriptorSetAllocator>;


// Command builder types
pub type CommandBufferType = Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

pub type PrimaryAutoCommandBuilderType = AutoCommandBufferBuilder<
    PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
    Arc<StandardCommandBufferAllocator>,
>;


#[derive(BufferContents, Clone, Vertex, Debug)]
#[repr(C)]
pub struct VertexData {
    #[format(R32G32B32_SFLOAT)]
    pub position: Vec3,

    #[format(R32G32B32_SFLOAT)]
    pub color: Vec3,

    #[format(R32G32_SFLOAT)]
    pub tex_coord: Vec2,   
}

#[derive(BufferContents, Vertex, Debug, Clone)]
#[repr(C)]
pub struct InstanceData {
    #[format(R32G32B32_SFLOAT)]
    pub global_position: Vec3,

    #[format(R32_SFLOAT)]
    pub local_scale: f32,

    #[format(R32_UINT)]
    pub tex_index: u32,

}

#[derive(BufferContents, Clone, Debug)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(BufferContents, Clone, Debug)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

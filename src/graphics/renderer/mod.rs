use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator, device::Device,
    render_pass::{Framebuffer, RenderPass}, sync::GpuFuture,
};

use super::{
    buffers::{base_buffer::DeviceBuffer, primitives::{self, InstanceData}, VertexData},
    components::vulkan::VulkanSwapchainInfo,
};

pub mod basic_renderer;
pub mod deferred_renderer;

trait VulkanRenderpassBuilder {
    fn create_render_pass(
        device: Arc<Device>,
        swapchain_info: &VulkanSwapchainInfo,
    ) -> Arc<RenderPass>;
}

pub trait RendererInterface {
    fn bind_vertex_buffer(&mut self, buffer: DeviceBuffer<VertexData>);
    fn bind_instance_buffer(&mut self, buffer: DeviceBuffer<InstanceData>);
    fn bind_index_buffer(&mut self, buffer: DeviceBuffer<u32>);

    fn render(
        &mut self,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue_family_index: u32,
        fbo: Arc<Framebuffer>
    ) -> primitives::CommandBufferType;
}

pub trait VulkanRendererInterface {
    fn new() -> Self;
    // fn build_cm
}

use std::sync::Arc;

use vulkano::{device::Device, render_pass::RenderPass};

use super::components::vulkan::VulkanSwapchainInfo;

pub mod basic_renderer;
pub mod deferred_renderer;

trait VulkanRenderpassBuilder {
    fn create_render_pass(
        device: Arc<Device>,
        swapchain_info: &VulkanSwapchainInfo
    ) -> Arc<RenderPass>;
}

pub trait VulkanRendererInterface {
    fn new() -> Self;
    // fn build_cm
}
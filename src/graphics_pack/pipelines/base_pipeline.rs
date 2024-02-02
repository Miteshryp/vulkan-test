use std::sync::Arc;

use vulkano::{device::Device, render_pass::RenderPass};
use winit::window::Window;

use crate::graphics_pack::components::vulkan::VulkanSwapchainInfo;

pub trait GraphicsPipelineBuilder {
    type NewPipeline;

    fn new(
        window: Arc<Window>,
        logical_device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        // swapchain_info: &VulkanSwapchainInfo,
        subpass_index: u32
    ) -> Self::NewPipeline;
}
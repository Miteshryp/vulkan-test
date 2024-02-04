use std::sync::Arc;

use vulkano::{device::Device, pipeline::GraphicsPipeline, render_pass::RenderPass};
use winit::window::Window;

use crate::graphics::components::vulkan::VulkanSwapchainInfo;

pub trait GraphicsPipelineInterface {
    // type NewPipeline;

    // fn new(
    //     window: Arc<Window>,
    //     logical_device: Arc<Device>,
    //     render_pass: Arc<RenderPass>,
    //     // swapchain_info: &VulkanSwapchainInfo,
    //     subpass_index: u32,
    // ) -> Self::NewPipeline;

    fn get_push_descriptor_set_index(&self) -> u32;
    fn get_attachment_descriptor_set_index(&self) -> Option<u32>;
    fn get_pipeline(&self) -> Arc<GraphicsPipeline>;
    // fn set_push_descriptor_set_index(&mut self);
}

// pub(crate) trait InitializePipeline {
    // fn create_pipeline(
    //     logical_device: Arc<Device>,
    //     window: Arc<Window>,
    //     render_pass: Arc<RenderPass>,
    //     subpass_index: u32,
    //     push_descriptor_set_index: u32,
    //     attachment_descriptor_set_index: Option<u32>
    // ) -> Arc<GraphicsPipeline>;
// }

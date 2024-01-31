use std::sync::Arc;

use vulkano::{device::Device, render_pass::RenderPass};
use winit::window::Window;

pub trait GraphicsPipelineBuilder {
    type NewPipeline;

    fn new(
        window: Arc<Window>,
        logical_device: Arc<Device>,
        render_pass: Arc<RenderPass>
    ) -> Self::NewPipeline;
}
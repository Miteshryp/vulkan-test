use std::sync::Arc;

use winit::window::Window;

use super::{components::vulkan::VulkanInstanceState, renderer::VulkanRenderer};


struct VulkanAPI {
    state: VulkanInstanceState,
    renderers: Vec<Box<dyn VulkanRenderer>>
}

impl VulkanAPI {
    pub fn initialize_vulkan(window: Arc<Window>) {

    }
}
use std::sync::Arc;
use vulkano::{device::Device, render_pass::RenderPass};

use crate::graphics::components::vulkan::VulkanSwapchainInfo;

pub fn create_render_pass(
    device: Arc<Device>,
    swapchain_info: &VulkanSwapchainInfo,
) -> Arc<RenderPass> {
    // need 3 things: device Arc, attachments, and a pass
    let format = swapchain_info.swapchain.create_info().image_format.clone();

    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },

            depth: {
                format: vulkano::format::Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare
            }
        },
        pass: {
            color: [color],
            // depth_stencil: {}
            depth_stencil: {depth},
        },
    )
    .unwrap()
}
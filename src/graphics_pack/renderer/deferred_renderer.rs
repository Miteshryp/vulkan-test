use std::sync::Arc;

use vulkano::{
    command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage}, device::Device, ordered_passes_renderpass, render_pass::{RenderPass, Subpass}
};
use winit::window::Window;

use crate::graphics_pack::{buffers::primitives, components::vulkan::{VulkanInstance, VulkanSwapchainInfo}, pipelines::{base_pipeline::GraphicsPipelineBuilder, deferred_pipeline::DeferredPipeline, lighting_pipeline::LightingPipeline}};

use super::VulkanRenderpassBuilder;

struct DeferredRenderer {
    render_pass: Arc<RenderPass>,
    // subpasses: Vec<Arc<Subpass>>, // This has to be in right order
    // deferred_subpass: Arc<Subpass>,
    // lighting_subpass: Arc<Subpass>,


    deferred_pipeline: DeferredPipeline,
    lighting_pipeline: LightingPipeline,

    // instance: &VulkanInstance,

    command_builder: primitives::PrimaryAutoCommandBuilderType,
}

// Things we need for a renderer
// 1. Uniforms to be passed
// 2. Command builder
// 3. Buffer bindings (vertex, index[optional] and instance[optional])
// 4. Attachment buffers (image object corresponding to attachment buffers)

impl super::VulkanRenderpassBuilder for DeferredRenderer {
    fn create_render_pass(
        device: Arc<Device>,
        swapchain_info: &VulkanSwapchainInfo,
    ) -> Arc<RenderPass> {
        let format = swapchain_info.swapchain.create_info().image_format.clone();

        vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {

                // each attachment has to be defined in this EXACT format order:
                // format, samples, load_op, store_op
                final_color: {
                    format: format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                color: {
                    format: vulkano::format::Format::A2B10G10R10_UNORM_PACK32,
                    // format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },

                normals: {
                    format: vulkano::format::Format::R16G16B16A16_SFLOAT,
                    // format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },

                depth: {
                    format: vulkano::format::Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare
                }
            },

            passes: [
                {
                    color: [color, normals],
                    depth_stencil: {depth},
                    input: []
                },
                {
                    color: [final_color],
                    depth_stencil: {}, // depth calculations were done by the previous pass
                    input: [color, normals]
                }
            ]
        )
        .unwrap()
    }
}

impl DeferredRenderer {
    pub fn new(
        window: Arc<Window>,
        device: Arc<Device>,
        instance: &VulkanInstance
        // command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        // swapchain_info: &VulkanSwapchainInfo,
        // queue_family_index: u32
    ) -> Self {
        let render_pass = DeferredRenderer::create_render_pass(device.clone(), &instance.swapchain_info);
        // let deferred_subpass = Subpass::from(render_pass.clone(), 0);

        let deferred_pipeline = DeferredPipeline::new(
            window.clone(), 
            device.clone(), 
            render_pass.clone(), 
            0
        );

        let lighting_pipeline = LightingPipeline::new(
            window.clone(), 
            device.clone(), 
            render_pass.clone(), 
            1
        );

        let mut command_builder = AutoCommandBufferBuilder::primary(
            &instance.allocators.command_buffer_allocator,
            instance.get_first_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        Self {
            render_pass: render_pass,
            deferred_pipeline,
            lighting_pipeline,
            command_builder   
        }
    }


    // pub fn add_vertex_buffer(StagingB)
}

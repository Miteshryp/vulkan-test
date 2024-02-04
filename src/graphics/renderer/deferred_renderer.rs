use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo
    }, device::Device, format::{ClearValue, Format}, image::{view::ImageView, Image, ImageCreateInfo, ImageUsage}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}, ordered_passes_renderpass, pipeline::{graphics::vertex_input::VertexBuffersCollection, GraphicsPipeline, Pipeline}, render_pass::{Framebuffer, RenderPass, Subpass}, sync::{self, GpuFuture}
};
use winit::window::Window;

use crate::graphics::{
    buffers::{
        base_buffer::DeviceBuffer,
        primitives::{self, InstanceData, PrimaryAutoCommandBuilderType, VertexData},
    },
    components::vulkan::{VulkanInstance, VulkanSwapchainInfo},
    pipelines::{
        base_pipeline::GraphicsPipelineInterface, deferred_pipeline::DeferredPipeline,
        lighting_pipeline::LightingPipeline,
    },
};

use super::{RendererInterface, VulkanRenderpassBuilder};

pub struct DeferredRendererAttachments {
    depth_stencil: Arc<ImageView>,
    normals: Arc<ImageView>,
    color: Arc<ImageView>,
}

pub struct DeferredRenderer {
    render_pass: Arc<RenderPass>,

    deferred_pipeline: DeferredPipeline,
    lighting_pipeline: LightingPipeline,

    // Attachments
    attachments: DeferredRendererAttachments,

    // Buffers
    vertex_buffer: Option<DeviceBuffer<VertexData>>,
    instance_buffer: Option<DeviceBuffer<InstanceData>>,
    index_buffer: Option<DeviceBuffer<u32>>,
}

// Things we need for a renderer
// 1. Uniforms to be passed
// 2. Buffer bindings (vertex, index[optional] and instance[optional])
// 3. Attachment buffers (image object corresponding to attachment buffers)

impl super::VulkanRenderpassBuilder for DeferredRenderer {
    // The Render pass is being created in the renderer solely for organizational purpose.
    // A render pass differs if the pipeline changes, and since a single renderer
    // is going to have a fixed pipeline structure, creating the renderpass here allows
    // for better visibility as to what is actually happening in the rendering process.
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

impl RendererInterface for DeferredRenderer {
    fn bind_vertex_buffer(&mut self, buffer: DeviceBuffer<VertexData>) {
        self.vertex_buffer = Some(buffer);
    }

    fn bind_instance_buffer(&mut self, buffer: DeviceBuffer<InstanceData>) {
        self.instance_buffer = Some(buffer)
    }

    fn bind_index_buffer(&mut self, buffer: DeviceBuffer<u32>) {
        self.index_buffer = Some(buffer)
    }

    fn render(
        &mut self,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue_family_index: u32,
        fbo: Arc<Framebuffer>
    ) -> primitives::CommandBufferType {
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Necessary Descriptor attachments (render pass attachments + other essential persistent descriptor attachments)

        // Push descriptor attachments (attachments for pushing frequently changing user data)

        builder.begin_render_pass(RenderPassBeginInfo {
            clear_values: Self::get_clear_values(),
            ..RenderPassBeginInfo::framebuffer(fbo)
        }, SubpassBeginInfo {
            contents: vulkano::command_buffer::SubpassContents::Inline,
            ..Default::default()
        }).unwrap();
        

        self.bind_buffers(&mut builder);
        self.draw_call(&mut builder);

        // Draw call

        builder.build().unwrap()
    }

    // Helper functions
}

impl DeferredRenderer {
    pub fn new(
        window: Arc<Window>,
        device: Arc<Device>,
        instance: &VulkanInstance,
    ) -> Self {
        let render_pass =
            DeferredRenderer::create_render_pass(device.clone(), &instance.swapchain_info);

        let attachments = Self::get_attachment_image_views(window.clone(), instance.allocators.memory_allocator.clone());

        let deferred_pipeline =
            DeferredPipeline::new(window.clone(), device.clone(), render_pass.clone(), 0);

        let lighting_pipeline =
            LightingPipeline::new(window.clone(), device.clone(), render_pass.clone(), 1);

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
            // command_builder
            vertex_buffer: None,
            instance_buffer: None,
            index_buffer: None,

            attachments
        }
    }


    fn get_attachment_image_views(window: Arc<Window>, allocator: primitives::GenericBufferAllocator) -> DeferredRendererAttachments {
        let depth_image_view = ImageView::new_default(Image::new(
            allocator.clone(),
            ImageCreateInfo {
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                stencil_usage: Some(ImageUsage::DEPTH_STENCIL_ATTACHMENT),
                format: Format::D16_UNORM,
                extent: [window.inner_size().width, window.inner_size().height, 1],
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        ).unwrap())
        .unwrap();

        let normal_image_view = ImageView::new_default(Image::new(
            allocator.clone(),
            ImageCreateInfo {
                usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                extent: [window.inner_size().width, window.inner_size().height, 1],
                format: Format::R16G16B16A16_SFLOAT,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap()).unwrap();

        let color_image_view = ImageView::new_default(Image::new(
            allocator.clone(),
            ImageCreateInfo {
                usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                extent: [window.inner_size().width, window.inner_size().height, 1],
                format: Format::A2B10G10R10_UNORM_PACK32,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap()).unwrap();


        DeferredRendererAttachments {
            depth_stencil: depth_image_view,
            color: color_image_view,
            normals: normal_image_view
        }
    }

    fn get_clear_values() -> Vec<Option<ClearValue>> {
        // This is the same order as passed in the attachments field 
        // of the ordered_passes_renderpass! macro
        vec![
            Some([0.0, 0.0, 0.0, 1.0].into()), // final_color
            Some([0.0, 0.0, 0.0, 1.0].into()), // color
            Some([0.0, 0.0, 0.0, 1.0].into()), // normals
            Some(1.0.into()), // depth
        ]
    }

    // fn bind_essential_attachments(&self, builder: &mut primitives::PrimaryAutoCommandBuilderType) {
    //     builder.bind_descriptor_sets(
    //         pipeline_bind_point,
    //         pipeline_layout,
    //         first_set,
    //         descriptor_sets
    //     )
    // }

    // fn bind_push_descriptors(&self, builder: &mut primitives::PrimaryAutoCommandBuilderType, pipeline: impl GraphicsPipelineInterface) {

    //     // let descriptor_writes = self.push_


    //     builder.push_descriptor_set(
    //         pipeline.get_pipeline().bind_point(), 
    //         pipeline.get_pipeline().layout().clone(), 
    //         pipeline.get_push_descriptor_set_index(),
    //         descriptor_writes
    //     ).unwrap();
    // }

    fn bind_buffers(&self, builder: &mut primitives::PrimaryAutoCommandBuilderType) {
        // Vertex buffer binding
        if let Some(vertex_buffer) = self.vertex_buffer.clone() {
            if let Some(instance_buffer) = self.instance_buffer.clone() {
                builder
                    .bind_vertex_buffers(
                        0,
                        (vertex_buffer.buffer.clone(), instance_buffer.buffer.clone()),
                    )
                    .unwrap();
            } else {
                builder
                    .bind_vertex_buffers(0, vertex_buffer.buffer.clone())
                    .unwrap();
            }
        }

        // Index buffer binding
        if let Some(index_buffer) = self.index_buffer.clone() {
            builder.bind_index_buffer(index_buffer.buffer.clone());
        }
    }

    fn draw_call(&self, builder: &mut primitives::PrimaryAutoCommandBuilderType) {
        let render_primitive_index_count = 3;
        builder
            .draw_indexed(
                self.index_buffer.clone().unwrap().count,
                self.index_buffer.clone().unwrap().count / render_primitive_index_count,
                0,
                0,
                0,
            )
            .unwrap();
    }

    // pub fn add_vertex_buffer(StagingB)
}

use std::{rc::Rc, sync::Arc};

use smallvec::smallvec;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    device::Device,
    format::{ClearValue, Format},
    image::{
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    ordered_passes_renderpass,
    pipeline::{
        graphics::vertex_input::VertexBuffersCollection, GraphicsPipeline, Pipeline,
        PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::{self, GpuFuture},
};
use winit::window::Window;

use crate::graphics::{
    buffers::{
        base_buffer::DeviceBuffer,
        primitives::{
            self, GenericBufferAllocator, InstanceData, PrimaryAutoCommandBuilderType, VertexData,
        },
        uniform_buffer::{UniformBuffer, UniformSet},
    },
    components::{
        camera::Camera,
        vulkan::{VulkanInstance, VulkanSwapchainInfo},
    },
    pipelines::{
        base_pipeline::GraphicsPipelineInterface, deferred_pipeline::DeferredPipeline,
        lighting_pipeline::LightingPipeline,
    },
    shaders,
};

use super::{RendererInterface, VulkanRenderpassBuilder};
use nalgebra_glm as glm;

pub struct DeferredRendererData<'a> {
    pub camera: &'a Camera,
    pub image_array_view: Arc<ImageView>,
}
struct DeferredRendererDescriptorData {
    // Necessary Descriptors
    sampler: Arc<Sampler>,
}

struct DeferredRendererAttachments {
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
    internal_data: DeferredRendererDescriptorData,
    frame_buffers: Vec<Arc<Framebuffer>>,

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

impl DeferredRenderer {
    pub fn new(
        device: Arc<Device>,
        window: Arc<Window>,
        swapchain_info: &VulkanSwapchainInfo,
        buffer_allocator: GenericBufferAllocator,
    ) -> Self {
        // Render pass
        let render_pass =
            DeferredRenderer::create_render_pass(device.clone(), swapchain_info);

        // Data required in the renderer
        let attachments = Self::get_attachment_image_views(
            window.clone(),
            buffer_allocator.clone(),
        );
        let image_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::Repeat; 3],
                min_filter: vulkano::image::sampler::Filter::Linear,
                mag_filter: vulkano::image::sampler::Filter::Linear,
                ..Default::default()
            },
        )
        .unwrap();

        // Pipelines
        let deferred_pipeline =
            DeferredPipeline::new(window.clone(), device.clone(), render_pass.clone(), 0);

        let lighting_pipeline =
            LightingPipeline::new(window.clone(), device.clone(), render_pass.clone(), 1);

        let frame_buffers = Self::create_framebuffer(
            render_pass.clone(),
            &attachments,
            swapchain_info, 
        );

        Self {
            render_pass: render_pass,
            deferred_pipeline,
            lighting_pipeline,

            vertex_buffer: None,
            instance_buffer: None,
            index_buffer: None,

            attachments,
            internal_data: DeferredRendererDescriptorData {
                sampler: image_sampler,
            },

            frame_buffers
        }

    }

    pub fn bind_vertex_buffer(&mut self, buffer: DeviceBuffer<VertexData>) {
        let _ = std::mem::replace(&mut self.vertex_buffer, Some(buffer));
    }

    pub fn bind_instance_buffer(&mut self, buffer: DeviceBuffer<InstanceData>) {
        let _ = std::mem::replace(&mut self.instance_buffer, Some(buffer));
    }

    pub fn bind_index_buffer(&mut self, buffer: DeviceBuffer<u32>) {
        let _ = std::mem::replace(&mut self.index_buffer, Some(buffer));
    }

    pub fn render(
        &mut self,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: GenericBufferAllocator,
        descriptor_set_allocator: primitives::DescriptorSetAllocator,
        queue_family_index: u32,
        data: DeferredRendererData,
    ) -> Vec<primitives::CommandBufferType> {
        self.frame_buffers
            .iter()
            .map(|fb| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue_family_index,
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: Self::get_clear_values(),
                            ..RenderPassBeginInfo::framebuffer(fb.clone())
                        },
                        SubpassBeginInfo {
                            contents: vulkano::command_buffer::SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap();

                self.deferred_subpass(&mut builder, memory_allocator.clone(), &data);
                Self::next_inline_subpass(&mut builder);
                self.lighting_subpass(
                    &mut builder,
                    memory_allocator.clone(),
                    descriptor_set_allocator.clone(),
                    &data,
                );

                builder.end_render_pass(Default::default()).unwrap();

                builder.build().unwrap()
            })
            .collect()
    }


    // Every time a swapchain is invalidated, these steps need to be carried out to reconfigure the command buffers.
    pub fn refresh_render_target(
        &mut self,
        device: Arc<Device>,
        window: Arc<Window>,
        swapchain_info: &VulkanSwapchainInfo,
        buffer_allocator: GenericBufferAllocator,
    ) {
        self.render_pass = Self::create_render_pass(device, swapchain_info);
        self.attachments = Self::get_attachment_image_views(window.clone(), buffer_allocator.clone());
        self.frame_buffers = Self::create_framebuffer( self.render_pass.clone(), &self.attachments, swapchain_info);
    }
}

// Helper functions
impl DeferredRenderer {
    fn create_framebuffer(
        render_pass: Arc<RenderPass>,
        attachments: &DeferredRendererAttachments,
        swapchain_info: &VulkanSwapchainInfo,
    ) -> Vec<Arc<Framebuffer>> {
        // self.attachments = Self::get_attachment_image_views(window.clone(), buffer_allocator.clone());

        swapchain_info
            .images
            .iter()
            .map(|img| {
                let final_image_view = ImageView::new_default(img.clone()).unwrap();

                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![
                            final_image_view,
                            attachments.color.clone(),
                            attachments.normals.clone(),
                            attachments.depth_stencil.clone(),
                        ],
                        // extent: [window.inner_size().into(),
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    }

    fn get_attachment_image_views(
        window: Arc<Window>,
        allocator: primitives::GenericBufferAllocator,
    ) -> DeferredRendererAttachments {
        let depth_image_view = ImageView::new_default(
            Image::new(
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
            )
            .unwrap(),
        )
        .unwrap();

        let normal_image_view = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    usage: ImageUsage::TRANSIENT_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::COLOR_ATTACHMENT,
                    extent: [window.inner_size().width, window.inner_size().height, 1],
                    format: Format::R16G16B16A16_SFLOAT,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let color_image_view = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    usage: ImageUsage::TRANSIENT_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::COLOR_ATTACHMENT,
                    extent: [window.inner_size().width, window.inner_size().height, 1],
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        DeferredRendererAttachments {
            depth_stencil: depth_image_view,
            color: color_image_view,
            normals: normal_image_view,
        }
    }

    fn get_clear_values() -> Vec<Option<ClearValue>> {
        // This is the same order as passed in the attachments field
        // of the ordered_passes_renderpass! macro
        vec![
            Some([0.0, 0.0, 0.0, 1.0].into()), // final_color
            Some([0.0, 0.0, 0.0, 1.0].into()), // color
            Some([0.0, 0.0, 0.0, 1.0].into()), // normals
            Some(1.0.into()),                  // depth
        ]
    }

    fn bind_deferred_pipeline_descriptor(
        &self,
        builder: &mut primitives::PrimaryAutoCommandBuilderType,
        allocator: primitives::GenericBufferAllocator,
        user_data: &DeferredRendererData,
    ) {
        let view = user_data.camera.get_view_matrix_data();
        let projection = user_data.camera.get_projection_matrix_data();

        let push_descriptor_writes = smallvec![
            UniformBuffer::create_buffer(
                allocator.clone(),
                // self.deferred_pipeline.get_push_descriptor_set_index(),
                0,
                shaders::deferred::vs::MvpMatrix {
                    // model: Into::<[[f32; 4]; 4]>::into(glm::identity::<f32, 4>()),
                    view: Into::<[[f32; 4]; 4]>::into(view),
                    projection: Into::<[[f32; 4]; 4]>::into(projection),
                },
                Default::default()
            )
            .get_write_descriptor(),
            UniformBuffer::create_immutable_sampler(1).get_write_descriptor(),
            UniformBuffer::create_image_view(2, user_data.image_array_view.clone())
                .get_write_descriptor(),
        ];

        builder
            .push_descriptor_set(
                self.deferred_pipeline.pipeline.bind_point(),
                self.deferred_pipeline.pipeline.layout().clone(),
                self.deferred_pipeline.get_push_descriptor_set_index(),
                push_descriptor_writes,
            )
            .unwrap();
    }

    fn bind_lighting_pipeline_descriptor(
        &self,
        builder: &mut primitives::PrimaryAutoCommandBuilderType,
        memory_allocator: primitives::GenericBufferAllocator,
        descriptor_set_allocator: primitives::DescriptorSetAllocator,
        user_data: &DeferredRendererData,
    ) {
        // User data
        let view = user_data.camera.get_view_matrix_data();
        let projection = user_data.camera.get_projection_matrix_data();

        let push_descriptor_writes = smallvec![UniformBuffer::create_buffer(
            memory_allocator.clone(),
            0,
            shaders::lighting::vs::MvpMatrix {
                // model: Into::<[[f32; 4]; 4]>::into(glm::identity::<f32, 4>()),
                view: Into::<[[f32; 4]; 4]>::into(view),
                projection: Into::<[[f32; 4]; 4]>::into(projection),
            },
            Default::default()
        )
        .get_write_descriptor()];

        builder
            .push_descriptor_set(
                self.lighting_pipeline.pipeline.bind_point(),
                self.lighting_pipeline.pipeline.layout().clone(),
                self.lighting_pipeline.get_push_descriptor_set_index(),
                push_descriptor_writes,
            )
            .unwrap();

        // Descriptor writes
        let mut attachment_descriptor_set = UniformSet::new(
            self.lighting_pipeline
                .get_attachment_descriptor_set_index()
                .unwrap() as usize,
        );

        attachment_descriptor_set.add_uniform_buffer(UniformBuffer::create_image_view(
            0,
            self.attachments.color.clone(),
        ));
        attachment_descriptor_set.add_uniform_buffer(UniformBuffer::create_image_view(
            1,
            self.attachments.normals.clone(),
        ));

        builder
            .bind_descriptor_sets(
                self.lighting_pipeline.pipeline.bind_point(),
                self.lighting_pipeline.pipeline.layout().clone(),
                self.lighting_pipeline
                    .get_attachment_descriptor_set_index()
                    .unwrap(),
                // 0,
                vec![attachment_descriptor_set.get_persistent_descriptor_set(
                    descriptor_set_allocator,
                    self.lighting_pipeline.pipeline.clone(),
                )],
            )
            .unwrap();
    }

    fn deferred_subpass(
        &self,
        mut builder: &mut primitives::PrimaryAutoCommandBuilderType,
        memory_allocator: GenericBufferAllocator,
        user_data: &DeferredRendererData,
    ) {
        // Deferred subpass
        builder
            .bind_pipeline_graphics(self.deferred_pipeline.pipeline.clone())
            .unwrap();

        self.bind_buffers(&mut builder);
        self.bind_deferred_pipeline_descriptor(&mut builder, memory_allocator, &user_data);
        self.draw_call(&mut builder);
    }

    fn lighting_subpass(
        &self,
        mut builder: &mut primitives::PrimaryAutoCommandBuilderType,
        memory_allocator: GenericBufferAllocator,
        descriptor_set_allocator: primitives::DescriptorSetAllocator,
        user_data: &DeferredRendererData,
    ) {
        // Lighting subpass
        builder
            .bind_pipeline_graphics(self.lighting_pipeline.pipeline.clone())
            .unwrap();

        self.bind_buffers(&mut builder);
        self.bind_lighting_pipeline_descriptor(
            &mut builder,
            memory_allocator,
            descriptor_set_allocator,
            &user_data,
        );
        self.draw_call(&mut builder);
    }

    fn next_inline_subpass(builder: &mut primitives::PrimaryAutoCommandBuilderType) {
        builder
            .next_subpass(
                Default::default(),
                SubpassBeginInfo {
                    contents: vulkano::command_buffer::SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();
    }

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
            builder
                .bind_index_buffer(index_buffer.buffer.clone())
                .unwrap();
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

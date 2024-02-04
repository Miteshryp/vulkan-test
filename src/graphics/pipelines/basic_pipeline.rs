use std::sync::Arc;
use vulkano::{
    descriptor_set::layout::DescriptorSetLayoutCreateFlags,
    device::Device,
    image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
    pipeline::{
        graphics::{
            self,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{self, DepthStencilState},
            input_assembly::InputAssemblyState,
            rasterization::{FrontFace, RasterizationState},
            vertex_input::{Vertex, VertexBufferDescription, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
    shader::{self, EntryPoint},
};
use winit::window::Window;

use crate::graphics::{
    buffers::primitives::*, components::vulkan::VulkanSwapchainInfo, pipelines::base_pipeline,
    shaders,
};

use super::{
    base_pipeline::{GraphicsPipelineInterface},
    renderpass::create_render_pass,
};

pub struct BasicPipeline {
    pub pipeline: Arc<GraphicsPipeline>,
    push_descriptor_set_index: u32,
}

impl BasicPipeline {

    pub fn new(
        window: Arc<Window>,
        logical_device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        // swapchain_info: &VulkanSwapchainInfo,
        subpass_index: u32,
    ) -> BasicPipeline {
        let push_descriptor_index = 1;
    
        BasicPipeline {
            pipeline: Self::create_pipeline(
                logical_device,
                window,
                render_pass,
                subpass_index,
                push_descriptor_index,
            ),
            push_descriptor_set_index: push_descriptor_index,
        }
    }

    fn create_pipeline(
        logical_device: Arc<Device>,
        window: Arc<Window>,
        render_pass: Arc<RenderPass>,
        subpass_index: u32,
        push_descriptor_set_index: u32,
    ) -> Arc<GraphicsPipeline> {
        // let render_pass = create_render_pass(logical_device.clone(), swapchain_info);
        let subpass = Subpass::from(render_pass.clone(), subpass_index).unwrap();

        let vertex_shader: EntryPoint = shaders::basic::load_vertex_shader(logical_device.clone())
            .entry_point("main")
            .unwrap();

        // println!("Vertex Shader entry point info: {:?}", vertex_shader.info());

        let fragment_shader = shaders::basic::load_fragment_shader(logical_device.clone())
            .entry_point("main")
            .unwrap();

        let vertex_shader_input_state = [VertexData::per_vertex(), InstanceData::per_instance()]
            .definition(&vertex_shader.info().input_interface)
            .unwrap();

        // This creation moves the vertex and fragment shaders,
        // so we cannot use those objects after this point
        let pipeline_stages = [
            PipelineShaderStageCreateInfo::new(vertex_shader),
            PipelineShaderStageCreateInfo::new(fragment_shader),
        ];

        let mut descriptor_set_layout =
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stages);

        // Enabling descriptor pushes on set 0
        let set_layout = &mut descriptor_set_layout.set_layouts[push_descriptor_set_index as usize];
        set_layout.flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;
        set_layout.bindings.get_mut(&0).unwrap().immutable_samplers = vec![Sampler::new(
            logical_device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::Repeat; 3],
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                ..Default::default()
            },
        )
        .unwrap()];

        let descriptor_create_info = descriptor_set_layout
            .into_pipeline_layout_create_info(logical_device.clone())
            .unwrap();

        // ERROR: from_stages is setting descriptor set count to 1
        let pipeline_layout =
            PipelineLayout::new(logical_device.clone(), descriptor_create_info).unwrap();

        let depth_stencil = depth_stencil::DepthState::simple();
        // let rasterization_state_info: RasterizationState = ;

        GraphicsPipeline::new(
            logical_device,
            None,
            GraphicsPipelineCreateInfo {
                // Defining the stages of the pipeline
                // we only have 2 => vertex shader and a fragment shader
                stages: pipeline_stages.into_iter().collect(),

                // Defining the mapping of vertex to the vertex shader inputs
                vertex_input_state: Some(vertex_shader_input_state),

                // Setting a fixed viewport for now
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0, 0.0],
                        extent: window.inner_size().into(),
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                }),

                // describes drawing primitives, default is a triangle
                input_assembly_state: Some(InputAssemblyState {
                    topology: graphics::input_assembly::PrimitiveTopology::TriangleList,
                    ..Default::default()
                }),

                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(depth_stencil),
                    ..Default::default()
                }),

                // rasterization_state: Some(Default::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: graphics::rasterization::CullMode::Back,
                    // front_face: FrontFace::CounterClockwise,
                    front_face: FrontFace::Clockwise,
                    ..Default::default()
                }),
                multisample_state: Some(Default::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),

                // Concerns with First pass of render pass
                subpass: Some(subpass.into()),

                ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
            },
        )
        .unwrap()
    }
}

impl base_pipeline::GraphicsPipelineInterface for BasicPipeline {


    fn get_push_descriptor_set_index(&self) -> u32 {
        self.push_descriptor_set_index
    }

    fn get_attachment_descriptor_set_index(&self) -> Option<u32> {
        None
    }

    fn get_pipeline(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }
}

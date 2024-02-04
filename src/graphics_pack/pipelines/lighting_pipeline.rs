use std::sync::Arc;

use vulkano::{descriptor_set::layout::DescriptorSetLayoutCreateFlags, pipeline::{graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState, vertex_input::{Vertex, VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo}, layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo}, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo}, render_pass::{RenderPass, Subpass}, sync::PipelineStages};

use crate::graphics_pack::{buffers::{primitives::InstanceData, VertexData}, shaders};

use super::{base_pipeline::{GraphicsPipelineBuilder, InitializePipeline}, renderpass};

#[derive(Clone)]
pub struct LightingPipeline {
    pub pipeline: Arc<GraphicsPipeline>,
    push_descriptor_set_index: u32,
    attachment_descriptor_set_index: u32
}

impl InitializePipeline for LightingPipeline {
    fn create_pipeline(
        logical: Arc<vulkano::device::Device>,
        window: Arc<winit::window::Window>,
        render_pass: Arc<RenderPass>,
        subpass_index: u32,
        push_descriptor_set_index: u32,
        attachment_descriptor_set_index: Option<u32>,
    ) -> Arc<GraphicsPipeline> {

         
        let subpass = Subpass::from(render_pass.clone(), subpass_index).unwrap();

        // load vertex and fragment shader
        let vertex_shader = shaders::lighting::load_vertex_shader(logical.clone())
            .entry_point("main")
            .unwrap();
        let fragment_shader = shaders::lighting::load_fragment_shader(logical.clone())
            .entry_point("main")
            .unwrap();

        // TODO: Maybe not passing this into the pipeline object can fail
        let vertex_shader_input_state = [VertexData::per_vertex(), InstanceData::per_instance()].definition(&vertex_shader.info().input_interface).unwrap();

        let pipeline_stages = [
            PipelineShaderStageCreateInfo::new(vertex_shader),
            PipelineShaderStageCreateInfo::new(fragment_shader)
        ];


        // create vertex shader input state for vertex_input_state field (maybe not needed if there is no data being passed in from the user)
        // create a pipeline stages object
        // get the descriptor set layout from pipeline stages
        // convert descriptor set layout into graphics pipeline layout
        // pass it as the layout of the pipeline

        // Creating pipeline with push descriptor configured for MVP matrix
        let mut descriptor_set_layout = PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stages);
        
        // Push Descriptor set binding
        let push_descriptor_set_layout = &mut descriptor_set_layout.set_layouts[push_descriptor_set_index as usize];
        push_descriptor_set_layout.flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;
        
        // Attachments Descriptor set
        // let attachment_descriptor_set_layout = &mut descriptor_set_layout.set_layouts[attachment_descriptor_set_index.unwrap() as usize];
        // attachment_descriptor_set_layout.flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;


        let descriptor_create_info = descriptor_set_layout.into_pipeline_layout_create_info(logical.clone()).unwrap();
        let pipeline_layout = PipelineLayout::new(logical.clone(), descriptor_create_info).unwrap();


       

        GraphicsPipeline::new(
            logical, 
            None,
            GraphicsPipelineCreateInfo {

                input_assembly_state: Some(InputAssemblyState {
                    topology: vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::TriangleList,
                    ..Default::default()
                }),
                
                stages: pipeline_stages.into_iter().collect(),
                
                vertex_input_state: Some(vertex_shader_input_state),
                
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0,0.0],
                        extent: window.inner_size().into(),
                        ..Default::default()
                    }]
                    .into_iter()
                    .collect(),

                    ..Default::default()
                }),
                subpass: Some(subpass.clone().into()),

                // flags: todo!(),
                rasterization_state: Some(RasterizationState {
                    cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::Back,
                    // front_face: vulkano::pipeline::graphics::rasterization::FrontFace::CounterClockwise,
                    front_face: vulkano::pipeline::graphics::rasterization::FrontFace::Clockwise,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.clone().num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                
                // dynamic_state: todo!(),
                ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
                // discard_rectangle_state: todo!(),
            }
        ).unwrap()

    }
}


impl GraphicsPipelineBuilder for LightingPipeline {
    type NewPipeline = LightingPipeline;

    fn new(
            window: Arc<winit::window::Window>,
            logical: Arc<vulkano::device::Device>,
            render_pass: Arc<RenderPass>,
            subpass_index: u32,
        ) -> Self::NewPipeline {
        
        let push_descriptor_set_index = 0;
        let attachment_descriptor_set_index = 1;
        
        Self::NewPipeline {
            pipeline: Self::create_pipeline(
                logical,
                window,
                render_pass, subpass_index, push_descriptor_set_index, Some(attachment_descriptor_set_index)),
            push_descriptor_set_index,
            attachment_descriptor_set_index
        }
    }

    fn get_push_descriptor_set_index(&self) -> u32 {
        self.push_descriptor_set_index
    }

    fn get_attachment_descriptor_set_index(&self) -> Option<u32> {
        Some(self.attachment_descriptor_set_index)
    }

}

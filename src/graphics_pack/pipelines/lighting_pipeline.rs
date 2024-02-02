use std::sync::Arc;

use vulkano::{pipeline::{graphics::{color_blend::ColorBlendState, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState, vertex_input::VertexDefinition, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo}, layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo}, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo}, render_pass::{RenderPass, Subpass}, sync::PipelineStages};

use crate::graphics_pack::shaders;

use super::{base_pipeline::GraphicsPipelineBuilder, renderpass};


pub(crate) struct LightingPipeline {
    pipeline: Arc<GraphicsPipeline>
}


impl GraphicsPipelineBuilder for LightingPipeline {
    type NewPipeline = LightingPipeline;

    fn new(
            window: Arc<winit::window::Window>,
            logical: Arc<vulkano::device::Device>,
            render_pass: Arc<RenderPass>,
            subpass_index: u32
        ) -> Self::NewPipeline {

        
        let subpass = Subpass::from(render_pass.clone(), subpass_index).unwrap();

        // load vertex and fragment shader
        let vertex_shader = shaders::lighting::load_vertex_shader(logical.clone())
            .entry_point("main")
            .unwrap();
        let fragment_shader = shaders::lighting::load_fragment_shader(logical.clone()).entry_point("main").unwrap();

        // TODO: Maybe not passing this into the pipeline object can fail
        let vertex_shader_input_state = [].definition(&vertex_shader.info().input_interface).unwrap();

        let pipeline_stages = [
            PipelineShaderStageCreateInfo::new(vertex_shader),
            PipelineShaderStageCreateInfo::new(fragment_shader)
        ];


        // create vertex shader input state for vertex_input_state field (maybe not needed if there is no data being passed in from the user)
        // create a pipeline stages object
        // get the descriptor set layout from pipeline stages
        // convert descriptor set layout into graphics pipeline layout
        // pass it as the layout of the pipeline


        // Creating pipeline layout with no descriptor sets configured
        let descriptor_set_layout = PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stages);
        let descriptor_create_info = descriptor_set_layout.into_pipeline_layout_create_info(logical.clone()).unwrap();
        let pipeline_layout = PipelineLayout::new(logical.clone(), descriptor_create_info).unwrap();
        
        Self::NewPipeline {
            pipeline: GraphicsPipeline::new(
                logical, 
                None,
                GraphicsPipelineCreateInfo {

                    input_assembly_state: Some(InputAssemblyState {
                        topology: vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    }),
                    
                    stages: pipeline_stages.into_iter().collect(),
                    
                    // vertex_input_state: todo!(),
                    
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
                    subpass: Some(subpass.into()),

                    // flags: todo!(),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::Back,
                        // front_face: vulkano::pipeline::graphics::rasterization::FrontFace::CounterClockwise,
                        front_face: vulkano::pipeline::graphics::rasterization::FrontFace::Clockwise,
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::default()),
                    
                    // dynamic_state: todo!(),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
                    // discard_rectangle_state: todo!(),
                }
            ).unwrap()
        }
    }

}

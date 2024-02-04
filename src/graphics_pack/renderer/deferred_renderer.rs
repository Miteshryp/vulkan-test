use std::sync::Arc;

use vulkano::render_pass::{RenderPass, Subpass};

use crate::graphics_pack::buffers::primitives;

struct DeferredRenderer {
    render_pass: Arc<RenderPass>,
    subpasses: Vec<Arc<Subpass>>, // This has to be in right order
    command_builder: primitives::PrimaryAutoCommandBuilderType
}

// Things we need for a renderer
// 1. Uniforms to be passed
// 2. Command builder
// 3. Buffer bindings (vertex, index[optional] and instance[optional])
// 4. Attachment buffers (image object corresponding to attachment buffers)


impl DeferredRenderer {
    pub fn new()  {

    }

}
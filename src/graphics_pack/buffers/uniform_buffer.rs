use std::{fmt::Write, sync::Arc};
use vulkano::{
    buffer::{BufferContents, BufferUsage},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, PersistentDescriptorSet,
        WriteDescriptorSet, persistent,
    },
    device::Device,
    memory::allocator::MemoryTypeFilter,
    pipeline::{GraphicsPipeline, Pipeline},
};

use crate::graphics_pack::buffers::base_buffer::*;

pub struct UniformSet {
    pub descriptor_set_index: usize,
    // pub persistent_descriptor_set: Arc<PersistentDescriptorSet>
    pub uniforms: Vec<UniformBuffer>,
}

impl UniformSet {

    // TODO: Put in a check to see if the descriptor index is in the
    //      range allowed by the physical device
    pub fn new(
        descriptor_set_index: usize,
    ) -> Self {
        // let pipeline_layout = graphics_pipeline
        //     .layout()
        //     .set_layouts()
        //     .get(self.descriptor_set_index)
        //     .unwrap();

        // let uniform_buffers: Vec<WriteDescriptorSet> = self.uniforms
        //     .into_iter()
        //     .map(|ub| {
        //         ub.get_descriptor_set()
        //     })
        //     .collect();

        // let persistent_descriptor_set = PersistentDescriptorSet::new(
        //     &descriptor_set_allocator,
        //     pipeline_layout.clone(),
        //     uniform_buffers,
        //     []
        // ).unwrap();

        UniformSet {
            descriptor_set_index: descriptor_set_index,
            // persistent_descriptor_set: persistent_descriptor_set,
            uniforms: Vec::new()
        }
    }

    pub fn get_persistent_descriptor_set(
        self,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        graphics_pipeline: Arc<GraphicsPipeline>,
    ) -> Arc<PersistentDescriptorSet> {
        // self.persistent_descriptor_set.clone()
        
        let pipeline_layout = graphics_pipeline
            .layout()
            .set_layouts()
            .get(self.descriptor_set_index)
            .unwrap();

        let uniform_buffers: Vec<WriteDescriptorSet> = self.uniforms
            .into_iter()
            .map(|ub| {
                ub.get_write_descriptor()
            })
            .collect();

        PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pipeline_layout.clone(),
            uniform_buffers,
            []
        ).unwrap()
    }

    // Creates a new uniform buffer and attaches it to the uniform set
    pub fn add_uniform_buffer<T>(
        &mut self,
        buffer_allocator: GenericBufferAllocator,
        graphics_pipeline: Arc<GraphicsPipeline>,
        data: T,
        options: BufferOptions,
    ) 
    where
        T: BufferContents
    {        
        let mut uniform_buffer = UniformBuffer::create(
            buffer_allocator, 
            // self.binded_uniforms,
            self.uniforms.len() as u32, 
            data, 
            Default::default()
        );
        
        self.uniforms.push(uniform_buffer);
    }


}

// TODO: New uniform buffer structure
// Step 1: UniformSet is a cover struct which will store all uniform buffers with the same set index
// Step 2: Each buffer in a single uniform set will have a different binding index (based on order of submission)
// Step 3: This UniformSet can then be compiled to produce a single persistent descriptor set
// Step 4: This persistent descriptor set can then be passed into a single bind_descriptor_set call

#[derive(Clone, Debug)]
pub struct UniformBuffer {
    descriptor_set: WriteDescriptorSet
    // descriptor_set: Arc<WriteDescriptorSet>
}


// TODO: Need to automatically create binding index.
// Allowing the user to create a binding index can lead to 
// clashing binding indexes.

impl UniformBuffer {
    pub fn create<T>(
        buffer_allocator: GenericBufferAllocator,
        binding_index: u32,
        data: T,
        options: BufferOptions,
    ) -> Self
    where
        T: BufferContents,
    {
        // Creating the buffer
        let buffer = create_buffer_from_single_data(
            buffer_allocator.clone(),
            data,
            BufferUsage::UNIFORM_BUFFER,
            options.memory_type_filter,
        );

        // Writing the buffer to a specific binding
        // Write descriptor set contains the raw bytes of the buffer passed into the buffer function.
        // Basically just converts our data into raw streamable bytes.
        let descriptor_set: WriteDescriptorSet = WriteDescriptorSet::buffer(binding_index, buffer);

        Self { descriptor_set: descriptor_set }
        // Self { descriptor_set: Arc::new(descriptor_set) }
    }

    pub fn get_write_descriptor(self) -> WriteDescriptorSet {
    // pub fn get_write_descriptor(&self) -> WriteDescriptorSet {
        self.descriptor_set
    }
}

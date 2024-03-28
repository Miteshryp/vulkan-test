use std::{fmt::Write, sync::Arc};
use vulkano::{
    buffer::{BufferContents, BufferUsage},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    memory::allocator::MemoryTypeFilter,
    pipeline::{GraphicsPipeline, Pipeline}, image::{sampler::Sampler, view::ImageView},
};

use crate::graphics::buffers::base_buffer::*;

use super::primitives::GenericBufferAllocator;

// INFO: New (ordered) uniform buffer structure
// Step 1: UniformSet is a cover struct which will store all uniform buffers with the same set index
// Step 2: Each buffer in a single uniform set will have a different binding index (based on order of submission)
// Step 3: This UniformSet can then be compiled to produce a single persistent descriptor set
// Step 4: This persistent descriptor set can then be passed into a single bind_descriptor_set call

/*
UniformSet Class represents a persistent descriptor set
entry into a graphics pipeline

METHODS:

*/


#[derive(Clone)]
pub struct UniformSet {
    pub descriptor_set_index: usize,
    pub uniforms: Vec<UniformBuffer>,
}


// TODO: Seperate the UniformSet into 2 types: 
// Ordered (binding is incremental) vs Unordered (binding is custom)
impl UniformSet {
    pub fn new(descriptor_set_index: usize) -> Self {
        UniformSet {
            descriptor_set_index: descriptor_set_index,
            uniforms: Vec::new(),
        }
    }

    pub fn get_persistent_descriptor_set(
        self,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        graphics_pipeline: Arc<GraphicsPipeline>,
    ) -> Arc<PersistentDescriptorSet> {
        let pipeline_layout = graphics_pipeline
            .layout()
            .set_layouts()
            .get(self.descriptor_set_index)
            .unwrap();

        let uniform_buffers: Vec<WriteDescriptorSet> = self
            .uniforms
            .into_iter()
            .map(|ub| ub.get_write_descriptor())
            .collect();

        let descriptor = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pipeline_layout.clone(),
            uniform_buffers,
            [],
        )
        .unwrap();
        // descriptor.

        return descriptor;
    }

    // pub fn get_dynamic_descriptor_set() {}

    // Creates a new uniform buffer and attaches it to the uniform set
    pub fn add_ordered_uniform_buffer<T>(
        &mut self,
        buffer_allocator: GenericBufferAllocator,
        data: T,
    ) where
        T: BufferContents,
    {
        let mut uniform_buffer = UniformBuffer::create_buffer(
            buffer_allocator,
            // self.binded_uniforms,
            self.uniforms.len() as u32,
            data
        );

        self.uniforms.push(uniform_buffer);
    }

    pub fn add_uniform_buffer(&mut self, uniform_buffer: UniformBuffer) {
        self.uniforms.push(uniform_buffer);
    }

    pub fn add_multiple_buffers(&mut self, buffers: Vec<UniformBuffer>) {
        self.uniforms = self.uniforms.iter().chain(buffers.iter()).cloned().collect();
    }
}

#[derive(Clone, Debug)]
pub struct UniformBuffer {
    descriptor_set: WriteDescriptorSet,
}

//INFO:
//  Independent Uniform Buffers can be used to create
//  and pass push descriptors into the command builder
impl UniformBuffer {
    pub fn create_sampler(binding_index: u32, sampler: Arc<Sampler>) -> Self {
        Self {
            descriptor_set: WriteDescriptorSet::sampler(binding_index, sampler)
        }
    }
    pub fn create_immutable_sampler(binding_index: u32) -> Self {
        Self {
            descriptor_set: WriteDescriptorSet::none(binding_index)
        }
    }

    pub fn create_image_view(binding_index: u32, image_view: Arc<ImageView>) -> Self {
        Self {
            descriptor_set: WriteDescriptorSet::image_view(binding_index, image_view)
        }
    }

    pub fn create_image_view_array(binding_index: u32, image_view: &Vec<Arc<ImageView>>) -> UniformBuffer {
        Self {
            descriptor_set: WriteDescriptorSet::image_view_array(binding_index, 0, image_view.to_vec())
        }
    }

    pub fn create_buffer<T>(
        buffer_allocator: GenericBufferAllocator,
        binding_index: u32,
        data: T
    ) -> Self
    where
        T: BufferContents,
    {
        // Creating the buffer
        let buffer = create_buffer_from_single_data(
            buffer_allocator.clone(),
            data,
            BufferUsage::UNIFORM_BUFFER,
            // options.memory_type_filter,
            MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
        );

        // Writing the buffer to a specific binding
        // Write descriptor set contains the raw bytes of the buffer passed into the buffer function.
        // Basically just converts our data into raw streamable bytes.
        let descriptor_set: WriteDescriptorSet = WriteDescriptorSet::buffer(binding_index, buffer);

        Self {
            descriptor_set: descriptor_set,
        }
    }

    pub fn get_write_descriptor(self) -> WriteDescriptorSet {
        self.descriptor_set
    }
}
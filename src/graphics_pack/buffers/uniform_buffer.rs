use std::sync::Arc;
use vulkano::{
    buffer::{BufferContents, BufferUsage},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::Device,
    memory::allocator::MemoryTypeFilter,
    pipeline::{GraphicsPipeline, Pipeline},
};

use crate::graphics_pack::buffers::base_buffer::*;

pub struct UniformBuffer<T>
where
    T: BufferContents,
{
    buffer: BufferSingle<T>,
    descriptor_set: Option<Arc<PersistentDescriptorSet>>,
}

impl<T> UniformBuffer<T>
where
    T: BufferContents,
{
    pub fn bind_descriptor_set(
        self,
        device: Arc<Device>,
        graphics_pipeline: Arc<GraphicsPipeline>,
        allocator: DescriptorSetAllocator,
        descriptor_set_index: u32,
        binding_index: u32,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .set_layouts()
            .get(descriptor_set_index as usize)
            .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(
                binding_index,
                self.buffer.raw_buffer,
            )], // 0 is the binding
            [],
        )
        .unwrap();

        descriptor_set
    }

}

impl<T> BufferOps<T> for UniformBuffer<T>
where
    T: BufferContents,
{
    type BufferAllocator = GenericBufferAllocator;
    // data should contain a single object
    fn from_vec(
        allocator: GenericBufferAllocator,
        data: Vec<T>,
        options: BufferOptions,
    ) -> Option<Self> {
        // let raw_buffer = create_buffer_from_vec(
        //     allocator.clone(),
        //     data,
        //     BufferUsage::UNIFORM_BUFFER,
        //     options.memory_type_filter
        // );

        // UniformBuffer {
        // buffer: BufferVec {
        //         raw_buffer: raw_buffer,
        //         options: options
        //     }
        // }

        None
    }

    fn consume(self) -> (vulkano::buffer::Subbuffer<[T]>, u32) {
        todo!()
    }

    fn from_data(
        allocator: GenericBufferAllocator,
        data: T,
        options: BufferOptions,
    ) -> Option<Self> {
        let buffer = create_buffer_from_single_data(
            allocator.clone(),
            data,
            BufferUsage::UNIFORM_BUFFER,
            options.memory_type_filter,
        );

        Some(Self {
            buffer: BufferSingle {
                raw_buffer: buffer,
                options: options,
            },
            descriptor_set: None
        })
    }
}

use vulkano::{
    buffer::{Buffer, BufferUsage, BufferCreateInfo, Subbuffer},
    memory::allocator::{MemoryTypeFilter, AllocationCreateInfo}, image::Image,
};

use crate::graphics_pack::buffers::base_buffer::{self, BufferOptions};

use super::base_buffer::{create_buffer_from_vec, GenericBufferAllocator, VecBuffer, VecBufferOps};

pub struct ImageBuffer {
    buffer: VecBuffer<u8>,
    width: u32,
    height: u32,
    channel_count: u8,
}

impl Clone for ImageBuffer {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            channel_count: self.channel_count,
            buffer: VecBuffer {
                raw_buffer: self.buffer.raw_buffer.clone(),
                options: BufferOptions {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                } 
            }
        }
    }
}

impl ImageBuffer {
    pub fn new(
        allocator: GenericBufferAllocator,
        data: Vec<u8>,
        dimensions: [u32; 3],
    ) -> Self {
        let options = BufferOptions {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE   
        };

        let buffer = create_buffer_from_vec(
            allocator,
            data,
            BufferUsage::TRANSFER_SRC,
            options.memory_type_filter
        );

        Self {
            buffer: VecBuffer {
                raw_buffer: buffer,
                options: options
            },
            width: dimensions[0],
            height: dimensions[1],
            channel_count: dimensions[2] as u8,
        }
    }


    pub fn consume(self) -> (vulkano::buffer::Subbuffer<[u8]>, u32) {
        (self.buffer.raw_buffer, self.width * self.height * self.channel_count as u32)
    }
}

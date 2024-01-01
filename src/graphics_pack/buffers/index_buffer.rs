
use vulkano::{
    buffer::{BufferUsage, Subbuffer, IndexType},
};

use crate::graphics_pack::buffers::{
    base_buffer::{
        self,
        BufferVec, BufferOps, BufferOptions, GenericBufferAllocator,
    },
    primitives::{
        self, VertexPoint, Vec3
    }
};

#[derive(Clone)]
pub struct IndexBuffer {
    pub buffer: BufferVec<u32>,
    pub indicies: u32
}

impl BufferOps<u32> for IndexBuffer {
    type BufferAllocator = GenericBufferAllocator;
    
    fn from_vec(
        allocator: GenericBufferAllocator,
        indicies: Vec<u32>, 
        options: BufferOptions
    ) -> Option<Self> where Self: Sized {
        let index_count = indicies.len();
        let index_buffer = base_buffer::create_buffer_from_vec(
            allocator.clone(), 
            indicies, 
            BufferUsage::INDEX_BUFFER, 
            options.memory_type_filter
        );

        Some(IndexBuffer {
            buffer: BufferVec {
                raw_buffer: index_buffer,
                options: options
            },
            indicies: index_count as u32
        })
    }

    fn consume(self) -> (Subbuffer<[u32]>, u32) {
        (self.buffer.raw_buffer, self.indicies)
    }

    fn from_data(allocator: GenericBufferAllocator, data: u32, options: BufferOptions) -> Option<Self> where Self: Sized {
        todo!()
    }
}

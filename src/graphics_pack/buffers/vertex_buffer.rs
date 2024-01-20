use vulkano::buffer::{Buffer, BufferUsage, IndexType, Subbuffer};

use crate::graphics_pack::buffers::{
    base_buffer::{self, BufferOptions, GenericBufferAllocator, VecBuffer, VecBufferOps},
    primitives::{InstanceData, VertexData},
};

use super::base_buffer::create_buffer_from_vec;

#[derive(Clone)]
pub struct VertexBuffer {
    pub buffer: VecBuffer<VertexData>,
    pub vertices: u32,
}

#[derive(Clone)]
pub struct InstanceBuffer {
    pub buffer: VecBuffer<InstanceData>,
    pub instances: u32,
}

impl VecBufferOps<VertexData> for VertexBuffer {
    type BufferAllocator = GenericBufferAllocator;

    fn from_vec(
        allocator: GenericBufferAllocator,
        vertices: Vec<VertexData>,
        options: BufferOptions,
    ) -> Option<Self> {
        let vertex_count = vertices.len();
        let buffer = base_buffer::create_buffer_from_vec(
            allocator.clone(),
            vertices,
            BufferUsage::VERTEX_BUFFER,
            options.memory_type_filter,
        );

        Some(VertexBuffer {
            buffer: VecBuffer {
                raw_buffer: buffer,
                options: options,
            },
            vertices: vertex_count as u32,
        })
    }

    fn consume(self) -> (Subbuffer<[VertexData]>, u32) {
        (self.buffer.raw_buffer, self.vertices)
    }

    // fn from_data(
    //     allocator: GenericBufferAllocator,
    //     data: VertexPoint,
    //     options: BufferOptions,
    // ) -> Option<Self>
    // where
    //     Self: Sized,
    // {
    //     None
    // }
}

impl VecBufferOps<InstanceData> for InstanceBuffer {
    type BufferAllocator = GenericBufferAllocator;

    fn from_vec(
        allocator: Self::BufferAllocator,
        data: Vec<InstanceData>,
        options: BufferOptions,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        let vertex_count = data.len();
        let buffer = base_buffer::create_buffer_from_vec(
            allocator.clone(),
            data,
            BufferUsage::VERTEX_BUFFER,
            options.memory_type_filter,
        );

        Some(InstanceBuffer {
            buffer: VecBuffer {
                raw_buffer: buffer,
                options: options,
            },
            instances: vertex_count as u32,
        })
    }

    // fn from_data(
    //     allocator: Self::BufferAllocator,
    //     data: InstanceData,
    //     options: BufferOptions,
    // ) -> Option<Self>
    // where
    //     Self: Sized,
    // {
    //     None
    // }

    fn consume(self) -> (Subbuffer<[InstanceData]>, u32) {
        (self.buffer.raw_buffer, self.instances)
    }
}

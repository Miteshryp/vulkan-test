
use vulkano::{
    buffer::{BufferUsage, Subbuffer, IndexType},
};

use crate::graphics_pack::buffers::{
    base_buffer::{
        self,
        VecBuffer, GenericBufferAllocator, BufferOptions, VecBufferOps
    },
    primitives::{
        VertexPoint
    }
};


#[derive(Clone)]
pub struct VertexBuffer {
    pub buffer: VecBuffer<VertexPoint>,
    pub vertices: u32,
}

impl VecBufferOps<VertexPoint> for VertexBuffer {
    type BufferAllocator = GenericBufferAllocator;

    fn from_vec(
        allocator: GenericBufferAllocator,
        vertices: Vec<VertexPoint>,
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

    fn consume(self) -> (Subbuffer<[VertexPoint]>, u32) {
        (self.buffer.raw_buffer, self.vertices)
    }

    fn from_data(allocator: GenericBufferAllocator, data: VertexPoint, options: BufferOptions) -> Option<Self> where Self: Sized {
        // let buffer = base_buffer::create_buffer_from_single_data(
        //     allocator.clone(),
        //     ,
        //     BufferUsage::VERTEX_BUFFER,
        //     options.memory_type_filter
        // );

        // Some(VertexBuffer {
        //     buffer: BufferVec {
        //         raw_buffer: buffer,
        //         options: options
        //     },
        //     vertices: 1
        // })

        None

        // Cannot create a vertex buffer with a single vertex
    }
}

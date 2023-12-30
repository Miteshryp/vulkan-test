pub mod primitives;

use std::sync::Arc;
use vulkano::{
    buffer::{BufferContents, BufferCreateInfo, BufferUsage, Subbuffer, IndexType},
    memory::allocator::{AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter},
};

use self::primitives::VertexPoint;

type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;

// Helper function
// fn create_buffer_from_iter<T, I>(
fn create_buffer_from_vec<T>(
    allocator: GenericBufferAllocator,
    // iter: I,
    data: Vec<T>,
    buffer_usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Subbuffer<[T]>
where
    T: BufferContents,
    // I: IntoIterator<Item = T>,
    // I::IntoIter: ExactSizeIterator,
{
    vulkano::buffer::Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: buffer_usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: memory_type_filter,
            ..Default::default()
        },
        data.into_iter(),
    )
    .unwrap()
}

// RUST philosophy questions
// Explicit vs duplication
// Rust prefers explicit nature in exchange for code duplication

#[derive(Debug, Clone)]
pub struct BufferOptions {
    pub memory_type_filter: MemoryTypeFilter,
}

impl Default for BufferOptions {
    fn default() -> Self {
        Self {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        }
    }
}

// Bare bones structure for sub buffer storage
#[derive(Clone)]
pub struct Buffer<T>
where
    T: BufferContents,
{
    pub raw_buffer: Subbuffer<[T]>,
    pub options: BufferOptions, // metadata. @TODO: Remove if not required in the future
}

pub trait BufferOps<T> {
    fn new(allocator: GenericBufferAllocator, data: Vec<T>, options: BufferOptions) -> Self;
    fn consume(self) -> (Subbuffer<[T]>, u32);
}

#[derive(Clone)]
pub struct VertexBuffer {
    pub buffer: Buffer<primitives::VertexPoint>,
    pub vertices: u32,
}

impl BufferOps<VertexPoint> for VertexBuffer {
    fn new(
        allocator: GenericBufferAllocator,
        vertices: Vec<primitives::VertexPoint>,
        options: BufferOptions,
    ) -> Self {
        let vertex_count = vertices.len();
        let buffer = create_buffer_from_vec(
            allocator.clone(),
            vertices,
            BufferUsage::VERTEX_BUFFER,
            options.memory_type_filter,
        );

        VertexBuffer {
            buffer: Buffer {
                raw_buffer: buffer,
                options: options,
            },
            vertices: vertex_count as u32,
        }
    }

    fn consume(self) -> (Subbuffer<[VertexPoint]>, u32) {
        (self.buffer.raw_buffer, self.vertices)
    }
}


#[derive(Clone)]
pub struct IndexBuffer {
    pub buffer: Buffer<u32>,
    pub indicies: u32
}

impl BufferOps<u32> for IndexBuffer {
    fn new(
        allocator: GenericBufferAllocator,
        indicies: Vec<u32>, 
        options: BufferOptions
    ) -> Self {
        let index_count = indicies.len();
        let index_buffer = create_buffer_from_vec(
            allocator.clone(), 
            indicies, 
            BufferUsage::INDEX_BUFFER, 
            options.memory_type_filter
        );

        IndexBuffer {
            buffer: Buffer {
                raw_buffer: index_buffer,
                options: options
            },
            indicies: index_count as u32
        }
    }

    fn consume(self) -> (Subbuffer<[u32]>, u32) {
        (self.buffer.raw_buffer, self.indicies)
    }
}



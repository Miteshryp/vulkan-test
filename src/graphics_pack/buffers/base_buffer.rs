use std::sync::Arc;
use vulkano::{
    buffer::{BufferContents, BufferCreateInfo, BufferUsage, IndexType, Subbuffer},
    memory::allocator::{AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter}, descriptor_set::allocator::StandardDescriptorSetAllocator,
};

use super::{VertexPoint, primitives::Vec3};

pub type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
pub type DescriptorSetAllocator =
    Arc<StandardDescriptorSetAllocator>;

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
            // CPU -> GPU streaming
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        }
    }
}

// Helper function
// fn create_buffer_from_iter<T, I>(
pub fn create_buffer_from_vec<T>(
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

// Maybe implement this to create buffer from single data object
pub fn create_buffer_from_single_data<T>(
    allocator: GenericBufferAllocator,
    data: T,
    buffer_usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Subbuffer<T>
where
    T: BufferContents,
{
    vulkano::buffer::Buffer::from_data(
        allocator.clone(),
        BufferCreateInfo {
            usage: buffer_usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: memory_type_filter,
            ..Default::default()
        },
        data,
    )
    .unwrap()
}

// Bare bones structure for sub buffer storage
#[derive(Clone)]
pub struct VecBuffer<T>
where
    T: BufferContents
{
    pub raw_buffer: Subbuffer<[T]>,
    pub options: BufferOptions, // metadata. @TODO: Remove if not required in the future
}

pub struct BufferSingle<T>
where  
    T: BufferContents
{
    pub raw_buffer: Subbuffer<T>,
    pub options: BufferOptions
}

pub trait VecBufferOps<T> 
{

    type BufferAllocator;
    // options parameter has the memory type filter field
    // We are letting the user determine this field right
    // so that we may implement staging buffers of different types
    // with ease in the future.
    fn from_vec(allocator: Self::BufferAllocator, data: Vec<T>, options: BufferOptions) -> Option<Self> where Self: Sized;
    fn from_data(allocator: Self::BufferAllocator, data: T, options: BufferOptions) -> Option<Self> where Self: Sized;
    fn consume(self) -> (Subbuffer<[T]>, u32);
}

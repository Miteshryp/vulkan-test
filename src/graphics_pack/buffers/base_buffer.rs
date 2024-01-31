use std::{mem::size_of, rc::Rc, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexType, Subbuffer}, descriptor_set::allocator::StandardDescriptorSetAllocator, image::view::{ImageView, ImageViewCreateInfo}, memory::allocator::{AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter}, VulkanObject
};

use super::{primitives::Vec3, VertexData};

pub type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
pub type DescriptorSetAllocator = Arc<StandardDescriptorSetAllocator>;

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
    data: &Vec<T>,
    buffer_usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Subbuffer<[T]>
where
    T: BufferContents + Clone,
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
        data.into_iter().cloned(),
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
// #[derive(Clone)]
// pub struct VecBuffer<T>
// where
//     T: BufferContents
// {
//     pub raw_buffer: Subbuffer<[T]>,
//     pub options: BufferOptions, // metadata. @TODO: Remove if not required in the future
// }

// pub struct BufferSingle<T>
// where
//     T: BufferContents
// {
//     pub raw_buffer: Subbuffer<T>,
//     pub options: BufferOptions
// }

// pub trait VecBufferOps<T>
// {

//     type BufferAllocator;
//     // options parameter has the memory type filter field
//     // We are letting the user determine this field right
//     // so that we may implement staging buffers of different types
//     // with ease in the future.

//     fn from_vec(allocator: Self::BufferAllocator, data: &Vec<T>, options: BufferOptions) -> Option<Self> where Self: Sized;
//     // fn from_data(allocator: Self::BufferAllocator, data: T, options: BufferOptions) -> Option<Self> where Self: Sized;
//     fn consume(self) -> (Subbuffer<[T]>, u32);
// }

pub struct StagingBuffer<T>
where
    T: BufferContents,
{
    buffer: Vec<T>,
}

impl<T> StagingBuffer<T>
where
    T: BufferContents + Clone,
{
    // Constructors
    pub fn new() -> Self {
        Self { buffer: vec![] }
    }

    pub fn from_vec(buffer: Vec<T>) -> Self {
        Self { buffer: buffer }
    }

    pub fn from_vec_ref(buffer: &Vec<T>) -> Self {
        Self {
            buffer: buffer.into_iter().cloned().collect(),
        }
    }

    // Methods
    pub fn byte_size(&self) -> usize {
        self.buffer.len() * size_of::<T>()
    }

    pub fn count(&self) -> usize {
        self.buffer.len()
    }

    pub fn add_single(&mut self, data: T) {
        self.buffer.push(data);
    }

    pub fn add_vec(&mut self, data: &Vec<T>) {
        self.buffer = self.buffer.iter().chain(data.iter()).cloned().collect();
    }

    // pub fn create_buffer(
    //     self,
    //     allocator: GenericBufferAllocator,
    //     usage: BufferUsage,
    // ) -> Subbuffer<[T]> {
    //     Buffer::from_iter(
    //         allocator,
    //         BufferCreateInfo {
    //             usage: BufferUsage::TRANSFER_SRC | usage,
    //             ..Default::default()
    //         },
    //         AllocationCreateInfo {
    //             memory_type_filter: MemoryTypeFilter::PREFER_HOST
    //                 | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    //             ..Default::default()
    //         },
    //         self.buffer.into_iter(),
    //     )
    //     .unwrap()
    // }

    pub fn create_host_buffer(self, allocator: GenericBufferAllocator, usage: BufferUsage) -> Subbuffer<[T]> {
        Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.buffer.into_iter(),
        )
        .unwrap()
    }


    // Returns a tuple with the format: (host_buffer, device_buffer)
    pub fn create_buffer_mapping(
        self,
        allocator: GenericBufferAllocator,
        usage: BufferUsage,
    ) -> (Subbuffer<[T]>, Subbuffer<[T]>)  {

        let buffer_size = self.byte_size();

        // Creating host buffer
        let host_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.buffer.into_iter(),
        )
        .unwrap();


        // Creating device buffer
        let device_buffer = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            buffer_size as u64
        ).unwrap();

        return (host_buffer, device_buffer);
        
    }


    // Static functions
    // pub fn get_device_buffer(
    //     allocator: GenericBufferAllocator,
    //     staging_buffer: StagingBuffer<T>,
    //     usage: BufferUsage
    // ) -> Subbuffer<[T]> {
    //     let buffer_size = staging_buffer.byte_size();
        
    //     Buffer::new_slice(
    //         allocator,
    //         BufferCreateInfo {
    //             usage: BufferUsage::TRANSFER_DST | usage,
    //             ..Default::default()
    //         },
    //         AllocationCreateInfo {
    //             memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
    //             ..Default::default()
    //         },
    //         buffer_size as u64
    //     ).unwrap()
    // }
}


pub struct StagingBufferMap {
    pub host_buffer: Subbuffer<[u8]>,
    pub device_buffer: Subbuffer<[u8]>
}



pub struct DeviceBuffer<T> {
    pub buffer: Subbuffer<[T]>,
    pub count: u32
}
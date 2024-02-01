use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    image::{Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::graphics_pack::buffers::base_buffer::{self, BufferOptions};

use super::base_buffer::StagingBuffer;

pub trait StagingImage {
    fn get_staging_buffer(self) -> StagingBuffer<u8>;
}



#[derive(Clone)]
pub struct StagingImageArrayBuffer {
    image_buffer: StagingBuffer<u8>,
    // image_count: u32,

    width: u32,
    height: u32,
    channel_count: u32, // image_array: Vec<SingleImageInfo> // Cannot support images with different sizes in texture2DArray, i guess?
                       // image_info: SingleImageInfo
}

impl StagingImageArrayBuffer {
    pub fn new(
        // allocator: GenericBufferAllocator,
        // image_count: u32,
        width: u32,
        height: u32,
        channel_count: u32,
    ) -> Self {
        // let buffer = Buffer::new(allocator.clone(), create_info, allocation_info, layout)
        Self {
            image_buffer: StagingBuffer::new(),
            // image_count: 0,

            width: width,
            height: height,
            channel_count: channel_count, // static for now
        }
    }

    pub fn width(&self) -> u32 { self.width }        
    pub fn height(&self) -> u32 { self.height }    
    pub fn channel_count(&self) -> u32 { self.channel_count }    


    pub fn add_image_data(&mut self, data: &Vec<u8>) {
        
        // Only images of allowed size is taken into the image buffer
        // std::assert!(data.len() as u32 == self.width * self.height * self.channel_count);

        self.image_buffer.add_vec(data);
    }

    pub fn get_image_count(&self) -> u32 {
        self.image_buffer.count() as u32 / (self.width * self.height * self.channel_count) as u32
    }

//     pub fn create_image_object(&self, allocator: GenericBufferAllocator) -> Arc<Image> {
//         Image::new(
//             allocator,
//             ImageCreateInfo {
//                 // array_layers: self.image_info.len() as u32,
//                 array_layers: self.image_count,
//                 format: vulkano::format::Format::R8G8B8A8_SRGB,
//                 extent: [self.width, self.height, 1],
//                 usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
//                 image_type: vulkano::image::ImageType::Dim2d,
//                 ..Default::default()
//             },
//             AllocationCreateInfo {
//                 memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
//                 ..Default::default()
//             },
//         )
//         .unwrap()
//     }
}

impl StagingImage for StagingImageArrayBuffer {
    fn get_staging_buffer(self) -> StagingBuffer<u8> {
        self.image_buffer
    }
}







pub struct StagingImageBuffer {
    // buffer: VecBuffer<u8>,
    buffer: StagingBuffer<u8>,
    width: u32,
    height: u32,
    channel_count: u32,
}

impl StagingImageBuffer {
    pub fn new() -> Self{
        Self {
            buffer: StagingBuffer::new(),
            width: 0,
            height: 0,
            channel_count: 0
        }
    }

    // Getters
    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    pub fn channel_count(&self) -> u32 { self.channel_count }

    pub fn from_vec_ref(data: &Vec<u8>, width: u32, height: u32, channel_count: u32) -> Self {
        std::assert!(data.len() as u32 == width * height * channel_count);
        Self {
            buffer: StagingBuffer::from_vec_ref(data),
            width: width,
            height: height,
            channel_count: channel_count
        }
    }

    pub fn reset_image_data(&mut self, data: &Vec<u8>, width: u32, height: u32, channel_count: u32) {
        *self = Self::from_vec_ref(data, width, height, channel_count);
    }

    
}


impl StagingImage for StagingImageBuffer {
    fn get_staging_buffer(self) -> StagingBuffer<u8> {
        self.buffer
    }
}
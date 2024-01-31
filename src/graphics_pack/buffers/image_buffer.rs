// use std::sync::Arc;
// use vulkano::{
//     buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
//     image::{Image, ImageCreateInfo, ImageUsage},
//     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
// };

// use crate::graphics_pack::buffers::base_buffer::{self, BufferOptions};

// use super::base_buffer::{create_buffer_from_vec, GenericBufferAllocator, VecBuffer, VecBufferOps};

// #[derive(Clone)]
// pub struct ImageArrayBuffer {
//     image_buffer: Vec<u8>,
//     image_count: u32,

//     width: u32,
//     height: u32,
//     channel_count: u8, // image_array: Vec<SingleImageInfo> // Cannot support images with different sizes in texture2DArray, i guess?
//                        // image_info: SingleImageInfo
// }

// impl VecBufferOps<u8> for ImageArrayBuffer {
//     type BufferAllocator = GenericBufferAllocator;

//     fn from_vec(allocator: Self::BufferAllocator, data: &Vec<u8>, options: BufferOptions) -> Option<Self> where Self: Sized {
//         todo!()
//     }

//     fn consume(self) -> (Subbuffer<[u8]>, u32) {
//         todo!()
//     }
// }

// impl ImageArrayBuffer {
//     pub fn new(
//         // allocator: GenericBufferAllocator,
//         // image_count: u32,
//         width: u32,
//         height: u32,
//         channel_count: u8,
//     ) -> Self {
//         // let buffer = Buffer::new(allocator.clone(), create_info, allocation_info, layout)
//         Self {
//             image_buffer: vec![],
//             image_count: 0,

//             width: width,
//             height: height,
//             channel_count: channel_count, // static for now
//         }
//     }

//     pub fn add_image_data(&mut self, data: &Vec<u8>) {
//         self.image_buffer = self
//             .image_buffer
//             .iter()
//             .chain(data.iter())
//             .copied()
//             .collect();
//         self.image_count += 1;
//     }

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

//     pub fn get_buffer(&mut self, allocator: GenericBufferAllocator) -> Subbuffer<[u8]> {
//         let buffer = create_buffer_from_vec(
//             allocator,
//             &self.image_buffer,
//             BufferUsage::TRANSFER_SRC,
//             MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
//         );

//         self.image_buffer.clear();
//         self.image_count = 0;
//         self.width = 0;
//         self.height = 0;

//         return buffer;
//     }
// }

// pub struct ImageBuffer {
//     buffer: VecBuffer<u8>,
//     width: u32,
//     height: u32,
//     channel_count: u8,
// }

// impl Clone for ImageBuffer {
//     fn clone(&self) -> Self {
//         Self {
//             width: self.width,
//             height: self.height,
//             channel_count: self.channel_count,
//             // image_array: self.image_array.clone(),
//             buffer: VecBuffer {
//                 raw_buffer: self.buffer.raw_buffer.clone(),
//                 options: BufferOptions {
//                     memory_type_filter: MemoryTypeFilter::PREFER_HOST
//                         | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
//                 },
//             }, // image_buffer: self.image_buffer.clone(),
//                // image_array: self.image_array.clone(),
//         }
//     }
// }

// /*
//     Maybe we done need to store the buffer object in the Image buffer
//     Instead, we can simply store the image buffer data in a vector.

//     Then, when the consume function is called, we can create the vector
//     based on the data we have stored in the buffer and return the buffer
//     there itself.
// */

// impl ImageBuffer {
//     pub fn new(allocator: GenericBufferAllocator, data: &Vec<u8>, dimensions: [u32; 3]) -> Self {
//         let options = BufferOptions {
//             memory_type_filter: MemoryTypeFilter::PREFER_HOST
//                 | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
//         };
//         let buffer = create_buffer_from_vec(
//             allocator,
//             data,
//             BufferUsage::TRANSFER_SRC,
//             options.memory_type_filter,
//         );

//         Self {
//             buffer: VecBuffer {
//                 raw_buffer: buffer,
//                 options: options,
//             },
//             width: dimensions[0],
//             height: dimensions[1],
//             channel_count: dimensions[2] as u8,
//         }
//     }

//     pub fn write_handle(&mut self) -> vulkano::buffer::BufferWriteGuard<'_, [u8]> {
//         self.buffer.raw_buffer.write().unwrap()
//     }

//     pub fn consume(self) -> (vulkano::buffer::Subbuffer<[u8]>, u32) {
//         (
//             self.buffer.raw_buffer,
//             self.width * self.height * self.channel_count as u32,
//         )
//     }
// }

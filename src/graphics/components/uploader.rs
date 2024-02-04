use std::sync::Arc;

use vulkano::{
    buffer::{BufferContents, BufferUsage}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, CopyBufferToImageInfo,
    }, format::Format, image::{view::ImageView, Image, ImageCreateInfo, ImageUsage}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}
};

use crate::graphics::buffers::{
    base_buffer::{DeviceBuffer, StagingBuffer},
    image_buffer::{StagingImage, StagingImageArrayBuffer, StagingImageBuffer},
    primitives::CommandBufferType,
};

// use crate::graphics_pack::buffers::primitives::GenericBufferAllocator;
use super::super::buffers::primitives::{GenericBufferAllocator, PrimaryAutoCommandBuilderType};

pub struct BufferUploader {
    // command_builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    command_builder: PrimaryAutoCommandBuilderType,
    buffer_allocator: GenericBufferAllocator,
}

impl BufferUploader {
    pub fn new(
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        buffer_allocator: GenericBufferAllocator,
        queue_family_index: u32,
    ) -> Self {
        Self {
            command_builder: AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue_family_index,
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap(),

            buffer_allocator: buffer_allocator,
        }
    }

    pub fn insert_buffer<T>(
        &mut self,
        buffer: StagingBuffer<T>,
        usage: BufferUsage,
    ) -> DeviceBuffer<T>
    where
        T: BufferContents + Clone,
    {
        // Get the host and device buffers from the staging buffer
        let buffer_count = buffer.count();
        let (host_buffer, device_buffer) =
            buffer.create_buffer_mapping(self.buffer_allocator.clone(), usage);

        // Create a mapping object and store it vec of mappings
        self.command_builder
            .copy_buffer(CopyBufferInfo::buffers(host_buffer, device_buffer.clone()))
            .unwrap();

        // return the device buffer for further usage in the rendering command buffer.
        DeviceBuffer {
            buffer: device_buffer,
            count: buffer_count as u32,
        }
    }

    pub fn insert_image(
        &mut self,
        // buffer: StagingBuffer<u8>,
        image_buffer: StagingImageBuffer,
    ) -> Arc<ImageView> {
        let image_object = self.create_image_object(ImageCreateInfo {
            image_type: vulkano::image::ImageType::Dim2d,
            extent: [image_buffer.width(), image_buffer.height(), 1],
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            format: Format::R8G8B8A8_SRGB,
            ..Default::default()  
        });
        
        let image_view = self.build_and_get_image_view(image_object, image_buffer);
        image_view
    }

    pub fn insert_image_array(&mut self, image_array_buffer: StagingImageArrayBuffer) -> Arc<ImageView> {
        let image_object = self.create_image_object(ImageCreateInfo {
            array_layers: image_array_buffer.get_image_count(),
            image_type: vulkano::image::ImageType::Dim2d,
            extent: [image_array_buffer.width(), image_array_buffer.height(), 1],
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            format: Format::R8G8B8A8_SRGB,
            ..Default::default() 
        });

        let image_view = self.build_and_get_image_view(image_object, image_array_buffer);
        image_view
    }

    pub fn get_one_time_command_buffer(self) -> CommandBufferType {
        self.command_builder.build().unwrap()
    }

    // private

    fn create_image_object(&self, create_info: ImageCreateInfo) -> Arc<Image> {
        Image::new(
            self.buffer_allocator.clone(),
            create_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn build_and_get_image_view(&mut self, image_object: Arc<Image>, image_buffer: impl StagingImage) -> Arc<ImageView> {
        let image_view = ImageView::new_default(image_object.clone()).unwrap();
        
        // Getting the source buffer to pass the data into the command builder
        let src_buffer = image_buffer
            .get_staging_buffer()
            .create_host_buffer(self.buffer_allocator.clone(), Default::default());

        // Building the command buffer further
        self.command_builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                src_buffer,
                image_object.clone(),
            ))
            .unwrap();

        image_view
    }
}

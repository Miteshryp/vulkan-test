use std::sync::Arc;

use winit::{
    event_loop::EventLoop, window::{Window}
};

use vulkano::{
    command_buffer::{
        allocator::{
            StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
        PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, 
    },
    device::{
        physical::{self, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
        QueueFlags,
    },

    pipeline::GraphicsPipeline,
    image::{
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageUsage,
    },

    instance::InstanceCreateInfo,

    memory::{
        allocator::{
            AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
        },
        MemoryType,
    },

    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{self, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    Version, VulkanLibrary, VulkanError
};


use crate::graphics_pack::pipelines::{base_pipeline::GraphicsPipelineBuilder, deferred_pipeline, lighting_pipeline};

// type name for buffer allocator
type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
type CommandBufferType = Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

type PrimaryAutoCommandBuilderType = AutoCommandBufferBuilder<
    PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
    Arc<StandardCommandBufferAllocator>,
>;






pub struct RenderTargetInfo {
    pub pipeline: Arc<GraphicsPipeline>,
    // pub render_pass: Arc<RenderPass>,
    pub fbos: Vec<Arc<Framebuffer>>,
    pub image_sampler: Arc<Sampler>,
}

pub struct InstanceAllocators {
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_allocator: GenericBufferAllocator,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

pub struct VulkanInstance {
    logical: Arc<Device>,
    physical: Arc<vulkano::device::physical::PhysicalDevice>,
    surface: Arc<Surface>,
    queue: Arc<Queue>,
    // device_queues: Vec<Arc<Queue>>,

    pub swapchain_info: VulkanSwapchainInfo,
    pub render_target: RenderTargetInfo,
    pub allocators: InstanceAllocators,
}

pub struct VulkanSwapchainInfo {
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<Image>>,
}

// New Rendering system
// 1. Push the data into a upload structure
// 2. Get the device buffer mapping, where the data is eventually going to end up. The device buffer which is extracted from this mapping is going to be passed into the rendering command buffer construction.
// 3. create a command buffer in the upload structure
// 4. get the per frame upload command buffer future and wait before executing the rendering command buffer

impl VulkanInstance {
    pub fn initialise(window: Arc<Window>, el: &EventLoop<()>) -> Self {
        // initialise_vulkan_runtime(window, eltw)

        /*
            Step 1. Select a Physical Device
        */

        // Loading the vulkan plugins
        let vulkan_library = vulkano::VulkanLibrary::new()
            .unwrap_or_else(|err| panic!("Failed to load Vulkan: \n {:?}", err));

        // Getting the surface extensions required by the created window
        let required_extentions = Surface::required_extensions(el);

        // Creating a vulkan instance
        let vulkan_instance = vulkano::instance::Instance::new(
            vulkan_library,
            InstanceCreateInfo {
                enabled_extensions: required_extentions,
                max_api_version: Some(Version {
                    major: 1,
                    minor: 3,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap_or_else(|err| panic!("Failed to create Vulkan Instance \n {:?}", err));

        // Creating a surface object binding the vulkan instance with the window
        let surface = Surface::from_window(vulkan_instance.clone(), window.clone())
            .unwrap_or_else(|err| panic!("Could not create surface object: \n{:?}", err));

        // We need a device with swapchain extensions for graphics rendering
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_push_descriptor: true,
            // ext_debug_marker: true,
            ..Default::default()
        };

        let (physical_device, queue_family_index) = vulkan_instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // Checking for proper support of
                        // 1. Surface support for the created surface
                        // 2. Required Queue family availability
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|queue_index| (p, queue_index))
            })
            // .next()
            .min_by_key(|(device, _)| match device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::Cpu => 2,
                _ => 3,
            })
            .unwrap_or_else(|| panic!("Failed to find a valid physical device\n"));

        println!(
            "Physical Device: {:?}",
            physical_device.properties().device_name
        );

        /*
        Creating Logical Device
        */
        let (logical_device, mut queues_iterator) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .unwrap_or_else(|err| panic!("Failed to create a logical device: \n{:?}", err));

        // let device_queues: Vec<Arc<Queue>> = queues_iterator.collect();
        let swapchain = create_swapchain(
            window.clone(),
            physical_device.clone(),
            logical_device.clone(),
            surface.clone(),
        );

        let allocators = InstanceAllocators {
            command_buffer_allocator: VulkanInstance::create_command_buffer_allocator(
                logical_device.clone(),
            ),
            memory_allocator: VulkanInstance::create_buffer_allocator(logical_device.clone()),
            descriptor_set_allocator: VulkanInstance::create_descriptor_set_allocator(
                logical_device.clone(),
            ),
        };

        let render_target_info = VulkanInstance::create_render_target(
            window.clone(),
            logical_device.clone(),
            &swapchain,
            allocators.memory_allocator.clone(),
        );

        VulkanInstance {
            logical: logical_device,
            physical: physical_device,
            // device_queues: device_queues,
            queue: queues_iterator.next().unwrap(),
            surface: surface,
            swapchain_info: swapchain,
            render_target: render_target_info,
            allocators: allocators,
        }
    }


    pub fn create_render_target(
        window: Arc<Window>,
        device: Arc<Device>,
        swapchain_info: &VulkanSwapchainInfo,
        allocator: GenericBufferAllocator,
    ) -> RenderTargetInfo {
        let render_pass = VulkanInstance::create_render_pass(device.clone(), swapchain_info); // defines the schema information required for configuring the output of shaders to the framebuffer
    
        let depth_image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                // image_type: vulkano::image::ImageType::Dim1d,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                stencil_usage: Some(ImageUsage::DEPTH_STENCIL_ATTACHMENT),
                format: vulkano::format::Format::D16_UNORM,
                extent: [window.inner_size().width, window.inner_size().height, 1],
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
    
        // Create a framebuffer for each swapchain image
        let fbos: Vec<Arc<Framebuffer>> = swapchain_info
            .images
            .iter()
            .map(|img| create_framebuffer_object(render_pass.clone(), img.clone(), depth_image.clone()))
            .collect();
    
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::Repeat; 3],
                min_filter: vulkano::image::sampler::Filter::Linear,
                mag_filter: vulkano::image::sampler::Filter::Linear,
                ..Default::default()
            },
        )
        .unwrap();
    
        let graphics_pipeline =
            deferred_pipeline::DeferredPipeline::new(window.clone(), device.clone(), render_pass.clone(), 0);
        // let lighting_pipeline = 
        //     lighting_pipeline::LightingPipeline::new(window.clone(), device.clone(), render_pass.clone(), 1);
    
        RenderTargetInfo {
            pipeline: graphics_pipeline.pipeline,
            image_sampler: sampler,
            // render_pass: render_pass,
            fbos: fbos,
        }
    }
    

    pub fn get_device_type(&self) -> PhysicalDeviceType {
        self.physical.properties().device_type
    }

    pub fn get_first_queue(&self) -> Arc<Queue> {
        // self.device_queues.iter().next().unwrap().clone()
        self.queue.clone()
    }

    pub fn get_logical_device(&self) -> Arc<Device> {
        self.logical.clone()
    }

    pub fn get_physical_device(&self) -> Arc<physical::PhysicalDevice> {
        self.physical.clone()
    }

    pub fn get_swapchain(&self) -> Arc<Swapchain> {
        self.swapchain_info.swapchain.clone()
    }

    pub fn get_graphics_pipeline(&self) -> Arc<GraphicsPipeline> {
        self.render_target.pipeline.clone()
    }


    pub fn refresh_instance_swapchain(&mut self, window: Arc<Window>) {
        let dimensions = window.inner_size().into();
        let (new_swapchain, new_images) = self
            .swapchain_info
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: dimensions,
                ..self.swapchain_info.swapchain.create_info()
            })
            .unwrap();
    
        self.swapchain_info.swapchain = new_swapchain;
        self.swapchain_info.images = new_images;
    }
    



    // creates a general buffer allocator
    fn create_buffer_allocator(device: Arc<Device>) -> GenericBufferAllocator {
        // We create memory allocator as an Arc because the Buffer::from_iter takes the allocator as an Arc copy
        Arc::new(StandardMemoryAllocator::new_default(device))
    }

    // Creates command buffer allocators required to be submitted to a render pass
    fn create_command_buffer_allocator(device: Arc<Device>) -> Arc<StandardCommandBufferAllocator> {
        Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                ..Default::default()
            },
        ))
    }

    fn create_descriptor_set_allocator(
        logical_device: Arc<Device>,
    ) -> Arc<StandardDescriptorSetAllocator> {
        Arc::new(StandardDescriptorSetAllocator::new(
            logical_device,
            Default::default(),
        ))
    }

    fn create_render_pass(
        device: Arc<Device>,
        swapchain_info: &VulkanSwapchainInfo,
    ) -> Arc<RenderPass> {
        // need 3 things: device Arc, attachments, and a pass
        let format = swapchain_info.swapchain.create_info().image_format.clone();

        // vulkano::single_pass_renderpass!(
        vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },

                depth: {
                    format: vulkano::format::Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare
                }
            },

            // pass: {
            //     {
            //         color: [color],
            //         depth_stencil: {depth},
            //     }
            // }

            passes: [{
                color: [color],
                depth_stencil: {depth},
                input: []
            }],
        )
        .unwrap()
    }

}

fn create_swapchain(
    window: Arc<Window>,
    physical_device: Arc<physical::PhysicalDevice>,
    logical_device: Arc<Device>,
    surface: Arc<Surface>,
) -> VulkanSwapchainInfo {
    /*
       Creating the Swapchain
    */

    let swapchain_capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    let window_dimensions = window.inner_size();
    let composite_alpha = swapchain_capabilities
        .supported_composite_alpha
        .into_iter()
        .next()
        .unwrap();
    let image_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;

    for i in physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()
    {
        println!("Surface Formats: {:?}", i);
    }
    for k in swapchain_capabilities.supported_composite_alpha {
        println!("Composite Alpha: {:?}", k);
    }

    let create_info = SwapchainCreateInfo {
        min_image_count: swapchain_capabilities.min_image_count + 1, // its good to have at least one more image than minimal to provide a bit more freedom to image queueing in the swapchain
        image_usage: ImageUsage::COLOR_ATTACHMENT,

        composite_alpha: composite_alpha,
        image_extent: window_dimensions.into(),
        image_format: image_format,

        ..Default::default()
    };

    println!("Successfully created the swapchain info");

    let (surface_swapchain, images) =
        Swapchain::new(logical_device.clone(), surface.clone(), create_info.clone())
            .expect("Failed to create surface swapchain");
    println!("Created the swapchain");

    VulkanSwapchainInfo {
        images: images,
        swapchain: surface_swapchain,
        // create_info: create_info,
    }
}


fn create_framebuffer_object(
    render_pass: Arc<RenderPass>,
    image: Arc<Image>,
    depth_stencil_image: Arc<Image>,
) -> Arc<Framebuffer> {
    let view = ImageView::new_default(image.clone()).unwrap();
    let depth_stencil_view = ImageView::new_default(depth_stencil_image.clone()).unwrap();

    Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![view, depth_stencil_view],
            ..Default::default()
        },
    )
    .unwrap()
}

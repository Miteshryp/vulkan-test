use std::{any::TypeId, cell::RefCell, sync::Arc};

use winit::{event_loop::EventLoop, window::Window};

use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::{self, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::{
        allocator::{
            AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
        },
        MemoryType,
    },
    pipeline::GraphicsPipeline,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{self, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    Version, VulkanError, VulkanLibrary,
};

use crate::graphics::{
    pipelines::{
        base_pipeline::GraphicsPipelineInterface,
        basic_pipeline,
        deferred_pipeline::{self, DeferredPipeline},
        lighting_pipeline::{self, LightingPipeline},
    },
    renderer::{deferred_renderer::DeferredRenderer, VulkanRenderer},
};

// type name for buffer allocator
type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
type CommandBufferType = Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

type PrimaryAutoCommandBuilderType = AutoCommandBufferBuilder<
    PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
    Arc<StandardCommandBufferAllocator>,
>;

// pub struct RenderTargetInfo {
//     // pub pipeline: Arc<GraphicsPipeline>,
//     // pub lighting_pipeline: Arc<GraphicsPipeline>,
//     // pub pipeline: DeferredPipeline,
//     // pub lighting_pipeline: LightingPipeline,
//     // // pub render_pass: Arc<RenderPass>,
//     // pub fbos: Vec<Arc<Framebuffer>>,
//     // pub attachments: RenderPassAttachments,
//     // pub image_sampler: Arc<Sampler>,
//     pub renderer: DeferredRenderer,
// }



/// Stores all universal allocators in a vulkan instance
/// 
/// The data in this struct is to be used by all graphics related purposes to 
/// allocate host(CPU) or device(GPU) based memory through vulkan.
/// 
/// We do not allow creation of multiple buffer allocators to ensure efficiency
/// and reduce memory usage.
/// In the future, this struct may include multiple allocator for the same buffer
/// with different configurations if the engine may require it. 
#[derive(Clone)]
pub struct InstanceAllocators {
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_allocator: GenericBufferAllocator,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}


/// Stores all vulkan API based metadata required for calling vulkan based 
/// API functions.
/// 
/// An instance of this struct is stored in the global vulkan instance,
/// which is then responsible for passing it as plugins appropriately 
pub struct VulkanTarget {
    pub logical: Arc<Device>,
    pub physical: Arc<vulkano::device::physical::PhysicalDevice>,
    pub surface: Arc<Surface>,
    pub queue: Arc<Queue>,
    pub vulkan_instance: Arc<Instance>,
    pub swapchain_info: VulkanSwapchainInfo,
}

pub struct VulkanInstanceState {
    
    pub target: VulkanTarget,
    // pub renderer: RefCell<DeferredRenderer>,
    pub allocators: InstanceAllocators,

    // pub renderers: 
}


/// Store swapchain specific metadata
#[derive(Clone)]
pub struct VulkanSwapchainInfo {
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<Image>>,
}




// Some sort of instance builder? 
// - How will this help resolve the issue we're having?
// - How will this function


// New Rendering system
// 1. Push the data into a upload structure
// 2. Get the device buffer mapping, where the data is eventually going to end up. The device buffer which is extracted from this mapping is going to be passed into the rendering command buffer construction.
// 3. create a command buffer in the upload structure
// 4. get the per frame upload command buffer future and wait before executing the rendering command buffer

impl VulkanInstanceState {


    /// Initialise the vulkan instance by getting initialising, allocating 
    /// and securing window construct for the vulkan runtime.
    /// 
    /// The [`window`](Arc<Window>) is a handle to the winit window wrapped in an 
    /// Arc, which can fetched from a window object.
    pub fn initialise(window: Arc<Window>, el: &EventLoop<()>) -> Self {

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
        let required_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_push_descriptor: true,
            // khr_display_swapchain: true,
            // ext_debug_marker: true,
            ..Default::default()
        };

        // TODO: Explore the possible features
        let requested_device_features = Features {
            wide_lines: true,
            ..vulkano::device::Features::empty()
        };

        let (physical_device, queue_family_index) = vulkan_instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
            .filter(|p| {
                p.supported_extensions()
                    .contains(&required_device_extensions)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // Checking for proper support of
                        // 1. Surface support for the created surface
                        // 2. Required Queue family availability

                        // Adding check for required graphics features
                        p.supported_features()
                            .intersects(&requested_device_features)
                            && q.queue_flags.intersects(QueueFlags::GRAPHICS)
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


        // Creating Logical Device
        let (logical_device, mut queues_iterator) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    ..Default::default()
                }],
                enabled_extensions: required_device_extensions,
                enabled_features: requested_device_features,
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
            command_buffer_allocator: VulkanInstanceState::create_command_buffer_allocator(
                logical_device.clone(),
            ),
            memory_allocator: VulkanInstanceState::create_buffer_allocator(logical_device.clone()),
            descriptor_set_allocator: VulkanInstanceState::create_descriptor_set_allocator(
                logical_device.clone(),
            ),
        };

        // let render_target_info = VulkanInstance::create_render_target(
        //     window.clone(),
        //     logical_device.clone(),
        //     &swapchain,
        //     allocators.memory_allocator.clone(),
        // );

        // renderer has to be optional
        // we create the renderer after creating the instance

        
        let instance = VulkanInstanceState {

            target: VulkanTarget {
                logical: logical_device,
                physical: physical_device,
                surface,
                swapchain_info: swapchain,
                queue: queues_iterator.next().unwrap(),
                vulkan_instance,
            },
            

            // renderer: DeferredRenderer::new(
            //     instance.logical.clone(),
            //     window.clone(),
            //     &instance
            //     // &instance.swapchain_info,
            //     // queue_family_index as u32,
            //     // instance.allocators.command_buffer_allocator.clone(),
            //     // instance.allocators.memory_allocator.clone(),
            //     // instance.allbocators.descriptor_set_allocator.clone()
            // ),
            allocators: allocators,
        };

        instance
    }

    // pub fn get_renderer<RendererType: VulkanRendererInterface>(&self) {
    // }

    pub fn get_device_type(&self) -> PhysicalDeviceType {
        self.target.physical.properties().device_type
    }

    pub fn get_first_queue(&self) -> Arc<Queue> {
        self.target.queue.clone()
    }

    pub fn get_logical_device(&self) -> Arc<Device> {
        self.target.logical.clone()
    }

    pub fn get_physical_device(&self) -> Arc<physical::PhysicalDevice> {
        self.target.physical.clone()
    }

    pub fn get_swapchain(&self) -> Arc<Swapchain> {
        self.target.swapchain_info.swapchain.clone()
    }

    pub fn refresh_instance_swapchain(&mut self, window: Arc<Window>) {
        let dimensions = window.inner_size().into();

        self.target.surface = Surface::from_window(self.target.vulkan_instance.clone(), window.clone()).unwrap();

        let (new_swapchain, new_images) = self.target
            .swapchain_info
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: dimensions,
                ..self.target.swapchain_info.swapchain.create_info()
            })
            .unwrap();

        self.target.swapchain_info.swapchain = new_swapchain;
        self.target.swapchain_info.images = new_images;
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

    fn create_deferred_render_pass(
        device: Arc<Device>,
        swapchain_info: &VulkanSwapchainInfo,
    ) -> Arc<RenderPass> {
        // need 3 things: device Arc, attachments, and a pass
        let format = swapchain_info.swapchain.create_info().image_format.clone();

        vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {

                // each attachment has to be defined in this EXACT format order:
                // format, samples, load_op, store_op
                final_color: {
                    format: format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                color: {
                    format: vulkano::format::Format::A2B10G10R10_UNORM_PACK32,
                    // format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },

                normals: {
                    format: vulkano::format::Format::R16G16B16A16_SFLOAT,
                    // format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },

                depth: {
                    format: vulkano::format::Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare
                }
            },

            passes: [
                {
                    color: [color, normals],
                    depth_stencil: {depth},
                    input: []
                },
                {
                    color: [final_color],
                    depth_stencil: {}, // depth calculations were done by the previous pass
                    input: [color, normals]
                }
            ]
        )
        .unwrap()
    }

    fn create_basic_render_pass(
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
        // present_mode: swapchain::PresentMode::Fifo,
        // present_mode: swapchain::PresentMode::Immediate,
        // present_mode: swapchain::PresentMode::Mailbox,
        // present_mode: swapchain::PresentMode::FifoRelaxed,
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

fn create_deferred_framebuffer_object(
    render_pass: Arc<RenderPass>,
    final_image_view: Arc<ImageView>,
    depth_stencil_view: Arc<ImageView>,
    color_view: Arc<ImageView>,
    normal_view: Arc<ImageView>,
) -> Arc<Framebuffer> {
    Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![
                final_image_view,
                color_view,
                normal_view,
                depth_stencil_view,
            ],
            ..Default::default()
        },
    )
    .unwrap()
}

fn create_basic_framebuffer_object(
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

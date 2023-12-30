mod graphics_pack;
// mod vertex;

use graphics_pack::{
    buffers::{
        primitives::{Vec3, VertexPoint},
        IndexBuffer, VertexBuffer,
    },
    shaders::{self, fs},
};

use std::{process::exit, sync::Arc, time::SystemTime};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        self,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        CommandBufferUsage, PrimaryAutoCommandBuffer,
    },
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferLevel,
        CopyImageToBufferInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{self, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Properties, Queue, QueueCreateInfo,
        QueueFamilyProperties, QueueFlags,
    },
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    instance::InstanceCreateInfo,
    memory::allocator::{
        AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            self,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil,
            input_assembly::InputAssemblyState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo},
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::spirv::LoopControl,
    swapchain::{self, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};
use winit::{
    event::{ElementState, Event, Modifiers, MouseButton, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::{Key, ModifiersState},
    platform::modifier_supplement::KeyEventExtModifierSupplement,
    window::{Window, WindowBuilder},
};

use crate::graphics_pack::buffers::{BufferOps, BufferOptions};

// type name for buffer allocator
type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
type CommandBufferType = Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

struct RenderTargetInfo {
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    fbos: Vec<Arc<Framebuffer>>,
}

struct InstanceAllocators {
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: GenericBufferAllocator,
}
struct VulkanInstance {
    logical: Arc<Device>,
    physical: Arc<vulkano::device::physical::PhysicalDevice>,
    surface: Arc<Surface>,
    // device_queues: Vec<Arc<Queue>>,
    queue: Arc<Queue>,
    swapchain_info: VulkanSwapchainInfo,
    render_target: RenderTargetInfo,

    allocators: InstanceAllocators,
}

struct VulkanSwapchainInfo {
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>
}

impl VulkanInstance {
    fn initialise(&mut self, window: Arc<Window>, eltw: &EventLoop<()>) {
        todo!()
    }

    fn get_device_type(&self) -> PhysicalDeviceType {
        self.physical.properties().device_type
    }

    fn get_first_queue(&self) -> Arc<Queue> {
        // self.device_queues.iter().next().unwrap().clone()
        self.queue.clone()
    }

    fn get_logical_device(&self) -> Arc<Device> {
        self.logical.clone()
    }

    fn get_physical_device(&self) -> Arc<physical::PhysicalDevice> {
        self.physical.clone()
    }

    fn get_swapchain(&self) -> Arc<Swapchain> {
        self.swapchain_info.swapchain.clone()
    }

    fn get_graphics_pipeline(&self) -> Arc<GraphicsPipeline> {
        self.render_target.pipeline.clone()
    }
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

// fn create_render_pass(device: Arc<Device>, instance: &VulkanInstance) -> Arc<RenderPass> {
fn create_render_pass(
    device: Arc<Device>,
    swapchain_info: &VulkanSwapchainInfo,
) -> Arc<RenderPass> {
    // need 3 things: device Arc, attachments, and a pass
    let format = swapchain_info.swapchain.create_info().image_format.clone();

    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },

            // depth: {
            //     format: vulkano::format::Format::D16_UNORM,
            //     samples: 1,
            //     load_op: Clear,
            //     store_op: DontCare
            // }
        },
        pass: {
            color: [color],
            depth_stencil: {}
            // depth_stencil: {depth},
        },
    )
    .unwrap()
}

// Creating the graphics pipeline objects by binding proper FBO and shaders
fn create_graphics_pipeline(
    window: Arc<Window>,
    logical_device: Arc<Device>,
    render_pass: Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let vertex_shader = shaders::get_vertex_shader(logical_device.clone())
        .entry_point("main")
        .unwrap();
    let fragment_shader = shaders::get_fragment_shader(logical_device.clone())
        .entry_point("main")
        .unwrap();

    let vertex_shader_input_state = VertexPoint::per_vertex()
        .definition(&vertex_shader.info().input_interface)
        .unwrap();

    // This creation moves the vertex and fragment shaders,
    // so we cannot use those objects after this point
    let pipeline_stages = [
        PipelineShaderStageCreateInfo::new(vertex_shader),
        PipelineShaderStageCreateInfo::new(fragment_shader),
    ];

    let window_size = window.inner_size();

    let pipeline_layout = PipelineLayout::new(
        logical_device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stages)
            .into_pipeline_layout_create_info(logical_device.clone())
            .unwrap(),
    )
    .unwrap();

    GraphicsPipeline::new(
        logical_device,
        None,
        GraphicsPipelineCreateInfo {
            // Defining the stages of the pipeline
            // we only have 2 => vertex shader and a fragment shader
            stages: pipeline_stages.into_iter().collect(),

            // Defining the mapping of vertex to the vertex shader inputs
            vertex_input_state: Some(vertex_shader_input_state),

            // Setting a fixed viewport for now
            viewport_state: Some(ViewportState {
                viewports: [Viewport {
                    offset: [0.0, 0.0],
                    extent: window_size.into(),
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
                ..Default::default()
            }),

            // describes drawing primitives, default is a triangle
            input_assembly_state: Some(InputAssemblyState::default()),

            rasterization_state: Some(Default::default()),
            multisample_state: Some(Default::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),

            // Concerns with First pass of render pass
            subpass: Some(subpass.into()),

            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        },
    )
    .unwrap()
}

fn get_framebuffer_object(render_pass: Arc<RenderPass>, image: Arc<Image>) -> Arc<Framebuffer> {
    let view = ImageView::new_default(image.clone()).unwrap();
    // let depth_buffer = ImageView::new_default(vulkano::image::Image::s)

    Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap()
}

fn create_buffer_from_iter<T, I>(
    allocator: GenericBufferAllocator,
    iter: I,
    buffer_usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Subbuffer<[T]>
where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: buffer_usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: memory_type_filter,
            ..Default::default()
        },
        iter,
    )
    .unwrap()
}

fn create_buffer_from_data<T>(
    allocator: GenericBufferAllocator,
    data: T,
    buffer_usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Subbuffer<T>
where
    T: BufferContents,
{
    Buffer::from_data(
        allocator,
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

#[allow(unused_variables)]
#[allow(unused_assignments)]
fn winit_handle_window_events(event: WindowEvent, window_target: &EventLoopWindowTarget<()>) {
    let mut modifiers = ModifiersState::default();

    match event {
        WindowEvent::CloseRequested => {
            window_target.exit();
            exit(0);
        }

        WindowEvent::MouseInput {
            device_id,
            state,
            button,
        } => match state {
            ElementState::Pressed => match button {
                MouseButton::Left => {
                    println!("Left click detected");
                }
                MouseButton::Right => {
                    println!("Right click detected")
                }
                _ => (),
            },
            _ => (),
        },

        // Updating the current modifiers on the keyboard
        WindowEvent::ModifiersChanged(new) => {
            println!("Modifier changed");
            modifiers = new.state();
        }

        WindowEvent::KeyboardInput {
            event: key_event, ..
        } => match key_event.state {
            ElementState::Pressed => match key_event.key_without_modifiers().as_ref() {
                Key::Character("w") => {
                    println!("Forward");
                }
                Key::Character("a") => {
                    println!("Left");
                }
                Key::Character("s") => {
                    println!("Backward");
                }
                Key::Character("d") => {
                    println!("Right");
                }
                _ => (),
            },
            _ => (),
        },
        _ => (),
    }
    return;
}

fn create_window() -> (Arc<Window>, EventLoop<()>) {
    let winit_event_loop = event_loop::EventLoop::new().unwrap();
    let winit_window: Arc<Window> = Arc::new(
        WindowBuilder::new()
            // .with_transparent(true)
            .build(&winit_event_loop)
            .unwrap(),
    );

    // setting the control flow for the event loop
    winit_event_loop.set_control_flow(event_loop::ControlFlow::Poll);

    (winit_window, winit_event_loop)
}

fn start_window_event_loop(window: Arc<Window>, el: EventLoop<()>, mut instance: VulkanInstance) {
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let color1 = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    let color2 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let color3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    let vertices = Vec::from([
        VertexPoint {
            position: Vec3 {
                x: -1.0,
                y: -1.0,
                z: 0.5,
            },
            color: color1.clone(),
        },
        VertexPoint {
            position: Vec3 {
                x: -1.0,
                y: 1.0,
                z: 0.5,
            },
            color: color2.clone(),
        },
        VertexPoint {
            position: Vec3 {
                x: 1.0,
                y: 1.0,
                z: 0.5,
            },
            color: color2.clone(),
        },
        VertexPoint {
            position: Vec3 {
                x: 1.0,
                y: -1.0,
                z: 0.5,
            },
            color: color3.clone(),
        },
    ]);

    let indicies: Vec<u32> = Vec::from([0, 1, 2, 0, 2, 3]);
    let mut command_buffers = create_command_buffers(&instance, vertices.clone(), indicies.clone());

    el.set_control_flow(ControlFlow::Poll);
    let _ = el.run(move |app_event, elwt| match app_event {
        // Window based Events
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
        }

        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            println!("Redrawing");
            // draw call

            if window_resized || recreate_swapchain {
                println!("Instance surface: {:?}", instance.surface);
                // recreating swapchains


                // WARNING: Creating and refreshing swapchain are 2 different things
                // For some reason, the AMD graphics card was not giving an error for this,
                // but this behavior crashes the program on the NVIDIA GPU.
                // Be careful about this operation.
                // instance.swapchain_info = create_swapchain(
                //     window.clone(),
                //     instance.get_physical_device(),
                //     instance.get_logical_device(),
                //     instance.surface.clone(),
                // );

                // let dimensions = window.inner_size().into();
                // let (new_swapchain, new_images) = instance
                //     .swapchain_info
                //     .swapchain
                //     .recreate(SwapchainCreateInfo {
                //         image_extent: dimensions,
                //         ..instance.swapchain_info.swapchain.create_info()
                //     })
                //     .unwrap();

                // instance.swapchain_info.swapchain = new_swapchain;
                // instance.swapchain_info.images = new_images;

                refresh_instance_swapchain(window.clone(), &mut instance);

                // refreshing the render targets
                // recreating the render pass, fbos, pipeline and command buffers with the new swapchain images
                instance.render_target = refresh_render_target(
                    window.clone(),
                    instance.get_logical_device(),
                    &instance.swapchain_info,
                );

                window_resized = false;
                recreate_swapchain = false;
            }
            command_buffers = create_command_buffers(&instance, vertices.clone(), indicies.clone());

            println!("Build command buffers");

            let (image_index, is_suboptimal, acquired_future) = match swapchain::acquire_next_image(
                instance.swapchain_info.swapchain.clone(),
                None,
            )
            .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    recreate_swapchain = false;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image from swapchain"),
            };

            if is_suboptimal {
                recreate_swapchain = true;
            }

            // Future is a point where GPU gets access to the data
            let future = sync::now(instance.get_logical_device())
                .join(acquired_future)
                .then_execute(
                    instance.get_first_queue(),
                    command_buffers[image_index as usize].clone(),
                )
                .unwrap()
                .then_swapchain_present(
                    instance.get_first_queue(),
                    SwapchainPresentInfo::swapchain_image_index(
                        instance.get_swapchain(),
                        image_index,
                    ),
                )
                .then_signal_fence_and_flush();
            // .unwrap();

            match future.map_err(Validated::unwrap) {
                Ok(future) => {
                    // Wait for the GPU to finish.
                    future.wait(None).unwrap();
                    window.request_redraw();
                }
                Err(VulkanError::OutOfDate) => {
                    recreate_swapchain = true;
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                }
            }

            // Steps to take
            // 1. Recreate the swapchain if it has been invalidated
            //      a. If the swapchain has been recreated, then recreate the framebuffer objects for each swapchain image as well
            //      b. Create the command buffers for each fbo created.
            // 2. Pass the fbo to the render call along with the graphics pipeline
            // 3. All the futures received
        }

        Event::WindowEvent { window_id, event } => {
            // Handling the window event if the event is bound to the owned window
            if window_id == window.id() {
                winit_handle_window_events(event, elwt);
            }
        }

        Event::LoopExiting => {
            println!("Exiting the event loop");
            exit(0);
        }
        _ => (),
    });
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

    let (surface_swapchain, images) =
        Swapchain::new(logical_device.clone(), surface.clone(), create_info.clone())
            .expect("Failed to create surface swapchain");

    VulkanSwapchainInfo {
        images: images,
        swapchain: surface_swapchain,
        // create_info: create_info,
    }
}

fn refresh_instance_swapchain(
    window: Arc<Window>,
    instance: &mut VulkanInstance,
) {
    let dimensions = window.inner_size().into();
    let (new_swapchain, new_images) = instance
        .swapchain_info
        .swapchain
        .recreate(SwapchainCreateInfo {
            image_extent: dimensions,
            ..instance.swapchain_info.swapchain.create_info()
        })
        .unwrap();

    instance.swapchain_info.swapchain = new_swapchain;
    instance.swapchain_info.images = new_images;
}

#[allow(unused_variables)]
#[allow(unused_assignments)]
fn initialise_vulkan_runtime(window: Arc<Window>, el: &EventLoop<()>) -> VulkanInstance {
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
            // max_api_version: Some(Version {
            //     major: 1,
            //     minor: 3,
            //     ..Default::default()
            // }),
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
        ..Default::default()
    };

    // 3 components to check for
    // a. Swapchain support in window
    // b. Surface support in device
    // c. Queue family support in device queue

    let (physical_device, queue_family_index) = vulkan_instance
        .enumerate_physical_devices()
        .expect("Failed to enumerate physical devices")
        // .filter(|p| {
        //     p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        // })
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
            // enabled_features: Features {
            //     dynamic_rendering: true,
            //     ..Features::empty()
            // },
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

    // TODO: Explore graphics pipeline or command builder extensibilities

    let render_target_info =
        refresh_render_target(window.clone(), logical_device.clone(), &swapchain);
    let allocators = InstanceAllocators {
        command_buffer_allocator: create_command_buffer_allocator(logical_device.clone()),
        memory_allocator: create_buffer_allocator(logical_device.clone()),
    };

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

    // return (logical_device, device_queues, surface_swapchain);

    /*
        FIXME:
        TODO: Fix the steps and shift it to the top of the file
        Steps to initialise vulkan instance

        Step 1. Query for the available physical devices based on the requirements
        Step 2. Select a physical device to be used
        Step 3. Create a Logical device based on the (VkQueue) queue types that we want to use
        Step 4. Initialise the window for the application (using winit)
        Step 5. Create a Vulkan Surface to render our graphics to which would be a reference to the current window
        Step 6. Create a SwapChain to render our images to. This swapchain will then swap the images from
        Step 7. Create command buffers
        Step 8. Create Graphics pipeline
        Step 9. Create the vertex and fragment shaders
    */

    // Err("Failed to initialise vulkan runtime")
    // Ok(())
}

static mut START: Option<SystemTime> = None;

fn refresh_render_target(
    window: Arc<Window>,
    device: Arc<Device>,
    swapchain_info: &VulkanSwapchainInfo,
) -> RenderTargetInfo {
    let render_pass = create_render_pass(device.clone(), swapchain_info); // defines the schema information required for configuring the output of shaders to the framebuffer

    // Create a framebuffer for each swapchain image
    let fbos: Vec<Arc<Framebuffer>> = swapchain_info
        .images
        .iter()
        .map(|img| get_framebuffer_object(render_pass.clone(), img.clone()))
        .collect();

    let graphics_pipeline =
        create_graphics_pipeline(window.clone(), device.clone(), render_pass.clone());

    RenderTargetInfo {
        pipeline: graphics_pipeline,
        render_pass: render_pass,
        fbos: fbos,
    }
}

/*
 Different from get_command_buffers
 This function creates the final command buffers for a swapchain state from start to finish,
 including building the renderpass, fbo, and the graphics pipeline

 Every time a swapchain is invalidated, these steps need to be carried out to reconfigure the command buffers.
*/
fn create_command_buffers(
    // window: Arc<Window>,
    instance: &VulkanInstance,
    vertices: Vec<VertexPoint>,
    indicies: Vec<u32>,
) -> Vec<CommandBufferType> {
    // buffer allocator for memory buffer objects
    let memory_allocator = instance.allocators.memory_allocator.clone();

    let vertex_buffer = graphics_pack::buffers::VertexBuffer::new(
        instance.allocators.memory_allocator.clone(),
        vertices,
        Default::default(),
    );

    let index_buffer: IndexBuffer = graphics_pack::buffers::IndexBuffer::new(
        instance.allocators.memory_allocator.clone(),
        indicies,
        Default::default(),
    );

    // let depth_stencil_buffer = Buffer::from(value)

    let mut f = 0.0;
    unsafe {
        f = START.unwrap().elapsed().unwrap().as_secs_f32() as f32; //std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as f32;
    }
    println!("{f}");
    let uniform_buffer = create_buffer_from_data(
        memory_allocator.clone(),
        shaders::vs::Data { view: f },
        BufferUsage::UNIFORM_BUFFER,
        MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    );

    let command_buffers = build_fbo_command_buffers_for_pipeline(
        instance.get_logical_device(),
        instance.allocators.command_buffer_allocator.clone(),
        instance.get_first_queue(),
        instance.get_graphics_pipeline(),
        &instance.render_target.fbos,
        vertex_buffer,
        index_buffer,
        uniform_buffer,
    );

    return command_buffers;
}

// Only builds the command buffers

// Components required:
// 1. Logical device
// 2. Command buffer allocator
// 3. Physical device queue (to get queue index)
// 4. Graphics pipeline instance
// 5. Frame buffer array
// 6. vertex_buffer
// 7. uniform_buffer
fn build_fbo_command_buffers_for_pipeline(
    // instance: &VulkanInstance,
    logical_device: Arc<Device>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    fbos: &Vec<Arc<Framebuffer>>,

    // vertex_buffer: Subbuffer<[vertex::VertexPoint]>,
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
    uniform_buffer: Subbuffer<shaders::vs::Data>,
) -> Vec<CommandBufferType> {
    // I need:
    // Command buffer built using AutoCommandBufferBuilder - done
    // render pass created for data
    // FBO - done
    // graphics pipeline object

    let queue_family_index = queue.queue_family_index();

    let descriptor_set_index: usize = 0;
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(logical_device, Default::default());
    let descriptor_set_layout = graphics_pipeline
        .layout()
        .set_layouts()
        .get(descriptor_set_index)
        .unwrap();

    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer)], // 0 is the binding
        [],
    )
    .unwrap();

    // let fbos = &instance.render_target.fbos;

    println!("Fine till here");

    fbos.into_iter()
        .map(move |fb| {
            let mut command_builder = AutoCommandBufferBuilder::primary(
                &cb_allocator,
                queue_family_index,
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            // FIXME: we are doing redundant copies for every framebuffer anyways
            let temp_vertex_buffer = vertex_buffer.clone();
            let (vertex_subbuffer, vertex_count) = temp_vertex_buffer.consume();

            let temp_index_buffer = index_buffer.clone();
            let (index_subbuffer, index_count) = temp_index_buffer.consume();

            // TODO: vertex and instance count are hardcoded.
            // Create rendering structure to parameterize them
            // based on the size of vertex buffer.
            command_builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(fb.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_index_buffer(index_subbuffer)
                .unwrap()
                .bind_vertex_buffers(0, vertex_subbuffer)
                .unwrap()
                .bind_pipeline_graphics(graphics_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    graphics_pipeline.clone().layout().clone(),
                    0,
                    descriptor_set.clone(),
                )
                .unwrap()
                .draw(vertex_count, 1, 0, 0)
                .unwrap()
                .end_render_pass(Default::default())
                .unwrap();

            let cb = command_builder.build().unwrap();
            return cb;
        })
        .collect()
}

fn get_image_buffer(allocator: GenericBufferAllocator) -> Arc<vulkano::image::Image> {
    vulkano::image::Image::new(
        allocator.clone(),
        vulkano::image::ImageCreateInfo {
            image_type: vulkano::image::ImageType::Dim2d,
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT
                | vulkano::image::ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap()
}

// ISSUE
// 1. We are unable to pass uniforms directly into the fragment shader
//      It results in the error: create_info.pool_sizes is empty
// 2. Depth stencil buffer is not setup. vertices are rendered in the order
//      that they are given

/*
   Steps to setup stencil buffer

   create a depth image.
   update the render pass.
   set a new depth stencil state in the pipeline.
   update the framebuffers.
   update draw function.
   clean.
*/

fn main() {
    unsafe {
        START = Some(SystemTime::now());
    }

    let (window, elwt) = create_window();
    let mut vulkan_instance: VulkanInstance = initialise_vulkan_runtime(window.clone(), &elwt);

    println!("First initialisation successful");

    start_window_event_loop(window.clone(), elwt, vulkan_instance);
}

// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

// use std::{error::Error, sync::Arc};
// use vulkano::{
//     buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
//     command_buffer::{
//         allocator::StandardCommandBufferAllocator, CommandBufferLevel,
//         CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
//         SubpassContents, AutoCommandBufferBuilder,
//     },
//     device::{
//         physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
//         QueueFlags,
//     },
//     image::{view::ImageView, Image, ImageUsage},
//     instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
//     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
//     pipeline::{
//         graphics::{
//             color_blend::{ColorBlendAttachmentState, ColorBlendState},
//             input_assembly::InputAssemblyState,
//             multisample::MultisampleState,
//             rasterization::RasterizationState,
//             vertex_input::{Vertex, VertexDefinition},
//             viewport::{Viewport, ViewportState},
//             GraphicsPipelineCreateInfo,
//         },
//         layout::PipelineDescriptorSetLayoutCreateInfo,
//         DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
//     },
//     render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
//     swapchain::{
//         acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
//     },
//     sync::{self, GpuFuture},
//     Validated, VulkanError, VulkanLibrary,
// };
// use winit::{
//     event::{Event, WindowEvent},
//     event_loop::{ControlFlow, EventLoop},
//     window::WindowBuilder,
// };

// fn main() -> Result<(), impl Error> {
//     let event_loop = EventLoop::new().unwrap();

//     let library = VulkanLibrary::new().unwrap();

//     // The first step of any Vulkan program is to create an instance.
//     //
//     // When we create an instance, we have to pass a list of extensions that we want to enable.
//     //
//     // All the window-drawing functionalities are part of non-core extensions that we need to
//     // enable manually. To do so, we ask `Surface` for the list of extensions required to draw to
//     // a window.
//     let required_extensions = Surface::required_extensions(&event_loop);//.unwrap();

//     // Now creating the instance.
//     let instance = Instance::new(
//         library,
//         InstanceCreateInfo {
//             // Enable enumerating devices that use non-conformant Vulkan implementations.
//             // (e.g. MoltenVK)
//             flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
//             enabled_extensions: required_extensions,
//             ..Default::default()
//         },
//     )
//     .unwrap();

//     // The objective of this example is to draw a triangle on a window. To do so, we first need to
//     // create the window. We use the `WindowBuilder` from the `winit` crate to do that here.
//     //
//     // Before we can render to a window, we must first create a `vulkano::swapchain::Surface`
//     // object from it, which represents the drawable surface of a window. For that we must wrap the
//     // `winit::window::Window` in an `Arc`.
//     let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
//     let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

//     // Choose device extensions that we're going to use. In order to present images to a surface,
//     // we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
//     let device_extensions = DeviceExtensions {
//         khr_swapchain: true,
//         ..DeviceExtensions::empty()
//     };

//     // We then choose which physical device to use. First, we enumerate all the available physical
//     // devices, then apply filters to narrow them down to those that can support our needs.
//     let (physical_device, queue_family_index) = instance
//         .enumerate_physical_devices()
//         .unwrap()
//         .filter(|p| {
//             // Some devices may not support the extensions or features that your application, or
//             // report properties and limits that are not sufficient for your application. These
//             // should be filtered out here.
//             p.supported_extensions().contains(&device_extensions)
//         })
//         .filter_map(|p| {
//             // For each physical device, we try to find a suitable queue family that will execute
//             // our draw commands.
//             //
//             // Devices can provide multiple queues to run commands in parallel (for example a draw
//             // queue and a compute queue), similar to CPU threads. This is something you have to
//             // have to manage manually in Vulkan. Queues of the same type belong to the same queue
//             // family.
//             //
//             // Here, we look for a single queue family that is suitable for our purposes. In a
//             // real-world application, you may want to use a separate dedicated transfer queue to
//             // handle data transfers in parallel with graphics operations. You may also need a
//             // separate queue for compute operations, if your application uses those.
//             p.queue_family_properties()
//                 .iter()
//                 .enumerate()
//                 .position(|(i, q)| {
//                     // We select a queue family that supports graphics operations. When drawing to
//                     // a window surface, as we do in this example, we also need to check that
//                     // queues in this queue family are capable of presenting images to the surface.
//                     q.queue_flags.intersects(QueueFlags::GRAPHICS)
//                         && p.surface_support(i as u32, &surface).unwrap_or(false)
//                 })
//                 // The code here searches for the first queue family that is suitable. If none is
//                 // found, `None` is returned to `filter_map`, which disqualifies this physical
//                 // device.
//                 .map(|i| (p, i as u32))
//         })
//         // All the physical devices that pass the filters above are suitable for the application.
//         // However, not every device is equal, some are preferred over others. Now, we assign each
//         // physical device a score, and pick the device with the lowest ("best") score.
//         //
//         // In this example, we simply select the best-scoring device to use in the application.
//         // In a real-world setting, you may want to use the best-scoring device only as a "default"
//         // or "recommended" device, and let the user choose the device themself.
//         .min_by_key(|(p, _)| {
//             // We assign a lower score to device types that are likely to be faster/better.
//             match p.properties().device_type {
//                 PhysicalDeviceType::DiscreteGpu => 0,
//                 PhysicalDeviceType::IntegratedGpu => 1,
//                 PhysicalDeviceType::VirtualGpu => 2,
//                 PhysicalDeviceType::Cpu => 3,
//                 PhysicalDeviceType::Other => 4,
//                 _ => 5,
//             }
//         })
//         .expect("no suitable physical device found");

//     // Some little debug infos.
//     println!(
//         "Using device: {} (type: {:?})",
//         physical_device.properties().device_name,
//         physical_device.properties().device_type,
//     );

//     // Now initializing the device. This is probably the most important object of Vulkan.
//     //
//     // An iterator of created queues is returned by the function alongside the device.
//     let (device, mut queues) = Device::new(
//         // Which physical device to connect to.
//         physical_device,
//         DeviceCreateInfo {
//             // A list of optional features and extensions that our program needs to work correctly.
//             // Some parts of the Vulkan specs are optional and must be enabled manually at device
//             // creation. In this example the only thing we are going to need is the `khr_swapchain`
//             // extension that allows us to draw to a window.
//             enabled_extensions: device_extensions,

//             // The list of queues that we are going to use. Here we only use one queue, from the
//             // previously chosen queue family.
//             queue_create_infos: vec![QueueCreateInfo {
//                 queue_family_index,
//                 ..Default::default()
//             }],

//             ..Default::default()
//         },
//     )
//     .unwrap();

//     // Since we can request multiple queues, the `queues` variable is in fact an iterator. We only
//     // use one queue in this example, so we just retrieve the first and only element of the
//     // iterator.
//     let queue = queues.next().unwrap();

//     // Before we can draw on the surface, we have to create what is called a swapchain. Creating a
//     // swapchain allocates the color buffers that will contain the image that will ultimately be
//     // visible on the screen. These images are returned alongside the swapchain.
//     let (mut swapchain, images) = {
//         // Querying the capabilities of the surface. When we create the swapchain we can only pass
//         // values that are allowed by the capabilities.
//         let surface_capabilities = device
//             .physical_device()
//             .surface_capabilities(&surface, Default::default())
//             .unwrap();

//         // Choosing the internal format that the images will have.
//         let image_format = device
//             .physical_device()
//             .surface_formats(&surface, Default::default())
//             .unwrap()[0]
//             .0;

//         // Please take a look at the docs for the meaning of the parameters we didn't mention.
//         Swapchain::new(
//             device.clone(),
//             surface,
//             SwapchainCreateInfo {
//                 // Some drivers report an `min_image_count` of 1, but fullscreen mode requires at
//                 // least 2. Therefore we must ensure the count is at least 2, otherwise the program
//                 // would crash when entering fullscreen mode on those drivers.
//                 min_image_count: surface_capabilities.min_image_count.max(2),

//                 image_format,

//                 // The size of the window, only used to initially setup the swapchain.
//                 //
//                 // NOTE:
//                 // On some drivers the swapchain extent is specified by
//                 // `surface_capabilities.current_extent` and the swapchain size must use this
//                 // extent. This extent is always the same as the window size.
//                 //
//                 // However, other drivers don't specify a value, i.e.
//                 // `surface_capabilities.current_extent` is `None`. These drivers will allow
//                 // anything, but the only sensible value is the window size.
//                 //
//                 // Both of these cases need the swapchain to use the window size, so we just
//                 // use that.
//                 image_extent: window.inner_size().into(),

//                 image_usage: ImageUsage::COLOR_ATTACHMENT,

//                 // The alpha mode indicates how the alpha value of the final image will behave. For
//                 // example, you can choose whether the window will be opaque or transparent.
//                 composite_alpha: surface_capabilities
//                     .supported_composite_alpha
//                     .into_iter()
//                     .next()
//                     .unwrap(),

//                 ..Default::default()
//             },
//         )
//         .unwrap()
//     };

//     let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

//     // We now create a buffer that will store the shape of our triangle. We use `#[repr(C)]` here
//     // to force rustc to use a defined layout for our data, as the default representation has *no
//     // guarantees*.
//     #[derive(BufferContents, Vertex)]
//     #[repr(C)]
//     struct Vertex {
//         #[format(R32G32_SFLOAT)]
//         position: [f32; 2],
//     }

//     let vertices = [
//         Vertex {
//             position: [-0.5, -0.25],
//         },
//         Vertex {
//             position: [0.0, 0.5],
//         },
//         Vertex {
//             position: [0.25, -0.1],
//         },
//     ];
//     let vertex_buffer = Buffer::from_iter(
//         memory_allocator,
//         BufferCreateInfo {
//             usage: BufferUsage::VERTEX_BUFFER,
//             ..Default::default()
//         },
//         AllocationCreateInfo {
//             memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
//                 | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
//             ..Default::default()
//         },
//         vertices,
//     )
//     .unwrap();

//     // The next step is to create the shaders.
//     //
//     // The raw shader creation API provided by the vulkano library is unsafe for various reasons,
//     // so The `shader!` macro provides a way to generate a Rust module from GLSL source - in the
//     // example below, the source is provided as a string input directly to the shader, but a path
//     // to a source file can be provided as well. Note that the user must specify the type of shader
//     // (e.g. "vertex", "fragment", etc.) using the `ty` option of the macro.
//     //
//     // The items generated by the `shader!` macro include a `load` function which loads the shader
//     // using an input logical device. The module also includes type definitions for layout
//     // structures defined in the shader source, for example uniforms and push constants.
//     //
//     // A more detailed overview of what the `shader!` macro generates can be found in the
//     // vulkano-shaders crate docs. You can view them at https://docs.rs/vulkano-shaders/
//     mod vs {
//         vulkano_shaders::shader! {
//             ty: "vertex",
//             src: r"
//                 #version 450

//                 layout(location = 0) in vec2 position;

//                 void main() {
//                     gl_Position = vec4(position, 0.0, 1.0);
//                 }
//             ",
//         }
//     }

//     mod fs {
//         vulkano_shaders::shader! {
//             ty: "fragment",
//             src: r"
//                 #version 450

//                 layout(location = 0) out vec4 f_color;

//                 void main() {
//                     f_color = vec4(1.0, 0.0, 0.0, 1.0);
//                 }
//             ",
//         }
//     }

//     // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
//     // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
//     // manually.

//     // The next step is to create a *render pass*, which is an object that describes where the
//     // output of the graphics pipeline will go. It describes the layout of the images where the
//     // colors, depth and/or stencil information will be written.
//     let render_pass = vulkano::single_pass_renderpass!(
//         device.clone(),
//         attachments: {
//             // `color` is a custom name we give to the first and only attachment.
//             color: {
//                 // `format: <ty>` indicates the type of the format of the image. This has to be one
//                 // of the types of the `vulkano::format` module (or alternatively one of your
//                 // structs that implements the `FormatDesc` trait). Here we use the same format as
//                 // the swapchain.
//                 format: swapchain.image_format(),
//                 // `samples: 1` means that we ask the GPU to use one sample to determine the value
//                 // of each pixel in the color attachment. We could use a larger value
//                 // (multisampling) for antialiasing. An example of this can be found in
//                 // msaa-renderpass.rs.
//                 samples: 1,
//                 // `load_op: Clear` means that we ask the GPU to clear the content of this
//                 // attachment at the start of the drawing.
//                 load_op: Clear,
//                 // `store_op: Store` means that we ask the GPU to store the output of the draw in
//                 // the actual image. We could also ask it to discard the result.
//                 store_op: Store,
//             },
//         },
//         pass: {
//             // We use the attachment named `color` as the one and only color attachment.
//             color: [color],
//             // No depth-stencil attachment is indicated with empty brackets.
//             depth_stencil: {},
//         },
//     )
//     .unwrap();

//     // Before we draw, we have to create what is called a **pipeline**. A pipeline describes how
//     // a GPU operation is to be performed. It is similar to an OpenGL program, but it also contains
//     // many settings for customization, all baked into a single object. For drawing, we create
//     // a **graphics** pipeline, but there are also other types of pipeline.
//     let pipeline = {
//         // First, we load the shaders that the pipeline will use:
//         // the vertex shader and the fragment shader.
//         //
//         // A Vulkan shader can in theory contain multiple entry points, so we have to specify which
//         // one.
//         let vs = vs::load(device.clone())
//             .unwrap()
//             .entry_point("main")
//             .unwrap();
//         let fs = fs::load(device.clone())
//             .unwrap()
//             .entry_point("main")
//             .unwrap();

//         // Automatically generate a vertex input state from the vertex shader's input interface,
//         // that takes a single vertex buffer containing `Vertex` structs.
//         let vertex_input_state = Vertex::per_vertex()
//             .definition(&vs.info().input_interface)
//             .unwrap();

//         // Make a list of the shader stages that the pipeline will have.
//         let stages = [
//             PipelineShaderStageCreateInfo::new(vs),
//             PipelineShaderStageCreateInfo::new(fs),
//         ];

//         // We must now create a **pipeline layout** object, which describes the locations and types of
//         // descriptor sets and push constants used by the shaders in the pipeline.
//         //
//         // Multiple pipelines can share a common layout object, which is more efficient.
//         // The shaders in a pipeline must use a subset of the resources described in its pipeline
//         // layout, but the pipeline layout is allowed to contain resources that are not present in the
//         // shaders; they can be used by shaders in other pipelines that share the same layout.
//         // Thus, it is a good idea to design shaders so that many pipelines have common resource
//         // locations, which allows them to share pipeline layouts.
//         let layout = PipelineLayout::new(
//             device.clone(),
//             // Since we only have one pipeline in this example, and thus one pipeline layout,
//             // we automatically generate the creation info for it from the resources used in the
//             // shaders. In a real application, you would specify this information manually so that you
//             // can re-use one layout in multiple pipelines.
//             PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
//                 .into_pipeline_layout_create_info(device.clone())
//                 .unwrap(),
//         )
//         .unwrap();

//         // We have to indicate which subpass of which render pass this pipeline is going to be used
//         // in. The pipeline will only be usable from this particular subpass.
//         let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

//         // Finally, create the pipeline.
//         GraphicsPipeline::new(
//             device.clone(),
//             None,
//             GraphicsPipelineCreateInfo {
//                 stages: stages.into_iter().collect(),
//                 // How vertex data is read from the vertex buffers into the vertex shader.
//                 vertex_input_state: Some(vertex_input_state),
//                 // How vertices are arranged into primitive shapes.
//                 // The default primitive shape is a triangle.
//                 input_assembly_state: Some(InputAssemblyState::default()),
//                 // How primitives are transformed and clipped to fit the framebuffer.
//                 // We use a resizable viewport, set to draw over the entire window.
//                 viewport_state: Some(ViewportState::default()),
//                 // How polygons are culled and converted into a raster of pixels.
//                 // The default value does not perform any culling.
//                 rasterization_state: Some(RasterizationState::default()),
//                 // How multiple fragment shader samples are converted to a single pixel value.
//                 // The default value does not perform any multisampling.
//                 multisample_state: Some(MultisampleState::default()),
//                 // How pixel values are combined with the values already present in the framebuffer.
//                 // The default value overwrites the old value with the new one, without any blending.
//                 color_blend_state: Some(ColorBlendState::with_attachment_states(
//                     subpass.num_color_attachments(),
//                     ColorBlendAttachmentState::default(),
//                 )),
//                 // Dynamic states allows us to specify parts of the pipeline settings when
//                 // recording the command buffer, before we perform drawing.
//                 // Here, we specify that the viewport should be dynamic.
//                 dynamic_state: [DynamicState::Viewport].into_iter().collect(),
//                 subpass: Some(subpass.into()),
//                 ..GraphicsPipelineCreateInfo::layout(layout)
//             },
//         )
//         .unwrap()
//     };

//     // Dynamic viewports allow us to recreate just the viewport when the window is resized.
//     // Otherwise we would have to recreate the whole pipeline.
//     let mut viewport = Viewport {
//         offset: [0.0, 0.0],
//         extent: [0.0, 0.0],
//         depth_range: 0.0..=1.0,
//     };

//     // The render pass we created above only describes the layout of our framebuffers. Before we
//     // can draw we also need to create the actual framebuffers.
//     //
//     // Since we need to draw to multiple images, we are going to create a different framebuffer for
//     // each image.
//     let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

//     // Before we can start creating and recording command buffers, we need a way of allocating
//     // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command pools
//     // underneath and provides a safe interface for them.
//     let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
//         device.clone(),
//         Default::default(),
//     ));

//     // Initialization is finally finished!

//     // In some situations, the swapchain will become invalid by itself. This includes for example
//     // when the window is resized (as the images of the swapchain will no longer match the
//     // window's) or, on Android, when the application went to the background and goes back to the
//     // foreground.
//     //
//     // In this situation, acquiring a swapchain image or presenting it will return an error.
//     // Rendering to an image of that swapchain will not produce any error, but may or may not work.
//     // To continue rendering, we need to recreate the swapchain by creating a new swapchain. Here,
//     // we remember that we need to do this for the next loop iteration.
//     let mut recreate_swapchain = false;

//     // In the loop below we are going to submit commands to the GPU. Submitting a command produces
//     // an object that implements the `GpuFuture` trait, which holds the resources for as long as
//     // they are in use by the GPU.
//     //
//     // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
//     // that, we store the submission of the previous frame here.
//     let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

//     event_loop.run(move |event, elwt| {
//         elwt.set_control_flow(ControlFlow::Poll);

//         match event {
//             Event::WindowEvent {
//                 event: WindowEvent::CloseRequested,
//                 ..
//             } => {
//                 elwt.exit();
//             }
//             Event::WindowEvent {
//                 event: WindowEvent::Resized(_),
//                 ..
//             } => {
//                 recreate_swapchain = true;
//             }
//             Event::WindowEvent {
//                 event: WindowEvent::RedrawRequested,
//                 ..
//             } => {
//                 // Do not draw the frame when the screen size is zero. On Windows, this can
//                 // occur when minimizing the application.
//                 let image_extent: [u32; 2] = window.inner_size().into();

//                 if image_extent.contains(&0) {
//                     return;
//                 }

//                 // It is important to call this function from time to time, otherwise resources
//                 // will keep accumulating and you will eventually reach an out of memory error.
//                 // Calling this function polls various fences in order to determine what the GPU
//                 // has already processed, and frees the resources that are no longer needed.
//                 previous_frame_end.as_mut().unwrap().cleanup_finished();

//                 // Whenever the window resizes we need to recreate everything dependent on the
//                 // window size. In this example that includes the swapchain, the framebuffers and
//                 // the dynamic state viewport.
//                 if recreate_swapchain {
//                     // Use the new dimensions of the window.

//                     let (new_swapchain, new_images) = swapchain
//                         .recreate(SwapchainCreateInfo {
//                             image_extent,
//                             ..swapchain.create_info()
//                         })
//                         .expect("failed to recreate swapchain");

//                     swapchain = new_swapchain;

//                     // Because framebuffers contains a reference to the old swapchain, we need to
//                     // recreate framebuffers as well.
//                     framebuffers = window_size_dependent_setup(
//                         &new_images,
//                         render_pass.clone(),
//                         &mut viewport,
//                     );

//                     recreate_swapchain = false;
//                 }

//                 // Before we can draw on the output, we have to *acquire* an image from the
//                 // swapchain. If no image is available (which happens if you submit draw commands
//                 // too quickly), then the function will block. This operation returns the index of
//                 // the image that we are allowed to draw upon.
//                 //
//                 // This function can block if no image is available. The parameter is an optional
//                 // timeout after which the function call will return an error.
//                 let (image_index, suboptimal, acquire_future) =
//                     match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
//                         Ok(r) => r,
//                         Err(VulkanError::OutOfDate) => {
//                             recreate_swapchain = true;
//                             return;
//                         }
//                         Err(e) => panic!("failed to acquire next image: {e}"),
//                     };

//                 // `acquire_next_image` can be successful, but suboptimal. This means that the
//                 // swapchain image will still work, but it may not display correctly. With some
//                 // drivers this can be when the window resizes, but it may not cause the swapchain
//                 // to become out of date.
//                 if suboptimal {
//                     recreate_swapchain = true;
//                 }

//                 // In order to draw, we have to record a *command buffer*. The command buffer object
//                 // holds the list of commands that are going to be executed.
//                 //
//                 // Recording a command buffer is an expensive operation (usually a few hundred
//                 // microseconds), but it is known to be a hot path in the driver and is expected to
//                 // be optimized.
//                 //
//                 // Note that we have to pass a queue family when we create the command buffer. The
//                 // command buffer will only be executable on that given queue family.
//                 let mut builder = AutoCommandBufferBuilder::primary(
//                     &command_buffer_allocator,
//                     queue.queue_family_index(),
//                     CommandBufferUsage::OneTimeSubmit,
//                 )
//                 .unwrap();

//                 builder
//                     // Before we can draw, we have to *enter a render pass*.
//                     .begin_render_pass(
//                         RenderPassBeginInfo {
//                             // A list of values to clear the attachments with. This list contains
//                             // one item for each attachment in the render pass. In this case, there
//                             // is only one attachment, and we clear it with a blue color.
//                             //
//                             // Only attachments that have `AttachmentLoadOp::Clear` are provided
//                             // with clear values, any others should use `None` as the clear value.
//                             clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

//                             ..RenderPassBeginInfo::framebuffer(
//                                 framebuffers[image_index as usize].clone(),
//                             )
//                         },
//                         SubpassBeginInfo {
//                             // The contents of the first (and only) subpass.
//                             // This can be either `Inline` or `SecondaryCommandBuffers`.
//                             // The latter is a bit more advanced and is not covered here.
//                             contents: SubpassContents::Inline,
//                             ..Default::default()
//                         },
//                     )
//                     .unwrap()
//                     // We are now inside the first subpass of the render pass.
//                     //
//                     // TODO: Document state setting and how it affects subsequent draw commands.
//                     .set_viewport(0, [viewport.clone()].into_iter().collect())
//                     .unwrap()
//                     .bind_pipeline_graphics(pipeline.clone())
//                     .unwrap()
//                     .bind_vertex_buffers(0, vertex_buffer.clone())
//                     .unwrap();

//                 unsafe {
//                     builder
//                         // We add a draw command.
//                         .draw(vertex_buffer.len() as u32, 1, 0, 0)
//                         .unwrap();
//                 }

//                 builder
//                     // We leave the render pass. Note that if we had multiple subpasses we could
//                     // have called `next_subpass` to jump to the next subpass.
//                     .end_render_pass(Default::default())
//                     .unwrap();

//                 // Finish recording the command buffer by calling `end`.
//                 // let command_buffer = builder.end().unwrap();
//                 let command_buffer = builder.build().unwrap();

//                 let future = previous_frame_end
//                     .take()
//                     .unwrap()
//                     .join(acquire_future)
//                     .then_execute(queue.clone(), command_buffer)
//                     .unwrap()
//                     // The color output is now expected to contain our triangle. But in order to
//                     // show it on the screen, we have to *present* the image by calling
//                     // `then_swapchain_present`.
//                     //
//                     // This function does not actually present the image immediately. Instead it
//                     // submits a present command at the end of the queue. This means that it will
//                     // only be presented once the GPU has finished executing the command buffer
//                     // that draws the triangle.
//                     .then_swapchain_present(
//                         queue.clone(),
//                         SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
//                     )
//                     .then_signal_fence_and_flush();

//                 match future.map_err(Validated::unwrap) {
//                     Ok(future) => {
//                         previous_frame_end = Some(future.boxed());
//                     }
//                     Err(VulkanError::OutOfDate) => {
//                         recreate_swapchain = true;
//                         previous_frame_end = Some(sync::now(device.clone()).boxed());
//                     }
//                     Err(e) => {
//                         panic!("failed to flush future: {e}");
//                         // previous_frame_end = Some(sync::now(device.clone()).boxed());
//                     }
//                 }
//             }
//             Event::AboutToWait => window.request_redraw(),
//             _ => (),
//         }
//     })
// }

// /// This function is called once during initialization, then again whenever the window is resized.
// fn window_size_dependent_setup(
//     images: &[Arc<Image>],
//     render_pass: Arc<RenderPass>,
//     viewport: &mut Viewport,
// ) -> Vec<Arc<Framebuffer>> {
//     let extent = images[0].extent();
//     viewport.extent = [extent[0] as f32, extent[1] as f32];

//     images
//         .iter()
//         .map(|image| {
//             let view = ImageView::new_default(image.clone()).unwrap();
//             Framebuffer::new(
//                 render_pass.clone(),
//                 FramebufferCreateInfo {
//                     attachments: vec![view],
//                     ..Default::default()
//                 },
//             )
//             .unwrap()
//         })
//         .collect::<Vec<_>>()
// }

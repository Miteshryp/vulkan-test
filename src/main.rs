mod shaders;
mod vertex;

use std::{process::exit, sync::Arc, time::{UNIX_EPOCH, SystemTime}, alloc::System};
use shaders::fs;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        self,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        CommandBufferUsage, PrimaryAutoCommandBuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferLevel, CopyImageToBufferInfo, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{self, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Properties, Queue, QueueCreateInfo,
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
    swapchain::{self, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary, shader::spirv::LoopControl,
};
use winit::{
    event::{ElementState, Event, Modifiers, MouseButton, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::{Key, ModifiersState},
    platform::modifier_supplement::KeyEventExtModifierSupplement,
    window::{Window, WindowBuilder},
};

// type name for buffer allocator
type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
type CommandBufferType = 
    Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

struct RenderTargetInfo {
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    fbos: Vec<Arc<Framebuffer>>
}

struct InstanceAllocators {
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: GenericBufferAllocator
}
struct VulkanInstance {
    logical: Arc<Device>,
    physical: Arc<vulkano::device::physical::PhysicalDevice>,
    surface: Arc<Surface>,
    device_queues: Vec<Arc<Queue>>,
    swapchain_info: VulkanSwapchainInfo,
    render_target: RenderTargetInfo,

    allocators: InstanceAllocators
}

struct VulkanSwapchainInfo {
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    create_info: SwapchainCreateInfo,
}

impl VulkanInstance {
    fn initialise(&mut self, window: Arc<Window>, eltw: &EventLoop<()>) {
        todo!()
    }

    fn get_device_type(&self) -> PhysicalDeviceType {
        self.physical.properties().device_type
    }

    fn get_first_queue(&self) -> Arc<Queue> {
        self.device_queues.iter().next().unwrap().clone()
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
fn create_render_pass(device: Arc<Device>, swapchain_info: &VulkanSwapchainInfo) -> Arc<RenderPass> {
    // need 3 things: device Arc, attachments, and a pass
    let format = swapchain_info.create_info.image_format.clone();

    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: format, // vulkano::format::Format::B8G8R8A8_SRGB, // vulkano::format::Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
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

    let vertex_shader_input_state = vertex::Vec2::per_vertex()
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
    memory_type_filter: MemoryTypeFilter
) -> Subbuffer<T>
    where
        T: BufferContents
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
        data
    ).unwrap()
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
            .with_transparent(true)
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

    // let queue = instance.device_queues.iter().next().unwrap();

    let vertices = Vec::from([
        vertex::Vec2 { x: 0.0, y: 1.0 },
        vertex::Vec2 { x: -1.0, y: -1.0 },
        vertex::Vec2 { x: 1.0, y: -1.0 },
    ]);

    // // buffer allocator for memory buffer objects
    // let memory_allocator: GenericBufferAllocator =
    //     create_buffer_allocator(instance.get_logical_device());

    // // let image_buffer = get_swapchain_image(device.clone());
    // // let image_buffer = get_image_buffer(memory_allocator.clone());
    // let vertex_buffer = create_buffer(
    //     memory_allocator.clone(),
    //     vertices.into_iter(),
    //     BufferUsage::VERTEX_BUFFER,
    //     MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    // ); // data is streamed from CPU to GPU

    // // TODO: Analyze the following step carefully and understand it more deeply
    // let render_pass = create_render_pass(instance.get_logical_device()); // defines the schema information required for configuring the output of shaders to the framebuffer

    // // Create a framebuffer for each swapchain image
    // let fbos = instance
    //     .swapchain_info
    //     .images
    //     .iter()
    //     .map(|img| get_framebuffer_object(render_pass.clone(), img.clone()))
    //     .collect();

    // let graphics_pipeline =
    //     create_graphics_pipeline(window.clone(), instance.get_logical_device(), render_pass);

    // let command_buffers = get_command_buffers(
    //     instance.get_logical_device(),
    //     queue.clone(),
    //     fbos,
    //     graphics_pipeline,
    //     vertex_buffer,
    // );

    let mut command_buffers = create_command_buffers(&instance, vertices.clone());
    println!("First call");
    
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
                // recreating swapchains

                // TODO
                instance.swapchain_info = create_swapchain(
                    window.clone(),
                    instance.get_physical_device(),
                    instance.get_logical_device(),
                    instance.surface.clone(),
                );

                // refreshing the render targets
                instance.render_target = refresh_render_target(window.clone(), instance.get_logical_device(), &instance.swapchain_info);

                // recreating the render pass, fbos, pipeline and command buffers with the new swapchain images
                command_buffers =
                    create_command_buffers(&instance, vertices.clone());
            }

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
        create_info: create_info,
    }
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
            max_api_version: Some(Version {
                major: 1,
                minor: 1,
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
        ..Default::default()
    };

    // 3 components to check for
    // a. Swapchain support in window
    // b. Surface support in device
    // c. Queue family support in device queue

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
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|queue_index| (p, queue_index))
        })
        .next()
        // .min_by_key(|(device, _)| match device.properties().device_type {
        //     PhysicalDeviceType::DiscreteGpu => 0,
        //     PhysicalDeviceType::IntegratedGpu => 1,
        //     PhysicalDeviceType::Cpu => 2,
        //     _ => 3,
        // })
        .unwrap_or_else(|| panic!("Failed to find a valid physical device\n"));

    println!(
        "Physical Device: {:?}",
        physical_device.properties().device_name
    );

    let physical_device_properties: &Properties = physical_device.properties();

    /*
    Creating Logical Device
    */
    let (logical_device, queues_iterator) = Device::new(
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

    let device_queues: Vec<Arc<Queue>> = queues_iterator.collect();
    let swapchain = create_swapchain(
        window.clone(),
        physical_device.clone(),
        logical_device.clone(),
        surface.clone(),
    );

    // TODO: Explore graphics pipeline or command builder extensibilities

    let render_target_info = refresh_render_target(window.clone(), logical_device.clone(), &swapchain);
    let allocators = InstanceAllocators {
        command_buffer_allocator: create_command_buffer_allocator(logical_device.clone()),
        memory_allocator: create_buffer_allocator(logical_device.clone())
    };

    VulkanInstance {
        logical: logical_device,
        physical: physical_device,
        device_queues: device_queues,
        surface: surface,
        swapchain_info: swapchain,
        render_target: render_target_info,
        allocators: allocators
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

static mut start: Option<SystemTime> = None;


fn refresh_render_target(
    window: Arc<Window>,
    device: Arc<Device>,
    swapchain_info: &VulkanSwapchainInfo,

) -> RenderTargetInfo {
        // TODO: Analyze the following step carefully and understand it more deeply
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
            fbos: fbos
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
    vertices: Vec<vertex::Vec2>,
) -> Vec<CommandBufferType> {
    // buffer allocator for memory buffer objects
    let memory_allocator: GenericBufferAllocator =
        create_buffer_allocator(instance.get_logical_device());

    // let image_buffer = get_swapchain_image(device.clone());
    // let image_buffer = get_image_buffer(memory_allocator.clone());
    let vertex_buffer = create_buffer_from_iter(
        memory_allocator.clone(),
        vertices.into_iter(),
        BufferUsage::VERTEX_BUFFER,
        MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    ); // data is streamed from CPU to GPU

    let mut f = 0.0;
    unsafe {
        f = start.unwrap().elapsed().unwrap().as_secs_f32() as f32;//std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as f32;
    }
    println!("{f}");
    let uniform_buffer = create_buffer_from_data(
        memory_allocator.clone(),
        shaders::vs::Data { view: f },
        BufferUsage::UNIFORM_BUFFER,
        MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    );


    let command_buffers = build_fbo_command_buffers_for_pipeline(
        // instance.get_logical_device(),
        // instance.get_first_queue(),
        // &instance.render_target.fbos,
        // instance.get_graphics_pipeline(),
        instance,
        vertex_buffer,
        uniform_buffer
    );

    return command_buffers;
}

// Only builds the command buffers
fn build_fbo_command_buffers_for_pipeline(
    instance: &VulkanInstance,
    vertex_buffer: Subbuffer<[vertex::Vec2]>,
    uniform_buffer: Subbuffer<shaders::vs::Data>
) -> Vec<CommandBufferType> {
    // I need:
    // Command buffer built using AutoCommandBufferBuilder - done
    // render pass created for data
    // FBO - done
    // graphics pipeline object
    // let cb_allocator = create_command_buffer_allocator(instance.get_logical_device().clone());
    let cb_allocator = &instance.allocators.command_buffer_allocator;
    
    // let queue_family_index = queue.queue_family_index();
    let queue_family_index = instance.get_first_queue().queue_family_index();
    let pipeline = instance.get_graphics_pipeline();

    let descriptor_set_index: usize = 0;
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(instance.get_logical_device(), Default::default());
    let descriptor_set_layout = 
        pipeline
            .layout()
            .set_layouts()
            .get(descriptor_set_index)
            .unwrap();

    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer)], // 0 is the binding
        [],
    ).unwrap();

    let fbos = &instance.render_target.fbos;
    
    fbos
        .into_iter()
        .map(|fb| {
            let mut command_builder = AutoCommandBufferBuilder::primary(
                cb_allocator,
                queue_family_index,
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

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
                .bind_pipeline_graphics(instance.get_graphics_pipeline())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics, 
                    instance.get_graphics_pipeline().layout().clone(), 
                    0, 
                    descriptor_set.clone(),
                )
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .draw(3, 1, 0, 0)
                .unwrap()
                .end_render_pass(Default::default())
                .unwrap();

            // Removing this since we no longer need to copy the image data into a buffer since we are not saving the image anymore
            // .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            //     framebuffer_object_image,
            //     target_buffer.clone(),
            // ))
            // .unwrap_or_else(|err| panic!("Failed to copy image to buffer: {:?}", err));

            let cb = command_builder.build().unwrap();
            return cb;
        })
        .collect()

    // let future = sync::now(device.clone())
    //     .then_execute(queue, command_buffer)
    //     .unwrap()
    //     .then_signal_fence_and_flush()
    //     .unwrap();

    // future.wait(None).unwrap();
}

// fn get_swapchain_image(device: Arc<Device>) -> _ {
//     todo!()
// }

// fn get_auto_command_buffer_builder(
//     cb_allocator: &StandardCommandBufferAllocator,
//     queue_family_index: u32,
// ) -> AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer> {
//     let mut cb_builder: AutoCommandBufferBuilder<
//         vulkano::command_buffer::PrimaryAutoCommandBuffer,
//     > = AutoCommandBufferBuilder::primary(
//         cb_allocator,
//         queue_family_index,
//         CommandBufferUsage::OneTimeSubmit,
//     )
//     .unwrap();

//     return cb_builder;
// }

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

fn main() {
    unsafe {
        start =  Some(SystemTime::now());
    }

    let (window, elwt) = create_window();
    let mut vulkan_instance: VulkanInstance = initialise_vulkan_runtime(window.clone(), &elwt);

    // let device = vulkan_instance.logical;
    // let queues = vulkan_instance.device_queues;

    // let use_queue = queues.iter().next().unwrap();

    // let vertices = Vec::from([
    //     vertex::Vec2 { x: 0.0, y: 1.0 },
    //     vertex::Vec2 { x: -1.0, y: -1.0 },
    //     vertex::Vec2 { x: 1.0, y: -1.0 },
    // ]);

    // // buffer allocator for memory buffer objects
    // let memory_allocator: GenericBufferAllocator = create_buffer_allocator(device.clone());

    // // let image_buffer = get_swapchain_image(device.clone());
    // // let image_buffer = get_image_buffer(memory_allocator.clone());
    // let vertex_buffer = create_buffer(
    //     memory_allocator.clone(),
    //     vertices.into_iter(),
    //     BufferUsage::VERTEX_BUFFER,
    //     MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    // ); // data is streamed from CPU to GPU

    // // TODO: Analyze the following step carefully and understand it more deeply
    // let render_pass = create_render_pass(device.clone()); // defines the schema information required for configuring the output of shaders to the framebuffer

    // // Create a framebuffer for each swapchain image
    // let fbos = vulkan_instance
    //     .swapchain_info
    //     .images
    //     .iter()
    //     .map(|&img| get_framebuffer_object(render_pass.clone(), img))
    //     .collect();

    // let graphics_pipeline = create_graphics_pipeline(window.clone(), device.clone(), render_pass);

    // let command_buffers = get_command_buffers(
    //     device.clone(),
    //     use_queue.clone(),
    //     fbos,
    //     graphics_pipeline,
    //     vertex_buffer,
    // );

    start_window_event_loop(window.clone(), elwt, vulkan_instance);
}

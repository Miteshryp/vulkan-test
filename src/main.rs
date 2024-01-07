mod graphics_pack;
// mod vertex;

use nalgebra_glm as glm;

use graphics_pack::{
    buffers::{
        self,
        base_buffer::{self, BufferOptions, VecBufferOps},
        primitives::{Vec3, VertexPoint},
        IndexBuffer, UniformBuffer, UniformSet, VertexBuffer,
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
        allocator::StandardDescriptorSetAllocator, persistent, PersistentDescriptorSet,
        WriteDescriptorSet, layout::DescriptorSetLayoutCreateFlags,
    },
    device::{
        physical::{self, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Properties, Queue, QueueCreateInfo,
        QueueFamilyProperties, QueueFlags,
    },
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    instance::InstanceCreateInfo,
    memory::{
        allocator::{
            AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
        },
        MemoryType,
    },
    pipeline::{
        graphics::{
            self,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{self, DepthState, DepthStencilState, StencilState},
            input_assembly::InputAssemblyState,
            rasterization::{FrontFace, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo},
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::{self, spirv::LoopControl, EntryPoint},
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

// use crate::graphics_pack::buffers::{BufferOptions};

// type name for buffer allocator
type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
type CommandBufferType = Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

struct RenderTargetInfo {
    pipeline: Arc<GraphicsPipeline>,
    // render_pass: Arc<RenderPass>,
    fbos: Vec<Arc<Framebuffer>>,
}

struct InstanceAllocators {
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: GenericBufferAllocator,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
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
    images: Vec<Arc<Image>>,
}

impl VulkanInstance {
    fn initialise(window: Arc<Window>, eltw: &EventLoop<()>) -> Self {
        initialise_vulkan_runtime(window, eltw)
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

            depth: {
                format: vulkano::format::Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare
            }
        },
        pass: {
            color: [color],
            // depth_stencil: {}
            depth_stencil: {depth},
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

    let vertex_shader: EntryPoint = shaders::get_vertex_shader(logical_device.clone())
        .entry_point("main")
        .unwrap();

    // println!("Vertex Shader entry point info: {:?}", vertex_shader.info());

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
    let mut descriptor_set_layout = PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stages);
    let set_layout = &mut descriptor_set_layout.set_layouts[0];
    set_layout.flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;
    // set_layout.bindings.get(&0).unwrap().

    let mut descriptor_create_info =
        descriptor_set_layout
            .into_pipeline_layout_create_info(logical_device.clone())
            .unwrap();
        

    // ERROR: from_stages is setting descriptor set count to 1
    let pipeline_layout =
        PipelineLayout::new(logical_device.clone(), descriptor_create_info).unwrap();

    let depth_stencil = depth_stencil::DepthState::simple();
    let rasterization_state_info: RasterizationState = RasterizationState {
        cull_mode: graphics::rasterization::CullMode::Back,
        front_face: FrontFace::CounterClockwise,
        ..Default::default()
    };

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

            depth_stencil_state: Some(DepthStencilState {
                depth: Some(depth_stencil),
                ..Default::default()
            }),

            // rasterization_state: Some(Default::default()),
            rasterization_state: Some(rasterization_state_info),
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

#[allow(unused_variables)]
#[allow(unused_assignments)]
fn winit_handle_window_events(
    event: WindowEvent,
    window_target: &EventLoopWindowTarget<()>,
    // view_matrix: &mut glm::TMat4<f32>,
    camera: &mut glm::TVec3<f32>
) {
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
                    unsafe {camera.z -= MOVE_SPEED; }
                }
                Key::Character("a") => {
                    println!("Left");
                    unsafe {camera.x -= MOVE_SPEED; }
                }
                Key::Character("s") => {
                    println!("Backward");
                    unsafe {camera.z += MOVE_SPEED; }
                }
                Key::Character("d") => {
                    println!("Right");
                    unsafe {camera.x += MOVE_SPEED; }
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
                y: -0.9,
                z: -0.5,
            },
            color: color1.clone(),
        },
        VertexPoint {
            position: Vec3 {
                x: -1.0,
                y: 1.0,
                z: -0.5,
            },
            color: color2.clone(),
        },
        VertexPoint {
            position: Vec3 {
                x: 1.0,
                y: 1.0,
                z: -0.5,
            },
            color: color2.clone(),
        },
        VertexPoint {
            position: Vec3 {
                x: 1.0,
                y: -1.0,
                z: -0.5,
            },
            color: color3.clone(),
        },
        ]);
        
        let indicies: Vec<u32> = Vec::from([0, 1, 2, 0, 2, 3]);
        
        let mut camera_position = glm::vec3(0.0,0.0,0.0);
    // Data for uniform 1

    el.set_control_flow(ControlFlow::Poll);
    let _ = el.run(|app_event, elwt| match app_event {
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


            let mut model = glm::identity::<f32, 4>();
            model = glm::translate(&model, &glm::vec3(0.1, 0.0, 0.0));
            // model = glm::scale(&model, &glm::vec3(10.0,10.0,10.0));

            let mut view = glm::look_at(
                &camera_position,
                &glm::vec3(camera_position.x, camera_position.y, camera_position.z + -0.01),
                &glm::vec3(0.0, 1.0, 0.0),
            );

            let projection = glm::perspective(
                (window.inner_size().width / window.inner_size().height) as f32,
                std::f32::consts::PI / 4.0,
                0.01,
                -1000.0,
            );
            // draw call

            if window_resized || recreate_swapchain {
                // recreating swapchains
                refresh_instance_swapchain(window.clone(), &mut instance);

                // refreshing the render targets
                // recreating the render pass, fbos, pipeline and command buffers with the new swapchain images
                instance.render_target = refresh_render_target(
                    window.clone(),
                    instance.get_logical_device(),
                    &instance.swapchain_info,
                    instance.allocators.memory_allocator.clone(),
                );

                window_resized = false;
                recreate_swapchain = false;
            }

            // let mut uniform_set = UniformSet::new(0);
            
            // uniform 0
            // {

                let mut data = shaders::vs::Data {
                    view: unsafe { START.unwrap().elapsed().unwrap().as_secs_f32() },
                };
                
                // uniform_set.add_uniform_buffer(
                    //     instance.allocators.memory_allocator.clone(),
                    //     instance.get_graphics_pipeline(),
                    //     data,
                    //     Default::default(),
                    // );     
                    let mut uni0 = UniformBuffer::create(
                        instance.allocators.memory_allocator.clone(), 
                        0, 
                        data, 
                        Default::default()
                    );
            // }


            // uniform 1
            // {
                let mut data = shaders::vs::MvpMatrix {
                    model: Into::<[[f32; 4]; 4]>::into(model),
                    view: Into::<[[f32; 4]; 4]>::into(view),
                    projection: Into::<[[f32; 4]; 4]>::into(projection),
                };

                // uniform_set.add_uniform_buffer(
                //     instance.allocators.memory_allocator.clone(),
                //     instance.get_graphics_pipeline(),
                //     data,
                //     Default::default(),
                // );
                let mut uni1 = UniformBuffer::create(
                    instance.allocators.memory_allocator.clone(), 
                    1, 
                    data, Default::default()
                );
            // }

            let command_buffers = create_command_buffers(
                &instance,
                vertices.clone(),
                indicies.clone(),
                // vec![uniform_set],
                vec![uni0, uni1]
            );

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
                winit_handle_window_events(event, elwt, &mut camera_position);
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

fn refresh_instance_swapchain(window: Arc<Window>, instance: &mut VulkanInstance) {
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
        command_buffer_allocator: create_command_buffer_allocator(logical_device.clone()),
        memory_allocator: create_buffer_allocator(logical_device.clone()),
        descriptor_set_allocator: create_descriptor_set_allocator(logical_device.clone()),
    };

    let render_target_info = refresh_render_target(
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

    // return (logical_device, device_queues, surface_swapchain);

    /*
        Steps to initialise vulkan instance

        // Vulkan initialisation (PHASE 1)
        Step 1. Initialise the window for the application (using winit)
        Step 2. Initialise the vulkan instance
        Step 3. Create Vulkan Surface to render our graphics to which would be a reference to the current window
            and acquired using the vulkan instance created
        Step 4. Query for the available physical devices based on the requirements (support for swapchain extensions)
        Step 5. Select a physical device to be used based on queue classes supported and support for swapchain
        Step 6. Create a Logical device based on the (VkQueue) queue types that we want to use and get the first supported queue and its family index

        // Render Target configs (PHASE 2) - Static Data, only changed on window events
        Step 7. Create a SwapChain to render our images to and bind it to the acquired surface.
            This swapchain will then swap the images from the surface to render the images to the screen
        Step 8. Create Render Pass, fbos, images and bind it all by creating the Graphics pipeline. Store the graphics pipeline and fbos
            to be used later to construct command buffers

        // Dynamic targets (PHASE 3) - Dynamic data, can be different for each frame
        Step 9. Create the vertex and fragment shaders
        Step 10. Create the vertex and index buffer to render primitives
        Step 11. Create command buffers and bind the shaders, pipelines, fbos, VBs, IBs and uniforms to them
        Step 12. Acquire the future by passing the command buffer into sync::now(),
            and execute the future to render the buffer to the binded swapchain image

        TODO: In-Flight rendering
    */

    // Err("Failed to initialise vulkan runtime")
    // Ok(())
}

fn create_descriptor_set_allocator(
    logical_device: Arc<Device>,
) -> Arc<StandardDescriptorSetAllocator> {
    Arc::new(StandardDescriptorSetAllocator::new(
        logical_device,
        Default::default(),
    ))
}

static mut START: Option<SystemTime> = None;
static mut MOVE_SPEED: f32 = 0.1;

fn refresh_render_target(
    window: Arc<Window>,
    device: Arc<Device>,
    swapchain_info: &VulkanSwapchainInfo,
    allocator: GenericBufferAllocator,
) -> RenderTargetInfo {
    let render_pass = create_render_pass(device.clone(), swapchain_info); // defines the schema information required for configuring the output of shaders to the framebuffer

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

    let graphics_pipeline =
        create_graphics_pipeline(window.clone(), device.clone(), render_pass.clone());

    RenderTargetInfo {
        pipeline: graphics_pipeline,
        // render_pass: render_pass,
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
    instance: &VulkanInstance,
    vertex_data: Vec<VertexPoint>,
    index_data: Vec<u32>,
    // uniform_sets: Vec<UniformSet>,
    uniforms: Vec<UniformBuffer>,
    // uniform_buffer_data: graphics_pack::shaders::vs::Data,
) -> Vec<CommandBufferType> {
    // let uniforms_descriptor_set = UniformBuffer::get_descriptor_set(
    //     instance.get_logical_device(),
    //     instance.get_graphics_pipeline(),
    //     instance.allocators.descriptor_set_allocator.clone(),
    //     0,
    //     uniforms
    // );

    // buffer allocator for memory buffer objects
    let memory_allocator = instance.allocators.memory_allocator.clone();

    let vertex_buffer = VertexBuffer::from_vec(
        instance.allocators.memory_allocator.clone(),
        vertex_data,
        Default::default(),
    )
    .unwrap();

    let index_buffer: IndexBuffer = graphics_pack::buffers::IndexBuffer::from_vec(
        instance.allocators.memory_allocator.clone(),
        index_data,
        Default::default(),
    )
    .unwrap();

    // let depth_stencil_buffer = Buffer::from(value)

    let mut f = 0.0;
    unsafe {
        f = START.unwrap().elapsed().unwrap().as_secs_f32() as f32; //std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as f32;
    }
    println!("{f}");
    // let uniform_buffer = create_buffer_from_data(
    //     memory_allocator.clone(),
    //     shaders::vs::Data { view: f },
    //     BufferUsage::UNIFORM_BUFFER,
    //     MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    // );

    let command_buffers = build_fbo_command_buffers_for_pipeline(
        instance.get_logical_device(),
        instance.allocators.command_buffer_allocator.clone(),
        instance.allocators.descriptor_set_allocator.clone(),
        instance.get_first_queue(),
        instance.get_graphics_pipeline(),
        &instance.render_target.fbos,
        vertex_buffer,
        index_buffer,
        // uniform_buffer,
        uniforms
        // uniform_sets,
    );

    return command_buffers;
}

// Only builds the command buffers
fn build_fbo_command_buffers_for_pipeline(
    // instance: &VulkanInstance,
    logical_device: Arc<Device>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    fbos: &Vec<Arc<Framebuffer>>,

    // vertex_buffer: Subbuffer<[vertex::VertexPoint]>,
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
    // uniform_buffer_descriptor: Arc<PersistentDescriptorSet>, // uniform_buffer: Subbuffer<shaders::vs::Data>,
    uniforms: Vec<UniformBuffer>,
    // uniform_sets: Vec<UniformSet>,
) -> Vec<CommandBufferType> {

    let queue_family_index = queue.queue_family_index();

    // let mut persistent_descriptor_sets: Vec<Arc<PersistentDescriptorSet>> = uniform_sets
    //     .into_iter()
    //     .map(|us| {
    //         us.get_persistent_descriptor_set(uniform_allocator.clone(), graphics_pipeline.clone())
    //     })
    //     .collect();

    // println!("Length: {}", persistent_descriptor_sets.len());
    // let sample = [WriteDescriptorSet::buffer(0, Buffer::from_data(
    //     create_buffer_allocator(logical_device.clone()), 
    //     BufferCreateInfo {
    //         usage: BufferUsage::UNIFORM_BUFFER,
    //         ..Default::default()
    //     }, AllocationCreateInfo {
    //         memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    //         ..Default::default()
    //     }, 

    //     // Into::<[[f32; 4]; 4]>::into(model)
    //     shaders::vs::MvpMatrix {
    //         model: Into::<[[f32; 4]; 4]>::into(glm::identity::<f32, 4>()),
    //         view: Into::<[[f32; 4]; 4]>::into(glm::identity::<f32, 4>()),
    //         projection: Into::<[[f32; 4]; 4]>::into(glm::identity::<f32, 4>())
    //     }
    // ).unwrap())];

    


    fbos.into_iter()
        .map(|fb| {
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

            // NOTE: This structure is only used to bind dynamic data to the graphics pipeline
            // Any modifications to the rendering stages have to be done in the
            // graphics pipeline while it is created.
            command_builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())],
                        ..RenderPassBeginInfo::framebuffer(fb.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(graphics_pipeline.clone())
                .unwrap();


            // Binding buffers
            command_builder
                .bind_index_buffer(index_subbuffer)
                .unwrap()
                .bind_vertex_buffers(0, vertex_subbuffer)
                .unwrap();

            // command_builder.push_descriptor_set(
            //     PipelineBindPoint::Graphics, 
            //     graphics_pipeline.layoyt(), 
            //     // set_num,
            //     // descriptor_writes
            // );


            // Binding uniforms (descriptor sets)
            // command_builder
            //     .bind_descriptor_sets(
            //         PipelineBindPoint::Graphics,
            //         graphics_pipeline.layout().clone(),
            //         0,
            //         persistent_descriptor_sets.clone(),
            //     )
            //     .unwrap();
            
            // let push_descriptor_writes:  = uniforms.into_iter().collect();
            let samples = uniforms.clone().into_iter().map(|ub| {ub.get_write_descriptor()}).collect();

            command_builder.push_descriptor_set(
                PipelineBindPoint::Graphics, 
                graphics_pipeline.layout().clone(), 
                0,
                samples
                // sample
            ).unwrap();

            // Draw call
            command_builder
                // NOTE: This is the draw call used for indexed rendering
                .draw_indexed(index_count, index_count / 3, 0, 0, 0)
                .unwrap()
                .end_render_pass(Default::default())
                .unwrap();

            let cb = command_builder.build().unwrap();
            return cb;
        })
        .collect()
}

// fn get_image_buffer(allocator: GenericBufferAllocator) -> Arc<vulkano::image::Image> {
//     vulkano::image::Image::new(
//         allocator.clone(),
//         vulkano::image::ImageCreateInfo {
//             image_type: vulkano::image::ImageType::Dim2d,
//             format: vulkano::format::Format::R8G8B8A8_UNORM,
//             extent: [1024, 1024, 1],
//             usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT
//                 | vulkano::image::ImageUsage::TRANSFER_SRC,
//             ..Default::default()
//         },
//         AllocationCreateInfo {
//             memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
//             ..Default::default()
//         },
//     )
//     .unwrap()
// }

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

    println!("Descriptor count: {}", vulkan_instance.get_graphics_pipeline().layout().set_layouts().len());
    start_window_event_loop(window.clone(), elwt, vulkan_instance);
}

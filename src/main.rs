mod shaders;
mod vertex;

use image::{buffer, ImageBuffer, Rgba};
use std::{process::exit, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        CommandBufferUsage,
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferLevel, CopyImageToBufferInfo, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassContents,
    },
    device::{
        physical::{self, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Properties, Queue, QueueCreateInfo,
        QueueFamilyProperties, QueueFlags,
    },
    image::{view::ImageView, Image, ImageCreateInfo},
    instance::InstanceCreateInfo,
    memory::allocator::{
        AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        cache::PipelineCache,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo},
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::Surface,
    sync::{self, GpuFuture},
    Version, VulkanLibrary,
};
use winit::{
    event::{ElementState, Event, Modifiers, MouseButton, WindowEvent},
    event_loop::{self, EventLoop, EventLoopWindowTarget},
    keyboard::{Key, ModifiersState},
    platform::modifier_supplement::KeyEventExtModifierSupplement,
    window::{Window, WindowBuilder},
};

type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;

struct VulkanInstance {
    logical: Arc<Device>,
    physical_properties: Properties,
    device_queues: Arc<Vec<Queue>>,
}

pub trait VulkanInstanceOps {
    fn initialise(&mut self, window: Arc<Window>, eltw: &EventLoop<()>);
    fn get_device_type(&mut self) -> PhysicalDeviceType;
    fn get_logical_device(&mut self) -> Arc<Device>;
}

impl VulkanInstanceOps for VulkanInstance {
    fn initialise(&mut self, window: Arc<Window>, eltw: &EventLoop<()>) {
        todo!()
    }

    fn get_device_type(&mut self) -> PhysicalDeviceType {
        self.physical_properties.device_type
    }

    fn get_logical_device(&mut self) -> Arc<Device> {
        self.logical.clone()
    }
}

// creates a general buffer allocator
fn create_buffer_allocator(device: Arc<Device>) -> GenericBufferAllocator {
    // We create memory allocator as an Arc because the Buffer::from_iter takes the allocator as an Arc copy
    Arc::new(StandardMemoryAllocator::new_default(device))
}

// Creates command buffer allocators required to be submitted to a render pass
fn create_command_buffer_allocator(device: Arc<Device>) -> StandardCommandBufferAllocator {
    let allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            ..Default::default()
        },
    );

    return allocator;
}

fn create_render_pass(device: Arc<Device>) -> Arc<RenderPass> {
    // need 3 things: device Arc, attachments, and a pass
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: vulkano::format::Format::R8G8B8A8_UNORM,
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
                    extent: [1024.0, 1024.0],
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

fn create_buffer<T, I>(
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

// fn create_buffer<T>(
//     allocator: GenericBufferAllocator,
//     data: &[T],
//     buffer_usage: BufferUsage,
//     memory_type_filter: MemoryTypeFilter
// ) -> Subbuffer<T>
//     where
//         T: BufferContents
// {
//     // Buffer::from_data(
//     //     allocator,
//     //             BufferCreateInfo {
//     //         usage: buffer_usage,
//     //         ..Default::default()
//     //     },
//     //     AllocationCreateInfo {
//     //         memory_type_filter: memory_type_filter,
//     //         ..Default::default()
//     //     },
//     //     data
//     // ).unwrap()
// }


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
        } => {

            match key_event.state {
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
            }
        }
        _ => (),
    }
    return;
}

fn create_window() -> (Arc<Window>, EventLoop<()>) {
    let winit_event_loop = event_loop::EventLoop::new().unwrap();
    let winit_window: Arc<Window> =
        Arc::new(WindowBuilder::new().build(&winit_event_loop).unwrap());

    // setting the control flow for the event loop
    winit_event_loop.set_control_flow(event_loop::ControlFlow::Poll);

    (winit_window, winit_event_loop)
}

fn start_window_event_loop(window: Arc<Window>, el: EventLoop<()>) {
    let _ = el.run(|app_event, elwt| match app_event {
        // Window based Events
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

#[allow(unused_variables)]
#[allow(unused_assignments)]
fn initialise_vulkan_runtime(
    window: Arc<Window>,
    el: &EventLoop<()>,
) -> (Arc<Device>, Vec<Arc<Queue>>) {
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
        .min_by_key(|(device, _)| match device.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::Cpu => 2,
            _ => 3,
        })
        .unwrap_or_else(|| panic!("Failed to find a valid physical device\n"));

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
            ..Default::default()
        },
    )
    .unwrap_or_else(|err| panic!("Failed to create a logical device: \n{:?}", err));

    let device_queues: Vec<Arc<Queue>> = queues_iterator.collect();

    return (logical_device, device_queues);


    /*
        @TODO: Fix the steps and shift it to the top of the file
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

fn draw_call(
    device: Arc<Device>,
    queue: Arc<Queue>,

    frambuffer: Arc<Framebuffer>,
    pipeline: Arc<GraphicsPipeline>,

    vertex_buffer: Subbuffer<[vertex::Vec2]>,
    framebuffer_object_image: Arc<Image>,

    target_buffer: Subbuffer<[u8]>,
) {
    // I need:
    // Command buffer built using AutoCommandBufferBuilder - done
    // render pass created for data
    // FBO - done
    // graphics pipeline object

    let queue_family_index = queue.queue_family_index();

    let cb_allocator = create_command_buffer_allocator(device.clone());
    let mut command_builder = AutoCommandBufferBuilder::primary(
        &cb_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    println!("This is good");

    command_builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(frambuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline)
        .unwrap()
        .bind_vertex_buffers(0, vertex_buffer)
        .unwrap()
        .draw(3, 1, 0, 0)
        .unwrap()
        .end_render_pass(Default::default())
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            framebuffer_object_image,
            target_buffer.clone(),
        ))
        .unwrap_or_else(|err| panic!("Failed to copy image to buffer: {:?}", err));
    println!("This is VGOOD");
    let command_buffer = command_builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    // Saving the rendered image to the filesystem
    let buffer_content = target_buffer.read().unwrap();
    let image: ImageBuffer<Rgba<u8>, &[u8]> = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();

    println!("Worked?");
}

// fn get_swapchain_image(device: Arc<Device>) -> _ {
//     todo!()
// }

fn get_auto_command_buffer(
    cb_allocator: &StandardCommandBufferAllocator,
    queue_family_index: u32,
) -> AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer> {
    let mut cb_builder: AutoCommandBufferBuilder<
        vulkano::command_buffer::PrimaryAutoCommandBuffer,
    > = AutoCommandBufferBuilder::primary(
        cb_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    return cb_builder;
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

fn main() {
    let (window, elwt) = create_window();
    let (device, queues) = initialise_vulkan_runtime(window.clone(), &elwt);

    let use_queue = queues.iter().next().unwrap();

    let vertices = Vec::from([
        vertex::Vec2 { x: 0.0, y: 1.0 },
        vertex::Vec2 { x: -1.0, y: -1.0 },
        vertex::Vec2 { x: 1.0, y: -1.0 },
    ]);

    // let i1 = vertices.into_iter(); // consumes the object
    // let element = vertices.get(0);

    // let i2 = vertices.iter();
    // let i3 = vertices.iter_mut();

    // buffer allocator for memory buffer objects
    let memory_allocator: GenericBufferAllocator = create_buffer_allocator(device.clone());

    // let image_buffer = get_swapchain_image(device.clone());
    
    let image_buffer = get_image_buffer(memory_allocator.clone()); // data is placed on the GPU
    let vertex_buffer = create_buffer(
        memory_allocator.clone(),
        vertices.into_iter(),
        BufferUsage::VERTEX_BUFFER,
        MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    ); // data is streamed from CPU to GPU

    // For testing purposes. This buffer should be drawn into the swapchain of the window
    let output_image_buffer = create_buffer(
        memory_allocator.clone(),
        vec![0; 1024 * 1024 * 4].into_iter(),
        BufferUsage::TRANSFER_DST,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS
    );

    // @TODO: Analyze the following step carefully and understand it more deeply
    // let image = get_image_buffer(memory_allocator.clone());

    let render_pass = create_render_pass(device.clone()); // defines the schema information required for configuring the output of shaders to the framebuffer
    let fbo = get_framebuffer_object(render_pass.clone(), image_buffer.clone()); // binds the image to the framebuffer object

    let graphics_pipeline = create_graphics_pipeline(device.clone(), render_pass);

    draw_call(
        device.clone(),
        use_queue.clone(),
        fbo,
        graphics_pipeline,
        vertex_buffer,
        image_buffer,
        output_image_buffer,
    );

    start_window_event_loop(window.clone(), elwt);
}

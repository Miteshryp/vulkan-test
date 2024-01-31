mod graphics_pack;
// mod vertex;

use image::{
    codecs::png::{self, PngDecoder},
    io, GenericImageView, ImageBuffer, ImageDecoder, ImageFormat,
};
use nalgebra_glm as glm;

use graphics_pack::{
    buffers::{
        self,
        base_buffer::{DeviceBuffer, StagingBuffer, StagingBufferMap},
        primitives::{InstanceData, Vec2, Vec3, VertexData},
        uniform_buffer::{UniformBuffer, UniformSet},
    },
    components::camera,
    shaders::{self, fs},
};

use std::{io::Read, process::exit, sync::Arc, time::SystemTime};
use vulkano::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
        CopyBufferToImageInfo, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayoutCreateFlags,
        persistent, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{self, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Properties, Queue, QueueCreateInfo,
        QueueFamilyProperties, QueueFlags,
    },
    image::{
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
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
    event::{
        DeviceEvent, ElementState, Event, KeyEvent, Modifiers, MouseButton, MouseScrollDelta,
        RawKeyEvent, WindowEvent,
    },
    event_loop::{
        self, ControlFlow, DeviceEvents, EventLoop, EventLoopBuilder, EventLoopWindowTarget,
    },
    keyboard::{Key, KeyCode, ModifiersState, PhysicalKey},
    platform::{modifier_supplement::KeyEventExtModifierSupplement, x11::EventLoopBuilderExtX11},
    window::{Window, WindowBuilder},
};

// type name for buffer allocator
type GenericBufferAllocator =
    Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;
type CommandBufferType = Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

type PrimaryAutoCommandBuilderType = AutoCommandBufferBuilder<
    PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
    Arc<StandardCommandBufferAllocator>,
>;

struct BufferUploader {
    // command_builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    command_builder: PrimaryAutoCommandBuilderType,
    buffer_allocator: GenericBufferAllocator,

    frame_map: Vec<StagingBufferMap>,
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
            frame_map: vec![],
        }
    }

    fn insert_buffer<T>(&mut self, buffer: StagingBuffer<T>, usage: BufferUsage) -> DeviceBuffer<T>
    where
        T: BufferContents + Clone,
    {
        // Get the host and device buffers from the staging buffer
        let buffer_count = buffer.count();
        let (host_buffer, device_buffer) =
            buffer.create_buffer_mapping(self.buffer_allocator.clone(), usage);

        // Create a mapping object and store it vec of mappings
        self.command_builder
            .copy_buffer(CopyBufferInfo::buffers(host_buffer, device_buffer.clone()));

        // let upload_mapping = StagingBufferMap {
        //     host_buffer: host_buffer.clone().into_bytes(),
        //     device_buffer: device_buffer.clone().into_bytes(),
        // };
        // self.frame_map.push(upload_mapping);

        // return the device buffer for further usage in the rendering command buffer.
        DeviceBuffer {
            buffer: device_buffer,
            count: buffer_count as u32,
        }
    }

    fn insert_image(
        &mut self,
        buffer: StagingBuffer<u8>,
        create_info: ImageCreateInfo,
    ) -> Arc<ImageView> {
        let image_object = Image::new(
            self.buffer_allocator.clone(),
            create_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let image_view = ImageView::new_default(image_object.clone()).unwrap();
        let src_buffer = buffer.create_host_buffer(self.buffer_allocator.clone(), Default::default());

        self.command_builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src_buffer, image_object.clone()));

        return image_view;
    }

    fn get_one_time_command_buffer(self) -> CommandBufferType {
        self.command_builder.build().unwrap()
    }
}

struct RenderTargetInfo {
    pipeline: Arc<GraphicsPipeline>,
    // render_pass: Arc<RenderPass>,
    fbos: Vec<Arc<Framebuffer>>,
    image_sampler: Arc<Sampler>,
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

// New Rendering system
// 1. Push the data into a upload structure
// 2. Get the device buffer mapping, where the data is eventually going to end up. The device buffer which is extracted from this mapping is going to be passed into the rendering command buffer construction.
// 3. create a command buffer in the upload structure
// 4. get the per frame upload command buffer future and wait before executing the rendering command buffer

impl VulkanInstance {
    fn initialise(window: Arc<Window>, el: &EventLoop<()>) -> Self {
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

    let vertex_shader_input_state = [VertexData::per_vertex(), InstanceData::per_instance()]
        .definition(&vertex_shader.info().input_interface)
        .unwrap();

    // This creation moves the vertex and fragment shaders,
    // so we cannot use those objects after this point
    let pipeline_stages = [
        PipelineShaderStageCreateInfo::new(vertex_shader),
        PipelineShaderStageCreateInfo::new(fragment_shader),
    ];

    let window_size = window.inner_size();
    let mut descriptor_set_layout =
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stages);

    // Enabling descriptor pushes on set 0
    unsafe {
        let set_layout = &mut descriptor_set_layout.set_layouts[PUSH_DESCRIPTOR_INDEX];
        set_layout.flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;
        set_layout.bindings.get_mut(&0).unwrap().immutable_samplers = vec![Sampler::new(
            logical_device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::Repeat; 3],
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                ..Default::default()
            },
        )
        .unwrap()];
    }

    let mut descriptor_create_info = descriptor_set_layout
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
            input_assembly_state: Some(InputAssemblyState {
                topology: graphics::input_assembly::PrimitiveTopology::TriangleList,
                ..Default::default()
            }),

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
    // key_event: RawKeyEvent,
    window_target: &EventLoopWindowTarget<()>,
    // view_matrix: &mut glm::TMat4<f32>,
    camera: &mut glm::TVec3<f32>,
) {
    let mut modifiers = ModifiersState::default();

    match event {
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

        // WindowEvent::CursorMoved {
        //     position,
        //     device_id,
        // } => {
        //     println!("Mouse moved: {:?}", position);
        // }
        WindowEvent::KeyboardInput {
            event: key_event, ..
        } => match key_event.state {
            ElementState::Pressed => match key_event.key_without_modifiers().as_ref() {
                Key::Character("w") => {
                    println!("Forward");
                    unsafe {
                        camera.z -= MOVE_SPEED;
                    }
                }
                Key::Character("a") => {
                    println!("Left");
                    unsafe {
                        camera.x -= MOVE_SPEED;
                    }
                }
                Key::Character("s") => {
                    println!("Backward");
                    unsafe {
                        camera.z += MOVE_SPEED;
                    }
                }
                Key::Character("d") => {
                    println!("Right");
                    unsafe {
                        camera.x += MOVE_SPEED;
                    }
                }
                _ => (),
            },
            _ => (),
        },
        _ => (),
    }
    return;

    // match key_event.state {
    //     ElementState::Pressed => match key_event.physical_key {
    //         PhysicalKey::Code(KeyCode::KeyA) => {
    //             println!("Left");
    //             unsafe {
    //                 camera.x -= MOVE_SPEED;
    //             }
    //         }
    //         PhysicalKey::Code(KeyCode::KeyD) => {
    //             println!("Right");
    //             unsafe {
    //                 camera.x += MOVE_SPEED;
    //             }
    //         }
    //         PhysicalKey::Code(KeyCode::KeyW) => {
    //             println!("Forward");
    //             unsafe {
    //                 camera.z -= MOVE_SPEED;
    //             }
    //         }
    //         PhysicalKey::Code(KeyCode::KeyS) => {
    //             println!("Backward");
    //             unsafe {
    //                 camera.z += MOVE_SPEED;
    //             }
    //         }
    //         _ => (),
    //     },

    //     ElementState::Released => {}
    // }
}

fn create_window() -> (Arc<Window>, EventLoop<()>) {
    // let winit_event_loop = event_loop::EventLoop::new().unwrap();

    // INFO Forcing X11 usage due to driver incompatibility with wayland
    // vulkan. Wayland vulkan is crashing consistently, even the vkcube-wayland fails.

    let mut winit_event_loop: Option<EventLoop<()>> = None;
    let mut event_loop_builder = event_loop::EventLoopBuilder::new();

    if cfg!(target_os = "linux") {
        println!("Initialising X11 Window");
        event_loop_builder.with_x11();
        winit_event_loop = Some(event_loop_builder.build().unwrap());
    } else {
        winit_event_loop = Some(EventLoop::new().unwrap());
    }

    let mut winit_event_loop: EventLoop<()> = winit_event_loop.unwrap();
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


fn create_data() -> (Vec<VertexData>, Vec<u32>, Vec<InstanceData>) {
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
        VertexData {
            position: Vec3 {
                x: -1.0,
                y: -1.0,
                z: -0.5,
            },
            color: color1.clone(),
            tex_coord: Vec2 { x: 0.0, y: 0.0 },
        },
        VertexData {
            position: Vec3 {
                x: -1.0,
                y: 1.0,
                z: -0.5,
            },
            color: color2.clone(),
            tex_coord: Vec2 { x: 0.0, y: 1.0 },
        },
        VertexData {
            position: Vec3 {
                x: 1.0,
                y: 1.0,
                z: -0.5,
            },
            color: color2.clone(),
            tex_coord: Vec2 { x: 1.0, y: 1.0 },
        },
        VertexData {
            position: Vec3 {
                x: 1.0,
                y: -1.0,
                z: -0.5,
            },
            color: color3.clone(),
            tex_coord: Vec2 { x: 1.0, y: 0.0 },
        },
    ]);

    let indicies: Vec<u32> = Vec::from([0, 1, 2, 0, 2, 3]);
    let instance_buffer_vec: Vec<InstanceData> = Vec::from([
        InstanceData {
            global_position: Vec3 {
                x: 0.0,
                y: 1.7,
                z: -3.0,
            },
            local_scale: 1.0,
        },
        InstanceData {
            global_position: Vec3 {
                x: 0.0,
                y: 4.7,
                z: -3.0,
            },
            local_scale: 1.4,
        },
    ]);

    return (vertices, indicies, instance_buffer_vec)

}


fn start_window_event_loop(window: Arc<Window>, el: EventLoop<()>, mut instance: VulkanInstance) {
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let (vertices, indicies, instance_buffer_vec) = create_data();
    

    let mut camera_position = glm::vec3(0.0, 0.0, 0.0);

    // Creating texture array
    let image1 = load_image(String::from("./sample1.png"));
    let image2 = load_image(String::from("./sample2.png"));

    let image_width = u32::max(image1.1 .0, image2.1 .0);
    let image_height = u32::max(image1.1 .1, image2.1 .1);

    let mut image_buffer_object = StagingBuffer::new();
    image_buffer_object.add_vec(&image1.0);
    // image_buffer_object.add_vec(&image2.0);
    // let mut image_buffer_object = ImageArrayBuffer::new(image_width, image_height, 4);
    // let image_object = image_buffer_object.create_image_object(instance.allocators.memory_allocator.clone());

    // let image_buffer_object = image_buffer_object.get_buffer(instance.allocators.memory_allocator.clone());

    let mut single_upload_buffer = BufferUploader::new(
        instance.allocators.command_buffer_allocator.clone(),
        instance.allocators.memory_allocator.clone(),
        instance.queue.queue_family_index(),
    );

    let image_view = single_upload_buffer.insert_image(
        image_buffer_object,
        ImageCreateInfo {
            // array_layers: 2,
            format: vulkano::format::Format::R8G8B8A8_SRGB,
            extent: [
                // u32::max(image1.1 .0, image2.1 .0),
                // u32::max(image1.1 .1, image2.1 .1),
                image1.1.0, image1.1.1, 
                1,
            ],
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            image_type: ImageType::Dim2d,
            ..Default::default()
        },
    );

    let single_upload_future = single_upload_buffer.get_one_time_command_buffer().execute(instance.queue.clone()).unwrap();
    sync::now(instance.get_logical_device()).join(single_upload_future);

    // let image_view = ImageView::new_default(image_object).unwrap();

    let sampler_uniform =
        UniformBuffer::create_immutable_sampler(0, instance.render_target.image_sampler.clone());

    // Writing texture data into the uniform
    let texture_uniform = UniformBuffer::create_image_view(1, image_view.clone()); // Single texture write
                                                                                   // let texture_uniform = UniformBuffer::create_image_view_array(1, image_view.clone());

    let mut first_buffer_write = true;

    el.set_control_flow(ControlFlow::Poll);
    let _ = el.run(|app_event, elwt| match app_event {
        // Window based Events
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
        }

        Event::DeviceEvent {
            device_id,
            event: DeviceEvent::MouseMotion { delta },
        } => {
            println!("Mouse moved: {:?}", delta);
            // Rotate the mouse based on this (delta / sensitivity_factor)
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
                &glm::vec3(
                    camera_position.x,
                    camera_position.y,
                    camera_position.z + -0.01,
                ),
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

            // let mut data = shaders::vs::Data {
            let mut data = shaders::vs::PushConstantData {
                view: unsafe { START.unwrap().elapsed().unwrap().as_secs_f32() },
            };

            // uniform_set.add_uniform_buffer(
            //     instance.allocators.memory_allocator.clone(),
            //     instance.get_graphics_pipeline(),
            //     data,
            //     Default::default(),
            // );
            // let mut uni0 = UniformBuffer::create(
            //     instance.allocators.memory_allocator.clone(),
            //     0,
            //     data,
            //     Default::default(),
            // );
            // }

            // uniform 1
            // {
            let mut data1 = shaders::vs::MvpMatrix {
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
                2,
                data1,
                Default::default(),
            );
            // }


            let mut vertex_staging_buffer = StagingBuffer::new();
            vertex_staging_buffer.add_vec(&vertices);
            let mut instance_staging_buffer = StagingBuffer::new();
            instance_staging_buffer.add_vec(&instance_buffer_vec);

            let mut index_staging_buffer = StagingBuffer::new();
            index_staging_buffer.add_vec(&indicies);

            // Uploading buffer
            let mut buffer_uploader = BufferUploader::new(
                instance.allocators.command_buffer_allocator.clone(),
                instance.allocators.memory_allocator.clone(),
                instance.queue.queue_family_index(),
            );

            let device_vertex_buffer = buffer_uploader.insert_buffer(vertex_staging_buffer, BufferUsage::VERTEX_BUFFER);
            let device_instance_buffer = buffer_uploader.insert_buffer(instance_staging_buffer, BufferUsage::VERTEX_BUFFER);
            let device_index_buffer = buffer_uploader.insert_buffer(index_staging_buffer, BufferUsage::INDEX_BUFFER);

            let upload_future = buffer_uploader
                .get_one_time_command_buffer()
                .execute(instance.get_first_queue())
                .unwrap();

            let command_buffers = create_command_buffers(
                &instance, 
                device_vertex_buffer, 
                device_index_buffer, 
                device_instance_buffer, 

                // image_data, 
                vec![uni1, sampler_uniform.clone(), texture_uniform.clone()], 
                data
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
                Err(e) => panic!("Failed to acquire next image from swapchain: {e}"),
            };

            if is_suboptimal {
                recreate_swapchain = true;
            }

            // Future is a point where GPU gets access to the data
            let future = sync::now(instance.get_logical_device())
                .join(acquired_future)
                // .join(single_upload_future)
                .join(upload_future)
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
                    // window.request_redraw();
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
        Event::AboutToWait => window.request_redraw(),

        Event::WindowEvent {
            window_id,
            event: WindowEvent::CloseRequested,
        } => {
            // Handling the window event if the event is bound to the owned window
            elwt.exit();
            exit(0);
        }

        // Event::DeviceEvent {
        //     device_id,
        //     event: DeviceEvent::Key(key_event),
        // } => {
        //     winit_handle_window_events(key_event, elwt, &mut camera_position);
        // }
        Event::WindowEvent {
            window_id,
            event, // event: WindowEvent::KeyboardInput { event, .. },
        } => {
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

static mut START: Option<SystemTime> = None;
static mut MOVE_SPEED: f32 = 0.1;
static mut PUSH_DESCRIPTOR_INDEX: usize = 1;

fn refresh_render_target(
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
        create_graphics_pipeline(window.clone(), device.clone(), render_pass.clone());

    RenderTargetInfo {
        pipeline: graphics_pipeline,
        image_sampler: sampler,
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

    // staging data objects
    device_vertex_buffer: DeviceBuffer<VertexData>,
    device_index_buffer: DeviceBuffer<u32>,
    device_instance_buffer: DeviceBuffer<InstanceData>,

    uniforms: Vec<UniformBuffer>,
    push_constant_data: shaders::vs::PushConstantData, // uniform_buffer_data: graphics_pack::shaders::vs::Data,
) -> Vec<CommandBufferType> {

    // TODO:
    // 1. Get the buffer from the staging buffer object
    // 2. Get the sub buffer object from the staging buffer
    // 3. Transfer the data into a final buffer using a new upload command buffer
    // 4. Pass the final buffer into the fbo builder function

    let graphics_pipeline = instance.get_graphics_pipeline();

    instance
        .render_target
        .fbos
        .iter()
        .map(|fb| {
            let mut command_builder = AutoCommandBufferBuilder::primary(
                &instance.allocators.command_buffer_allocator,
                instance.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

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
                // .bind_index_buffer(index_subbuffer)
                .bind_index_buffer(device_index_buffer.buffer.clone())
                .unwrap()
                // .bind_vertex_buffers(0, (vertex_subbuffer, instance_subbuffer))
                .bind_vertex_buffers(
                    0,
                    (device_vertex_buffer.buffer.clone(), device_instance_buffer.buffer.clone()),
                )
                .unwrap();

            // Binding uniforms (descriptor sets)
            // command_builder
            //     .bind_descriptor_sets(
            //         PipelineBindPoint::Graphics,
            //         graphics_pipeline.layout().clone(),
            //         0,
            //         persistent_descriptor_sets.clone(),
            //     )
            //     .unwrap();

            let push_descriptor_writes = uniforms
                .clone()
                .into_iter()
                .map(|ub| ub.get_write_descriptor())
                .collect();
            let mut push_descriptor_index: u32 = 0;
            unsafe {
                push_descriptor_index = PUSH_DESCRIPTOR_INDEX as u32;
            }

            command_builder
                .push_descriptor_set(
                    PipelineBindPoint::Graphics,
                    graphics_pipeline.layout().clone(),
                    push_descriptor_index, // index of set where the data is being written,
                    push_descriptor_writes,
                )
                .unwrap();

            command_builder
                .push_constants(graphics_pipeline.layout().clone(), 0, push_constant_data)
                .unwrap();

            // Draw call
            command_builder
                // NOTE: This is the draw call used for indexed rendering
                // .draw_indexed(index_count, index_count / 3, 0, 0, 0)
                .draw_indexed(
                    device_index_buffer.count,
                    device_index_buffer.count / 3,
                    0,
                    0,
                    0,
                )
                .unwrap()
                .end_render_pass(Default::default())
                .unwrap();

            let cb = command_builder.build().unwrap();
            return cb;
        })
        .collect()

    // let command_buffers = build_fbo_command_buffers_for_pipeline(
    //     instance.allocators.command_buffer_allocator.clone(),
    //     instance.get_first_queue(),
    //     instance.get_graphics_pipeline(),
    //     &instance.render_target.fbos,
    //     vertex_buffer,
    //     instance_buffer,
    //     index_buffer,
    //     uniforms,
    //     push_constant_data,
    //     image_object,
    //     image_buffer,
    // );

    // return command_buffers;
}

/*
   Steps to setup stencil buffer

   create a depth image.
   update the render pass.
   set a new depth stencil state in the pipeline.
   update the framebuffers.
   update draw function.
   clean.
*/

fn load_image(path: String) -> (Vec<u8>, (u32, u32)) {
    let dynamic_image = io::Reader::open(path)
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();
    let image_dimensions = dynamic_image.dimensions();
    let image_buffer = dynamic_image.as_bytes();

    (image_buffer.to_vec(), image_dimensions)
}

// fn load_png_into_image(
//     allocator: GenericBufferAllocator,
//     path: String,
// ) -> (
//     Arc<Image>,
//     graphics_pack::buffers::image_buffer::ImageBuffer,
// ) {
//     // let raw_img = include_bytes!(path).as_slice();

//     // let img_path = std::path::Path::new(path.as_str());
//     // let img_file = std::fs::File::open(&img_path).unwrap();
//     let dynamic_image = image::io::Reader::open(path)
//         .unwrap()
//         .with_guessed_format()
//         .unwrap()
//         .decode()
//         .unwrap();
//     let image_dimensions = dynamic_image.dimensions();

//     let img_src_buffer = buffers::image_buffer::ImageBuffer::new(
//         allocator.clone(),
//         // img_buffer,
//         &dynamic_image.into_rgba8().into_vec(),
//         [image_dimensions.0, image_dimensions.1, 4],
//     );
//     println!("Image size: {} {}", image_dimensions.0, image_dimensions.1);

//     // let (src_buffer, _) = img_src_buffer.consume();

//     let dst_img: Arc<Image> = Image::new(
//         allocator.clone(),
//         ImageCreateInfo {
//             usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
//             image_type: vulkano::image::ImageType::Dim2d,
//             extent: [image_dimensions.0, image_dimensions.1, 1],
//             format: vulkano::format::Format::R8G8B8A8_SRGB,
//             ..Default::default()
//         },
//         AllocationCreateInfo {
//             memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
//             ..Default::default()
//         },
//     )
//     .unwrap();

//     return (dst_img, img_src_buffer);
// }

// different - image object parameters, buffer size

// Model 1 -
// 1. create image data buffer with appropriate size to contain data of imageS.
// 2. add single image to created buffer
// 3. create image object and image view from the buffer data (Image buffer needs extra parameters for this, basically needs to be revamped entirely)
// 4. pass image and buffer into the command builder function to copy the data

// Model 2 -
// 1. Create a load_png_image_array function, which returns a image object configured to store texture array and buffer with the image data
// 2. pass the buffer and image into the rendering function

// fn create_image_view_from_buffer(buffer: graphics_pack::buffers::image_buffer::ImageBuffer) {}

fn main() {
    unsafe {
        START = Some(SystemTime::now());
    }

    let (window, elwt) = create_window();
    let vulkan_instance = VulkanInstance::initialise(window.clone(), &elwt);

    start_window_event_loop(window.clone(), elwt, vulkan_instance);
}

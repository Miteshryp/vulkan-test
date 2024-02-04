mod graphics_pack;
// mod vertex;

use image::{
    codecs::png::{self, PngDecoder},
    io, DynamicImage, GenericImageView, ImageBuffer, ImageDecoder, ImageFormat,
};
use nalgebra_glm as glm;

use graphics_pack::{
    buffers::{
        self,
        base_buffer::{DeviceBuffer, StagingBuffer},
        image_buffer::{StagingImageArrayBuffer, StagingImageBuffer},
        primitives::{CommandBufferType, InstanceData, Vec2, Vec3, VertexData},
        uniform_buffer::{UniformBuffer, UniformSet},
    },
    components::{
        camera::{self, Camera},
        input_handler::KeyboardInputHandler,
        uploader::BufferUploader,
        vulkan::{VulkanInstance, VulkanSwapchainInfo},
    },
    pipelines::base_pipeline::GraphicsPipelineBuilder,
    shaders::{self, deferred, lighting},
};
use smallvec::SmallVec;

use std::{env, io::Read, process::exit, sync::Arc, time::SystemTime};
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
        SubpassEndInfo,
    },
    image::{
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    instance::InstanceCreateInfo,
    memory::allocator::{
        AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint},
    swapchain::{self, SwapchainPresentInfo},
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

// import this type from base_buffer.rs
// type GenericBufferAllocator =
//     Arc<GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>;

// type CommandBufferType = Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>;

// type PrimaryAutoCommandBuilderType = AutoCommandBufferBuilder<
//     PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
//     Arc<StandardCommandBufferAllocator>,
// >;

#[allow(unused_variables)]
#[allow(unused_assignments)]
fn winit_handle_window_events(
    event: WindowEvent,
    input_handler: &mut KeyboardInputHandler,
    // key_event: RawKeyEvent,
    window_target: &EventLoopWindowTarget<()>,
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
                    println!("Right click detected");
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
        } => {
            input_handler.update_input(key_event);
        }

        _ => (),
    }
}



fn update_camera_position(camera: &mut Camera, input_handler: &KeyboardInputHandler) {
    unsafe {
        if (input_handler.is_pressed(KeyCode::KeyW)) {
            camera.move_forward(MOVE_SPEED);
        }
        if (input_handler.is_pressed(KeyCode::KeyS)) {
            camera.move_backward(MOVE_SPEED);
        }
        if (input_handler.is_pressed(KeyCode::KeyA)) {
            camera.move_left(MOVE_SPEED);
        }
        if (input_handler.is_pressed(KeyCode::KeyD)) {
            camera.move_right(MOVE_SPEED);
        }
    }
}

fn create_window() -> (Arc<Window>, EventLoop<()>) {
    // INFO: Forcing X11 usage due to driver incompatibility with wayland
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

fn create_cube_vertices() -> (Vec<VertexData>, Vec<u32>) {
    let color1 = Vec3::new(1.0, 0.0, 0.0);
    let color2 = Vec3::new(0.0, 1.0, 0.0);
    let color3 = Vec3::new(0.0, 0.0, 1.0);

    let vertices = Vec::from([
        // front face
        VertexData {
            position: Vec3::new(-1.0, -1.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            color: color1.clone(),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, -1.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            color: color3.clone(),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, 1.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            color: color2.clone(),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        VertexData {
            position: Vec3::new(-1.0, 1.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            color: color2.clone(),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        // Back face
        VertexData {
            position: Vec3::new(1.0, -1.0, -1.0),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color3.clone(),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        VertexData {
            position: Vec3::new(-1.0, -1.0, -1.0),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color1.clone(),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        VertexData {
            position: Vec3::new(-1.0, 1.0, -1.0),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color2.clone(),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        VertexData {
            position: Vec3::new(1.0, 1.0, -1.0),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color2.clone(),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        // Left face
        VertexData {
            position: Vec3::new(-1.0, -1.0, -1.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            color: color3.clone(),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        VertexData {
            position: Vec3::new(-1.0, -1.0, 1.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            color: color1.clone(),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        VertexData {
            position: Vec3::new(-1.0, 1.0, 1.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        VertexData {
            position: Vec3::new(-1.0, 1.0, -1.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        // Right face
        VertexData {
            position: Vec3::new(1.0, -1.0, 1.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            color: color1.clone(),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, -1.0, -1.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            color: color3.clone(),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, 1.0, -1.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        VertexData {
            position: Vec3::new(1.0, 1.0, 1.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        // Top face
        VertexData {
            position: Vec3::new(-1.0, -1.0, -1.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
            color: color3.clone(),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, -1.0, -1.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
            color: color1.clone(),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, -1.0, 1.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        VertexData {
            position: Vec3::new(-1.0, -1.0, 1.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(0.0, 1.0),
        },
        // Bottom face
        VertexData {
            position: Vec3::new(-1.0, 1.0, 1.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            color: color3.clone(),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, 1.0, 1.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            color: color1.clone(),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, 1.0, -1.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        VertexData {
            position: Vec3::new(-1.0, 1.0, -1.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            color: color2.clone(),
            tex_coord: Vec2::new(0.0, 1.0),
        },
    ]);

    let indicies = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    ];

    (vertices, indicies)
}

fn create_square_geometry() -> (Vec<VertexData>, Vec<u32>) {
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
            position: Vec3::new(-1.0, -1.0, -0.5),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color1.clone(),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, -1.0, -0.5),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color3.clone(),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        VertexData {
            position: Vec3::new(1.0, 1.0, -0.5),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color2.clone(),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        VertexData {
            position: Vec3::new(-1.0, 1.0, -0.5),
            normal: Vec3::new(0.0, 0.0, -1.0),
            color: color2.clone(),
            tex_coord: Vec2::new(0.0, 1.0),
        },
    ]);

    let indicies: Vec<u32> = Vec::from([0, 1, 2, 0, 2, 3]);

    (vertices, indicies)
}

fn create_data() -> (Vec<VertexData>, Vec<u32>, Vec<InstanceData>) {
    // let (vertices, indicies) = create_square_geometry();
    let (vertices, indicies) = create_cube_vertices();

    let instance_buffer_vec: Vec<InstanceData> = Vec::from([
        InstanceData {
            global_position: Vec3 {
                x: 0.0,
                y: 1.7,
                z: -3.0,
            },
            local_scale: 1.0,
            tex_index: 0,
        },
        InstanceData {
            global_position: Vec3 {
                x: 0.0,
                y: 4.7,
                z: -3.0,
            },
            local_scale: 1.4,
            tex_index: 1,
        },
    ]);

    return (vertices, indicies, instance_buffer_vec);
}

fn start_window_event_loop(window: Arc<Window>, el: EventLoop<()>, mut instance: VulkanInstance) {
    let mut keyboard_handler = KeyboardInputHandler::new();

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let (vertices, indicies, instance_buffer_vec) = create_data();

    // let mut camera_position = glm::vec3(0.0, 0.0, 0.0);
    let mut camera = Camera::new(
        glm::vec3(0.0, 0.0, 0.0),
        glm::vec3(0.0, 0.0, -1.0),
        glm::vec3(0.0, 1.0, 0.0),
        std::f32::consts::FRAC_PI_4,
        window.inner_size().width as f32 / window.inner_size().height as f32,
        0.01,
        1000.0,
    );

    // Creating texture array
    let image1 = load_image(String::from("./sample1.png"));
    let image2 = load_image(String::from("./sample2.png"));

    let image_width = image1.1 .0;
    let image_height = image1.1 .1;

    // let mut image_buffer_object = StagingImageBuffer::from_vec_ref(&image1.0, image1.1.0, image1.1.1, 4);
    let mut image_array_buffer = StagingImageArrayBuffer::new(image_width, image_height, 4);
    image_array_buffer.add_image_data(&image1.0);
    image_array_buffer.add_image_data(&image2.0);

    let mut single_upload_buffer = BufferUploader::new(
        instance.allocators.command_buffer_allocator.clone(),
        instance.allocators.memory_allocator.clone(),
        instance.get_first_queue().queue_family_index(),
    );

    // let image_view = single_upload_buffer.insert_image(image_buffer_object);
    let image_view = single_upload_buffer.insert_image_array(image_array_buffer);

    // Single time uploader
    let single_upload_future = single_upload_buffer
        .get_one_time_command_buffer()
        .execute(instance.get_first_queue().clone())
        .unwrap();
    let _ = sync::now(instance.get_logical_device()).join(single_upload_future);

    // let image_view = ImageView::new_default(image_object).unwrap();

    // let sampler_uniform =
    //     UniformBuffer::create_immutable_sampler(0, instance.render_target.image_sampler.clone());
    let sampler_uniform =
        UniformBuffer::create_immutable_sampler(1, instance.render_target.image_sampler.clone());

    // Writing texture data into the uniform
    // let texture_uniform = UniformBuffer::create_image_view(1, image_view.clone()); // Single texture write
    let texture_uniform = UniformBuffer::create_image_view(2, image_view.clone()); // Single texture write
                                                                                   // let texture_uniform = UniformBuffer::create_image_view_array(1, image_view.clone());

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
            let sen_x = 0.5;
            let sen_y = 0.3;
            camera.rotate(delta.0 as f32 * sen_x, delta.1 as f32 * sen_y);
            // println!("Mouse moved: {:?}", delta);
            // Rotate the mouse based on this (delta / sensitivity_factor)
        }

        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            // draw call

            update_camera_position(&mut camera, &keyboard_handler);

            if window_resized || recreate_swapchain {
                // recreating swapchains
                instance.refresh_instance_swapchain(window.clone()); // refresh_instance_swapchain(window.clone(), &mut instance);

                // refreshing the render targets
                // recreating the render pass, fbos, pipeline and command buffers with the new swapchain images
                instance.render_target = VulkanInstance::create_render_target(
                    window.clone(),
                    instance.get_logical_device(),
                    &instance.swapchain_info,
                    instance.allocators.memory_allocator.clone(),
                );

                window_resized = false;
                recreate_swapchain = false;
            }

            // let mut uniform_set = UniformSet::new(0);

            
            // let mut data = shaders::vs::Data {
            // let mut data = shaders::basic::vs::PushConstantData {
            //     view: unsafe { START.unwrap().elapsed().unwrap().as_secs_f32() },
            // };


            let model = glm::identity::<f32, 4>();
            let view = camera.get_view_matrix_data();
            let projection = camera.get_projection_matrix_data();

            let mut mvp_data = shaders::basic::vs::MvpMatrix {
                model: Into::<[[f32; 4]; 4]>::into(model),
                view: Into::<[[f32; 4]; 4]>::into(view),
                projection: Into::<[[f32; 4]; 4]>::into(projection),
            };

            let mut mvp_uniform = UniformBuffer::create(
                instance.allocators.memory_allocator.clone(),
                // 1,
                0,
                mvp_data,
                Default::default(),
            );



            let mut attachment_color = UniformBuffer::create_image_view(
                0,
                instance.render_target.attachments.color_image_view.clone(),
            );
            let mut attachment_normal = UniformBuffer::create_image_view(
                1,
                instance.render_target.attachments.normal_image_view.clone(),
            );



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
                instance.get_first_queue().queue_family_index(),
            );

            // Getting final device buffers
            let device_vertex_buffer =
                buffer_uploader.insert_buffer(vertex_staging_buffer, BufferUsage::VERTEX_BUFFER);
            let device_instance_buffer =
                buffer_uploader.insert_buffer(instance_staging_buffer, BufferUsage::VERTEX_BUFFER);
            let device_index_buffer =
                buffer_uploader.insert_buffer(index_staging_buffer, BufferUsage::INDEX_BUFFER);

            // Point in time where device buffers are populated
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
                vec![mvp_uniform, sampler_uniform.clone(), texture_uniform.clone()],
                // vec![attachment_color, attachment_normal], // None,
                vec![attachment_color]
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

            match future.map_err(Validated::unwrap) {
                Ok(future) => {
                    // Wait for the GPU to finish.
                    future.wait(None).unwrap();
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
                // winit_handle_window_events(event, elwt, &mut camera_position);
                winit_handle_window_events(event, &mut keyboard_handler, elwt);
            }
        }
        Event::LoopExiting => {
            println!("Exiting the event loop");
            exit(0);
        }
        _ => (),
    });
}

static mut START: Option<SystemTime> = None;
static mut MOVE_SPEED: f32 = 0.1;
static mut PUSH_DESCRIPTOR_INDEX: usize = 1;

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
    attachments: Vec<UniformBuffer>, // push_constant_data: Option<impl BufferContents + Clone>, // uniform_buffer_data: graphics_pack::shaders::vs::Data,
) -> Vec<CommandBufferType> {
    // TODO:
    // 1. Get the buffer from the staging buffer object
    // 2. Get the sub buffer object from the staging buffer
    // 3. Transfer the data into a final buffer using a new upload command buffer
    // 4. Pass the final buffer into the fbo builder function

    // let graphics_pipeline = instance.get_graphics_pipeline();
    let graphics_pipeline = instance.render_target.pipeline.clone();
    let lighting_pipeline = instance.render_target.lighting_pipeline.clone();

    instance
        .render_target
        .fbos
        .iter()
        .map(|fb| {
            let mut command_builder = AutoCommandBufferBuilder::primary(
                &instance.allocators.command_buffer_allocator,
                instance.get_first_queue().queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            // Push descriptor writes
            let push_descriptor_writes: SmallVec<_> = uniforms
                .clone()
                .into_iter()
                .map(|ub| ub.get_write_descriptor())
                .collect();

            // let attachment_writes: SmallVec<_> = attachments
            //     .clone()
            //     .into_iter()
            //     .map(|ub| ub.get_write_descriptor())
            //     .collect();

            let mut attachment_set = UniformSet::new(
                lighting_pipeline
                    .get_attachment_descriptor_set_index()
                    .unwrap() as usize
            );
            attachment_set.add_multiple_buffers(attachments.clone());

            // NOTE: This structure is only used to bind dynamic data to the graphics pipeline
            // Any modifications to the rendering stages have to be done in the
            // graphics pipeline while it is created.
            command_builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        // clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())],
                        // clear_values: vec![Some([0.0,0.0,0.0,1.0].into())],
                        clear_values: vec![
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            // Some([0_u32, 0, 0, 1].into()),
                            Some([0.0,0.0,0.0,1.0].into()),
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            Some(1.0.into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(fb.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap();

            // First Render Pass

            command_builder
                .bind_pipeline_graphics(graphics_pipeline.pipeline.clone())
                .unwrap();

            // Binding buffers
            command_builder
                .bind_index_buffer(device_index_buffer.buffer.clone())
                .unwrap()
                .bind_vertex_buffers(
                    0,
                    (
                        device_vertex_buffer.buffer.clone(),
                        device_instance_buffer.buffer.clone(),
                    ),
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

            // Push Descriptors
            command_builder
                .push_descriptor_set(
                    PipelineBindPoint::Graphics,
                    graphics_pipeline.pipeline.layout().clone(),
                    // push_descriptor_index, // index of set where the data is being written,
                    graphics_pipeline.get_push_descriptor_set_index(),
                    push_descriptor_writes.clone(),
                )
                .unwrap();

            
            // Draw call
            command_builder
                .draw_indexed(
                    device_index_buffer.count,
                    device_index_buffer.count / 3,
                    0,
                    0,
                    0,
                )
                .unwrap();





            // Second pass

            command_builder
                .next_subpass(
                    Default::default(),
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(lighting_pipeline.pipeline.clone())
                .unwrap();

            // Binding buffers
            command_builder
                .bind_vertex_buffers(
                    0,
                    (
                        device_vertex_buffer.buffer.clone(),
                        device_instance_buffer.buffer.clone(),
                    ),
                )
                .unwrap()
                .bind_index_buffer(device_index_buffer.buffer.clone())
                .unwrap();


            // Push Descriptors
            command_builder
                .push_descriptor_set(
                    GraphicsPipeline::bind_point(&lighting_pipeline.pipeline),
                    lighting_pipeline.pipeline.layout().clone(),
                    lighting_pipeline.get_push_descriptor_set_index(),
                    // TODO: Change this to a better solution
                    smallvec::smallvec![push_descriptor_writes[0].clone()], // Only taking the MVP for now.
                                                                            // push_descriptor_writes.clone()
                )
                .unwrap();

            command_builder
                .bind_descriptor_sets(
                    lighting_pipeline.pipeline.bind_point(),
                    lighting_pipeline.pipeline.layout().clone(),
                    lighting_pipeline.get_attachment_descriptor_set_index().unwrap(),
                    vec![attachment_set.clone().get_persistent_descriptor_set(
                        instance.allocators.descriptor_set_allocator.clone(),
                        lighting_pipeline.pipeline.clone(),
                    )],
                )
                .unwrap();

            command_builder
                .draw_indexed(
                    device_index_buffer.count,
                    device_index_buffer.count / 3,
                    0,
                    0,
                    0,
                )
                .unwrap();

            // Ending render pass
            command_builder.end_render_pass(Default::default()).unwrap();

            let cb = command_builder.build().unwrap();
            return cb;
        })
        .collect()
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
    // let dynamic_image = io::Reader::open(path)
    //     .unwrap()
    //     .with_guessed_format()
    //     .unwrap()
    //     .decode()
    //     .unwrap();

    let dynamic_image = image::open(path).unwrap();

    // only supporting 8bit image channels for now
    // TODO: Implement support to load and process images with different channel width in the rendering system
    let format = match dynamic_image.color() {
        image::ColorType::Rgba8 => Ok(()),
        _ => Err("Image does not have a 8bit channel"),
    }
    .unwrap();

    // dynamic_image.resize(128, 128, image::imageops::FilterType::Gaussian);

    let image_dimensions = dynamic_image.dimensions();
    let image_buffer: Vec<u8> = dynamic_image.into_bytes();
    // let image_buffer = dynamic_image.as_bytes();

    // (image_buffer.to_vec(), image_dimensions)
    (image_buffer, image_dimensions)
}

fn main() {
    // env::set_var("RUST_BACKTRACE", "1");
    unsafe {
        START = Some(SystemTime::now());
    }

    let (window, elwt) = create_window();
    let vulkan_instance = VulkanInstance::initialise(window.clone(), &elwt);

    start_window_event_loop(window.clone(), elwt, vulkan_instance);
}

mod graphics;

use image::{
    codecs::png::{self, PngDecoder},
    io, DynamicImage, GenericImageView, ImageBuffer, ImageDecoder, ImageFormat,
};
use nalgebra_glm as glm;

use graphics::{
    buffers::{
        self,
        base_buffer::{StagingBuffer},
        image_buffer::{StagingImageArrayBuffer, StagingImageBuffer},
        primitives::{InstanceData, Vec2, Vec3, VertexData},
    }, components::{
        camera::{self, Camera},
        input_handler::KeyboardInputHandler,
        uploader::BufferUploader,
        vulkan::{VulkanInstance, },
    }, renderer::deferred_renderer::{DeferredRendererData}, shaders::{self}
};

use std::{env, io::Read, process::exit, sync::Arc, time::SystemTime};
use vulkano::{
    buffer::{BufferUsage},
    command_buffer::{
        PrimaryCommandBufferAbstract
    },
    swapchain::{self, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
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

    // let global_pos1 = glm::vec3(0.0,1.7,-3.0);
    // let global_pos2 = glm::vec3(0.0,4.7,-3.0);

    
    let instance_buffer_vec: Vec<InstanceData> = Vec::from([
        InstanceData {
            global_position: Vec3 {
                x: 0.0,
                y: 1.7,
                z: -3.0,
            },
            local_scale: 1.0,
            tex_index: 0,
            // model: Into::<[[f32; 4]; 4]>::into(
            //     glm::translate(&glm::identity::<f32, 4>(), &global_pos1)
            // ),
        },
        InstanceData {
            global_position: Vec3 {
                x: 0.0,
                y: 4.7,
                z: -3.0,
            },
            local_scale: 1.4,
            tex_index: 1,
            // model: Into::<[[f32; 4]; 4]>::into(
            //     glm::translate(&glm::identity::<f32, 4>(), &global_pos2)
            // ),
        },
    ]);

    return (vertices, indicies, instance_buffer_vec);
}

fn start_window_event_loop(window: Arc<Window>, el: EventLoop<()>, mut instance: VulkanInstance) {
    let mut keyboard_handler = KeyboardInputHandler::new();

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let (vertices, indicies, instance_buffer_vec) = create_data();

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

    let mut image_array_buffer = StagingImageArrayBuffer::new(image_width, image_height, 4);
    image_array_buffer.add_image_data(&image1.0);
    image_array_buffer.add_image_data(&image2.0);

    let mut single_upload_buffer = BufferUploader::new(
        instance.allocators.command_buffer_allocator.clone(),
        instance.allocators.memory_allocator.clone(),
        instance.get_first_queue().queue_family_index(),
    );

    let image_view = single_upload_buffer.insert_image_array(image_array_buffer);

    // Single time uploader
    let single_upload_future = single_upload_buffer
        .get_one_time_command_buffer()
        .execute(instance.get_first_queue().clone())
        .unwrap();

    let _ = sync::now(instance.get_logical_device()).join(single_upload_future);


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

                // refreshing the renderer
                instance.renderer.refresh_render_target(instance.get_logical_device(), window.clone(), &instance.swapchain_info, instance.allocators.memory_allocator.clone());

                window_resized = false;
                recreate_swapchain = false;
            }


            let model = glm::identity::<f32, 4>();
            let view = camera.get_view_matrix_data();
            let projection = camera.get_projection_matrix_data();

            let mut mvp_data = shaders::deferred::vs::MvpMatrix {
                view: Into::<[[f32; 4]; 4]>::into(view),
                projection: Into::<[[f32; 4]; 4]>::into(projection),
            };



            let mut vertex_staging_buffer = StagingBuffer::from_vec_ref(&vertices);
            let mut instance_staging_buffer = StagingBuffer::from_vec_ref(&instance_buffer_vec);
            let mut index_staging_buffer = StagingBuffer::from_vec_ref(&indicies);


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

            instance.renderer.bind_vertex_buffer(device_vertex_buffer);
            instance.renderer.bind_instance_buffer(device_instance_buffer);
            instance.renderer.bind_index_buffer(device_index_buffer);

            let command_buffers = instance.renderer.render(
                instance.allocators.command_buffer_allocator.clone(),
                instance.allocators.memory_allocator.clone(),
                instance.allocators.descriptor_set_allocator.clone(),
                instance.get_first_queue().queue_family_index(),
                DeferredRendererData {
                    camera: &camera,
                    image_array_view: image_view.clone()
                }
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

        Event::WindowEvent {
            window_id,
            event, // event: WindowEvent::KeyboardInput { event, .. },
        } => {
            if window_id == window.id() {
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
    env::set_var("RUST_BACKTRACE", "1");
    unsafe {
        START = Some(SystemTime::now());
    }

    let (window, elwt) = create_window();
    let vulkan_instance = VulkanInstance::initialise(window.clone(), &elwt);

    start_window_event_loop(window.clone(), elwt, vulkan_instance);
}

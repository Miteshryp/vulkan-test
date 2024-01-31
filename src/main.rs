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
        base_buffer::{DeviceBuffer, StagingBuffer},
        primitives::{InstanceData, Vec2, Vec3, VertexData},
        uniform_buffer::{UniformBuffer, UniformSet},
    },
    components::{
        camera, 
        vulkan::{VulkanInstance, VulkanSwapchainInfo}
    },
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
    image::{
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    instance::InstanceCreateInfo,
    memory::allocator::{
            AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
        },
    pipeline::{
        Pipeline, PipelineBindPoint,
    },
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
            .copy_buffer(CopyBufferInfo::buffers(host_buffer, device_buffer.clone())).unwrap();


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
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src_buffer, image_object.clone())).unwrap();

        return image_view;
    }

    fn get_one_time_command_buffer(self) -> CommandBufferType {
        self.command_builder.build().unwrap()
    }
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
        instance.get_first_queue().queue_family_index(),
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

    let single_upload_future = single_upload_buffer.get_one_time_command_buffer().execute(instance.get_first_queue().clone()).unwrap();
    sync::now(instance.get_logical_device()).join(single_upload_future);

    // let image_view = ImageView::new_default(image_object).unwrap();

    let sampler_uniform =
        UniformBuffer::create_immutable_sampler(0, instance.render_target.image_sampler.clone());

    // Writing texture data into the uniform
    let texture_uniform = UniformBuffer::create_image_view(1, image_view.clone()); // Single texture write
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
                instance.get_first_queue().queue_family_index(),
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
                instance.get_first_queue().queue_family_index(),
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

fn main() {
    unsafe {
        START = Some(SystemTime::now());
    }

    let (window, elwt) = create_window();
    let vulkan_instance = VulkanInstance::initialise(window.clone(), &elwt);

    start_window_event_loop(window.clone(), elwt, vulkan_instance);
}

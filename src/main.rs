use std::{fmt::Error, process::exit, sync::Arc};
use vulkano::{
    device::{
        physical::{self, PhysicalDevice},
        Device, DeviceCreateInfo, QueueCreateInfo, QueueFamilyProperties, QueueFlags,
    },
    instance::InstanceCreateInfo,
    DeviceAddress, VulkanLibrary,
};
use winit::{
    event::{ElementState, Event, Modifiers, MouseButton, WindowEvent},
    event_loop::{self, EventLoop, EventLoopWindowTarget},
    keyboard::{Key, ModifiersState},
    platform::modifier_supplement::KeyEventExtModifierSupplement,
    window::{Window, WindowBuilder},
};


#[allow(unused_variables)]
#[allow(unused_assignments)]
fn winit_handle_window_events(event: WindowEvent, window_target: &EventLoopWindowTarget<()>) {
    let mut modifiers = ModifiersState::default();

    match event {
        WindowEvent::CloseRequested => {
            // println!("Close window requested, proceed? [y/n]");
            // let mut ans = [0; 1];
            // std::io::stdin().read_exact(&mut ans).unwrap();

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
            // println!("Keyboard event detected!");

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


#[allow(unused_variables)]
#[allow(unused_assignments)]
fn initialise_vulkan_runtime() -> Result<(), Error> {
    /*
       Step 1. Select a Physical Device
    */

    // Loading the vulkan plugins
    let vulkan_library = vulkano::VulkanLibrary::new()
        .unwrap_or_else(|err| panic!("Failed to load Vulkan: \n {:?}", err));

    // Creating a vulkan instance
    let vulkan_instance =
        vulkano::instance::Instance::new(vulkan_library, InstanceCreateInfo::default())
            .unwrap_or_else(|err| panic!("Failed to create Vulkan Instance \n {:?}", err));

    let mut logical_device: Arc<Device>;
    let mut queues: Vec<vulkano::device::Queue>;

    /*
    Creating Logical Device
    */
    // Listing the available physical devices supporting vulkan API
    for physical_device in vulkan_instance.enumerate_physical_devices().unwrap() {
        println!(
            "Device Available: \n {:?}",
            physical_device.properties().device_name
        );

        let mut device_selected = false;

        // Going through the available queues in the physical device
        for (family_index, family_properties) in
            physical_device.queue_family_properties().iter().enumerate()
        {
            if family_properties.queue_flags.contains(QueueFlags::GRAPHICS) {
                // This graphics card supports the graphics queue family.
                // Selecting the graphics card for rendering purposes

                // Creating the logical device for this physical device
                let (created_logical_device, created_queues) = Device::new(
                    physical_device,
                    DeviceCreateInfo {
                        queue_create_infos: vec![QueueCreateInfo {
                            queue_family_index: family_index as u32,
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                )
                .unwrap();

                logical_device = created_logical_device;
                device_selected = true;
                break;
            }
        }

        // Found the device, exiting the physical device search
        if device_selected {    
            println!("Device Selected");
            break;
        }

        // // Checking if this physical device supports the Graphics Queue Family
        // let family_index = physical_device.queue_family_properties().iter().enumerate().position(|(queue_family_index, queue_family_properties)| {
        //     queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        // }).unwrap() as u32;

        // // Graphics Queue family is supported. Selecting this physical device
        // if family_index >= 0 {

        // }

        // let (s_logical_device, s_queues) = Device::new(
        //     physical_device,
        //     DeviceCreateInfo {
        //         queue_create_infos: vec![QueueCreateInfo {
        //             queue_family_index: family_index,
        //             .. Default::default()
        //         }],
        //         .. Default::default()
        //     }
        // ).unwrap();
    }

    // Step 2. Create a Window

    let winit_event_loop: EventLoop<_> = event_loop::EventLoop::new().unwrap();
    let winit_window: Window = WindowBuilder::new().build(&winit_event_loop).unwrap();

    // setting the control flow for the event loop

    winit_event_loop.set_control_flow(event_loop::ControlFlow::Poll);
    let _ = winit_event_loop.run(move |app_event, elwt| match app_event {
        // Window based Events
        Event::WindowEvent { window_id, event } => {
            // Handling the window event if the event is bound to the owned window
            if window_id == winit_window.id() {
                winit_handle_window_events(event, elwt);
            }
        }

        // Custom user induced events (EventLoopProxy::send_event)
        Event::UserEvent(_event) => {}

        Event::LoopExiting => {
            println!("Exiting the event loop");
            exit(0);
        }

        // Event::DeviceEvent { device_id, event } => {
        //     println!("Device event");
        // },
        _ => (),
    });

    /*
        Steps to initialise vulkan instance

        Step 1. Query for the available physical devices based on the requirements
        Step 2. Select a physical device to be used
        Step 3. Create a Logical device based on the (VkQueue) queue types that we want to use
        Step 4. Initialise the window for the application (using winit)
        Step 5. Create a Vulkan Surface to render our graphics to which would be a reference to the current window
        Step 6. Create a SwapChain to render our images to. This swapchain will then swap the images from
    */

    // Err("Failed to initialise vulkan runtime")
    Ok(())
}

fn main() {
    println!("Hello, world!");
    let _ = initialise_vulkan_runtime();
}

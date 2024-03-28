use std::sync::Arc;

use winit::{event::{DeviceEvent, Event, WindowEvent}, event_loop::{self, EventLoop}, platform::x11::EventLoopBuilderExtX11, window::{Window, WindowBuilder}};

use crate::graphics::components::input_handler::{KeyboardInputHandler, MouseInputHandler};

// type RenderCall = impl FnMut() -> ();

/// Enclosing structure for managing affairs related to the application window
/// Responsible for creating window, handling all window events, updating input
/// state
pub struct WindowManager<FnRender, FnMoveHandle>
where
    FnRender: Fn() -> (),
    FnMoveHandle: Fn((f64, f64)) -> (),
{
    window: Arc<Window>,
    event_loop: EventLoop<()>,
    keys: KeyboardInputHandler,
    mouse: MouseInputHandler<FnMoveHandle>,
    render_loop: FnRender
}

impl<FnRender, FnMoveHandle> WindowManager<FnRender, FnMoveHandle>
where
    FnRender: Fn() -> (),
    FnMoveHandle: Fn((f64, f64)) -> ()
{
    pub fn create(render_closure: FnRender) -> WindowManager<FnRender, FnMoveHandle> {
        let (window, event_loop) = Self::create_window();
        
        WindowManager {
            window,
            event_loop,
            keys: KeyboardInputHandler::new(),
            mouse: MouseInputHandler::new(),
            render_loop: render_closure,
        }
    }

    pub fn start_event_loop(mut self) {
        self.event_loop.run(|event, elwt| {
            match event {

                // Keyboard handler
                Event::WindowEvent {  event: WindowEvent::KeyboardInput { device_id, event, is_synthetic }, ..} => {
                    self.keys.update_input(event);
                }

                // Mouse click
                Event::WindowEvent { event: WindowEvent::MouseInput { device_id, state, button }, .. } => {
                    self.mouse.update_mouse_click(button, state);
                }

                // Mouse move
                Event::DeviceEvent{ event: DeviceEvent::MouseMotion { delta }, .. } => {
                    self.mouse.update_mouse_move(delta);
                }

                // Render loop call
                Event::WindowEvent { event: WindowEvent::RedrawRequested , .. } => {
                    (self.render_loop)();
                }

                _ => (),
            };

        });

    }

}


impl<FnRender, FnMoveHandle> WindowManager<FnRender, FnMoveHandle> 
where 
    FnRender: Fn() -> (),
    FnMoveHandle: Fn((f64, f64)) -> ()
{   
    /* Creates a `winit::window` arc and an event loop associated with the window.

        Note: On ubuntu devices, the wayland window does not launch for some 
        incompatibility reasons. This incompatibility has existed in the wayland
        server for some time and has not been fixed.
        For this reason, the window created on linux based platforms will always
        be an X11 window.
    */
    fn create_window() -> (Arc<Window>, EventLoop<()>) {
        let mut winit_event_loop: Option<EventLoop<()>> = None;
        let mut event_loop_builder = event_loop::EventLoopBuilder::new();
        

        // Forcing x11 window
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
}
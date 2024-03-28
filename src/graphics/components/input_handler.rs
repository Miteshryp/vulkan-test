use std::collections::HashMap;

use winit::{
    event::{ElementState, KeyEvent, MouseButton, RawKeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};


/// Handles mouse event captures and updates mouse state.
/// 
/// Stores the state of the current mouse input. This is able to report 
/// multiple types of inputs simultaneously, unlike the winit event loop, where 
/// we can only process one at a time.
/// Sibling structs - [`KeyboardInputHandler`]
pub struct MouseInputHandler<MoveHandler>
where   
    MoveHandler: Fn((f64, f64)) -> ()
{
    pub pressed_buttons: HashMap<MouseButton, bool>,
    // pub move_delta: (i32, i32),

    pub move_triggers: Vec<MoveHandler>
}


/// Interface implementation for [`MouseInputHandler`]
impl<MoveHandler> MouseInputHandler<MoveHandler>
where   
    MoveHandler: Fn((f64, f64)) -> ()
{
    
    pub fn new() -> Self {
        MouseInputHandler {
            pressed_buttons: HashMap::new(),
            // move_delta: (0,0),
            move_triggers: vec![]
        }
    }

    pub fn update_mouse_click(&mut self, button: MouseButton, state: ElementState) {
        self.pressed_buttons.insert(button, match state {
            ElementState::Pressed => true,
            ElementState::Released => false
        });
    }

    pub fn update_mouse_move(&mut self, delta: (f64, f64)) {
        // self.move_delta = delta;
        for handle in self.move_triggers.iter() {
            (handle)(delta);
        }
    }

    // pub fn update_mouse_scroll(&mut self, ) {
        
    // }
}



/// Handles keyboard event captures and updates keyboard state.
/// 
/// Stores the state of the current keyboard input. This is able to report 
/// multiple keys simultaneously, unlike the winit event loop, where 
/// we can only process one at a time.
/// Sibling structs - [`MouseInputHandler`]
pub struct KeyboardInputHandler {
    pub pressed_keys: std::collections::HashMap<PhysicalKey, bool>,
}



/// Interface methods for [`KeyboardInputHandler`]
impl KeyboardInputHandler {
    pub fn new() -> Self {
        Self {
            pressed_keys: HashMap::new()
        }
    }


    /// For reporting a single update input from winit into the 
    /// Handler.
    /// 
    /// This method updates the state to add the current key press or release
    pub fn update_input(&mut self, key_event: KeyEvent) {
        match key_event.state {
            ElementState::Pressed => {
                self.pressed_keys.insert(key_event.physical_key, true);
            }

            ElementState::Released => {
                self.pressed_keys.insert(key_event.physical_key, false);
            }
        }
    }

    /// Clears all entries in the hashset
    /// 
    /// Useful if any event missed during window resize causes 
    /// incorrect behavior
    pub fn reset_inputs(&mut self) {
        self.pressed_keys.clear();
    }

    /// Detect if a given key is pressed on the keyboard based on the KeyCode
    /// 
    /// Note: This function does not account for modifiers along with 
    /// key press. 
    /// 
    /// For example, We cannot detect case for the pressed key from this function
    pub fn is_pressed(&self, key_code: KeyCode) -> bool {
        match self.pressed_keys.get(&PhysicalKey::Code(key_code)) {
            Some(is_pressed) => is_pressed.to_owned(),
            None => false,
        }   
    }
}

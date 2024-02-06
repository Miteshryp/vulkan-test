use std::collections::HashMap;

use winit::{
    event::{ElementState, KeyEvent, RawKeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

pub struct KeyboardInputHandler {
    pub pressed_keys: std::collections::HashMap<PhysicalKey, bool>,
}

impl KeyboardInputHandler {
    pub fn new() -> Self {
        Self {
            pressed_keys: HashMap::new()
        }
    }

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

    // Clears all entries in the hashset
    // This function is useful if any event missed during window resize causes incorrect behavior
    pub fn reset_inputs(&mut self) {
        self.pressed_keys.clear();
    }

    pub fn is_pressed(&self, key_code: KeyCode) -> bool {
        match self.pressed_keys.get(&PhysicalKey::Code(key_code)) {
            Some(is_pressed) => is_pressed.to_owned(),
            None => false,
        }   
    }
}

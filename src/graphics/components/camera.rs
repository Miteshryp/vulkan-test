use std::{ops::{Deref, DerefMut}, sync::Arc};

use nalgebra_glm as glm;

#[derive(Clone)]
pub struct Camera {

    // view components:
    position: glm::TVec3<f32>,
    camera_front: glm::TVec3<f32>,
    camera_right: glm::TVec3<f32>,
    camera_up: glm::TVec3<f32>,

    pitch: f32,
    yaw: f32,

    // projection components:
    fov: f32,
    aspect_ratio: f32,
    near_z: f32,
    far_z: f32,
}

impl Camera {
    pub fn new(
        position: glm::TVec3<f32>,
        look_at: glm::TVec3<f32>,
        up: glm::TVec3<f32>,
        fov: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {

        let front = (look_at - position).normalize();
        let right = front.cross(&up).normalize();

        Self {
            position,
            camera_front: front,
            camera_right: right,
            camera_up: up.normalize(),
            fov,
            aspect_ratio,
            near_z: near,
            far_z: far,
            pitch: f32::to_radians(0.0),
            yaw: f32::to_radians(-90.0),
            // projection_descriptor: None,
        }
    }

    pub fn move_forward(&mut self, units: f32) {
        self.position += self.camera_front * units;
    }

    pub fn move_backward(&mut self, units: f32) {
        self.position -= self.camera_front * units;
    }

    pub fn move_left(&mut self, units: f32) {
        self.position -= self.camera_right * units;
    }

    pub fn move_right(&mut self, units: f32) {
        self.position += self.camera_right * units;
    }


    pub fn rotate(&mut self, yaw:f32, pitch: f32) {
        self.yaw += f32::to_radians(yaw);
        self.pitch += f32::to_radians(pitch);

        self.pitch = self.pitch.max(f32::to_radians(-89.9)).min(f32::to_radians(89.9));
        
        self.update_vectors();
    }



    fn update_vectors(&mut self) {
        let mut new_front = glm::vec3(0.0,0.0,0.0);
        
        new_front.x = f32::cos(self.yaw) * f32::cos(self.pitch);
        new_front.y = f32::sin(self.pitch);
        new_front.z = f32::sin(self.yaw) * f32::cos(self.pitch);

        self.camera_front = new_front.normalize();
        self.camera_right = self.camera_front.cross(&self.camera_up).normalize();
        
        // To enable comfortable free roam, we let the up vector always point to the world up
        // self.camera_up = self.camera_right.cross(&self.camera_front).normalize();
    }


    // pub fn get_translate(&self) -> Vec3 {
    // }

    // pub fn set_translate(&mut self, position: glm::Vec3) {
    // }

    pub fn get_view_matrix_data(&self) -> glm::TMat4<f32> {

        let look_at_vec = self.position + self.camera_front;

        glm::look_at(
            &self.position,
            &look_at_vec,
            &self.camera_up
        )
    }

    pub fn update_aspect_ratio(&mut self, width: u32, height: u32) {
        self.aspect_ratio = (width as f32) / height as f32;
    }

    pub fn get_projection_matrix_data(&self) -> glm::TMat4<f32> {
        glm::perspective(self.aspect_ratio, self.fov, self.near_z, self.far_z)
    }
}
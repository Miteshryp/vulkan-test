use std::sync::Arc;

use nalgebra_glm as glm;
use vulkano::descriptor_set::PersistentDescriptorSet;

// use crate::graphics_pack::buffers::UniformSet;

// TODO: Camera manager is required if I just store the parameters of the
// projection matrix in the camera struct.
// Storing the entire matrix in the struct can be pointless since we only
// use the parameters once to create the matrix, and once the descriptor set
// uniform for the projection matrix has been created, there is no use of the
// matrix untill the projection parameters change, which is rare, so the RAM
// memory is being wasted.

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

    // The projection descriptor set.
    // projection_descriptor: Option<Arc<PersistentDescriptorSet>>,
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
        
        self.update_vectors();
    }



    fn update_vectors(&mut self) {
        let mut new_front = glm::vec3(0.0,0.0,0.0);
        
        new_front.x = f32::cos(self.yaw) * f32::cos(self.pitch);
        new_front.y = f32::sin(self.pitch);
        new_front.z = f32::sin(self.yaw) * f32::cos(self.pitch);

        self.camera_front = new_front.normalize();
        self.camera_right = self.camera_front.cross(&self.camera_up).normalize();
        // self.camera_up = self.camera_right.cross(&self.camera_front).normalize();
    }


    // pub fn get_translate(&self) -> Vec3 {
    // }

    // pub fn set_translate(&mut self, position: glm::Vec3) {
    // }

    pub fn get_view_matrix_data(&mut self) -> glm::TMat4<f32> {

        let look_at_vec = self.position + self.camera_front;
        println!("{}", look_at_vec);

        glm::look_at(
            &self.position,
            &look_at_vec,
            &self.camera_up
        )
    }

    pub fn get_projection_matrix_data(&self) -> glm::TMat4<f32> {
        glm::perspective(self.aspect_ratio, self.fov, self.near_z, self.far_z)
    }
}
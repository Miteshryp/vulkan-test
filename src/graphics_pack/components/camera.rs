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
    // projection: glm::TMat4<f32>,
    // view: glm::TMat4<f32>,

    // view components:
    position: glm::Vec3,
    look_at: glm::Vec3,
    up: glm::Vec3,

    pitch: f32,
    yaw: f32,

    // projection components:
    fov: f32,
    aspect_ratio: f32,
    near_z: f32,
    far_z: f32,

    // The projection descriptor set.
    projection_descriptor: Option<Arc<PersistentDescriptorSet>>,
}

impl Camera {
    pub fn new(
        position: glm::Vec3,
        look_at: glm::Vec3,
        up: glm::Vec3,
        fov: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            look_at,
            up,
            fov,
            aspect_ratio,
            near_z: near,
            far_z: far,
            pitch: 0.0,
            yaw: 0.0,
            projection_descriptor: None,
        }
    }

    pub fn move_forward(&mut self, units: f32) {
        let forward = glm::normalize(&(self.look_at - self.position));
        self.position -= forward;
    }

    pub fn move_backward(&mut self, units: f32) {
        let forward = glm::normalize(&(self.look_at - self.position));
        self.position += forward;
    }

    pub fn move_left(&mut self, units: f32) {
        // Steps
        // 1. Find the vector perp to (look_at - position) and up
        // 2. This vector obtained should point left
        // 3. Normalize the vector obtained
        // 4. Multiply the vector obtained and add it to the position
        // 5. The camera should have move left.

        let mut left_vec = glm::cross(&(self.look_at - self.position), &self.up);
        left_vec = glm::normalize(&left_vec);

        self.position += left_vec * units;
    }

    pub fn move_right(&mut self, units: f32) {
        // Steps
        // 1. Find the vector perp to (look_at - position) and up
        // 2. This vector obtained should point left
        // 3. Normalize the vector obtained
        // 4. Multiply the vector with magnitude and add its negation to the position
        // 5. The camera should have move right.

        let mut left_vec = glm::cross(&(self.look_at - self.position), &self.up);
        left_vec = glm::normalize(&left_vec);

        self.position -= left_vec * units;
    }

    pub fn get_translate(&self) -> glm::Vec3 {
        self.position
    }

    pub fn set_translate(&mut self, position: glm::Vec3) {
        self.position = position;
    }

    pub fn get_view_matrix_data(&self) -> glm::TMat4<f32> {
        glm::look_at(&self.position, &self.look_at, &self.up)
    }
}

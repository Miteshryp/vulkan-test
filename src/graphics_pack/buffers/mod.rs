pub mod primitives;

pub mod base_buffer;
pub mod vertex_buffer;
pub mod index_buffer;
pub mod uniform_buffer;

// pub use base_buffer::{BufferOptions, Buffer;

pub use vertex_buffer::{*};
pub use index_buffer::{*};
pub use uniform_buffer::{*};

pub use self::primitives::VertexPoint;


// TODO: Implement staging buffers for vertex and index buffers

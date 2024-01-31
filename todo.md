
# TODAY's GOALS

[x] Restructure the entire codebase into appropriate modules (Staging and Rendering buffers)
[x] Find a way to solve the buffer interface problem


[] texture array rendering (Similar to texture rendering)

### VULKAN WEBSITE STEPS TO CREATE A TEXTURE IMAGE
- Create an image object backed by device memory
- Fill it with pixels from an image file
- Create an image sampler
- Add a combined image sampler descriptor to sample colors from the texture

### QUESTIONS
1. Can we modify the contents of an image after binding it to an image view object?

### LOW PRIORITY
[x] Solve the global store problem in rust. - Will be solved using plugins (like in nest.js)

[] Find a solution to repeat key detection lag in winit - 
    Might have to build a custom solution with hashmap structure.
    The DeviceEvent Capture only captures the key press or release event
    This can be used to trigger the status of a button which can then be queried through a plugin in the engine.



# FUTURE GOALS
[] Multipass / Deffered rendering
[] Different types of lighting

[] Indirect drawing
[] Dynamic shader loading



# MISC GOALS
[] Learn about writing macros in rust
[] Explore the vulkan and SPIR-V sdk tools
[] Explore NVRHI

# WORKAROUNDS
- [] Renderdoc integration for shader debugging - FAILED (binding not working with renderdoc 1.30)


## RESOURCES
- Rust User guide for rendering using vulkano - https://taidaesal.github.io/vulkano_tutorial/
- Vulkano official guide - https://vulkano.rs/
- Vulkano code examples - https://github.com/vulkano-rs/vulkano/tree/master/examples 
- Vulkan official tutorial (C based) - https://vulkan-tutorial.com/
- Rust version of official vulkan tutorial - https://github.com/bwasty/vulkan-tutorial-rs/
- Vulkan guide - https://vkguide.dev
- Descriptor sets guide - https://vkguide.dev/docs/chapter-4/descriptors/#:~:text=Binding%20descriptors&text=We%20will%20be%20doing%20it,slot%2C%20there%20will%20be%20errors.

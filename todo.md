
# TODAY's GOALS
[x] Texture rendering in shaders


Steps for texture rendering
1. [x] Load the image data into the memory buffer - load_png_image function
2. [x] Create and store the data into a vulkan source buffer for the image data - load_png_image function
3. [x] Create an Vulkan image object
4. [x] Transfer the image data into the image object
5. [x] Bind the image object to an image view
6. [x] Bind the image view to the texture2D object in the descriptor set
7. [x] Create a Sampler object and pass it in as a sampler parameter in the descriptor set
8. [x] Pass the descriptor writes into the command buffer
9. [x] Create a new vertex attribute taking the texture coordinate for the mapping
10. [x] Use the texture coordinate attribute to sample the texture onto the point in the fragment shader (Sampler coords are btw 0-1)

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

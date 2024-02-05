
# TODAY's GOALS
[x] Multipass / Deffered rendering
    - [x] Find a solution to image view binding requirement for attachment image buffers

    - [x] Modularize the code by creating "Renderers"       

    - [x] Find out why the formatting change is not working in render subpass attachments - Stupid mistake (I passed UINT instead of a UNORM)

[] Document learnings from Deferred rendering solution in the notes.md file

[] Explore interfacing common renderer functions
[] Create a input manager class, which could have callbacks attached to each input
[] Learn how to implement a cubemap
[] Model loading







### LOW PRIORITY
[x] Solve the global store problem in rust. - Will be solved using plugins (like in nest.js)

[] Modify the shader to take in the rotation axis and angle of rotation, translation and scale, then construct the model matrix in the shader itself.

[] Modify the load_image function to return the format of the image along with the data, which could then be used to create images of appropriate format for the image data. (This could be used to load images other than 8bit channel format)


# FUTURE GOALS
[] Different types of lighting
[] Indirect drawing
[] Dynamic shader loading



# MISC GOALS
[] Learn about writing macros in rust
[] Explore the vulkan and SPIR-V sdk tools
[] Explore NVRHI


## RESOURCES
- Rust User guide for rendering using vulkano - https://taidaesal.github.io/vulkano_tutorial/
- Vulkano official guide - https://vulkano.rs/
- Vulkano code examples - https://github.com/vulkano-rs/vulkano/tree/master/examples 
- Vulkan official tutorial (C based) - https://vulkan-tutorial.com/
- Rust version of official vulkan tutorial - https://github.com/bwasty/vulkan-tutorial-rs/
- Vulkan guide - https://vkguide.dev
- Descriptor sets guide - https://vkguide.dev/docs/chapter-4/descriptors/#:~:text=Binding%20descriptors&text=We%20will%20be%20doing%20it,slot%2C%20there%20will%20be%20errors.

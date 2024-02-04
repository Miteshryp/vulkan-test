
# TODAY's GOALS
[x] Multipass / Deffered rendering
    - [x] Find a solution to image view binding requirement for attachment image buffers

    - [] Modularize the code by creating "Renderers"
        - [] Store the push descriptor data in the renderer and create functions to pass the appropriate push descriptor data
        - [] Write functions to pass the correct attachment in the descriptor set write assigned for intermidiate attachments.        

    - [x] Find out why the formatting change is not working in render subpass attachments - Stupid mistake (I passed UINT instead of a UNORM)

[] Modify the load_image function to return the format of the image along with the data, which could then be used to create images of appropriate format for the image data. (This could be used to load images other than 8bit channel format)


# SHORT NOTES (Will remove later)

- All attachment specific to the render pass used by the renderer are stored in the renderer now. Any attachment required can be fetched in the renderer struct implementation.

- Renderer will explicitly bind the required attachment to the corresponding subpass pipeline shader.

- Renderer will also explicitly store data that is being passed in the push descriptor of the pipeline shader. The renderer is then responsible for writing the appropriate data to the appropriate subpass pipeline shader (to prevent redundant writes and crashes due to missing input binding format since the shader might not declare or use the descriptor set)




### LOW PRIORITY
[x] Solve the global store problem in rust. - Will be solved using plugins (like in nest.js)



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

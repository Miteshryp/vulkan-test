
# TODAY's GOALS
[x] Multipass / Deffered rendering
    - [x] Find a solution to image view binding requirement for attachment image buffers

    - [] Modularize the code by creating "Renderers"
        - [] Store the push descriptor data in the renderer and create functions to pass the appropriate push descriptor data
        - [] Write functions to pass the correct attachment in the descriptor set write assigned for intermidiate attachments.        

    - [x] Find out why the formatting change is not working in render subpass attachments - Stupid mistake (I passed UINT instead of a UNORM)

[x] Fix the unknown issue with texture loading (new_sample2 is not loading properly, and further more its messing up )

# SHORT NOTES (Will remove later)
- Samplers uniforms (or rather any other uniforms which do not change for a given pipeline) should reside inside Renderer, since the Renderer can automatically bind the uniforms while creating command buffers




### LOW PRIORITY
[x] Solve the global store problem in rust. - Will be solved using plugins (like in nest.js)

[x] Find a solution to repeat key detection lag in winit - 
    Might have to build a custom solution with hashmap structure.
    The DeviceEvent Capture only captures the key press or release event
    This can be used to trigger the status of a button which can then be queried through a plugin in the engine.



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

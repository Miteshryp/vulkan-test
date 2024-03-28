
# PROBLEM ZONES
1. Call to the render function in the main function where we access the renderer from the instance object. I want to pass instance as a reference while calling a method on the renderer object which resides inside the instance object.

2. Creating the instance: The renderer might want to have an internal plugin of the parent instance which could be stored as a reference. 
This reference could then be used to access any immutable member in the instance object (which makes sense because the instance object is kind of like the global store for vulkan specific objects.)

3. Seeing the bevy code, I realised that there is a used of a ECS system in design, which includes the design of internal components. While examining this, I found that the plugins are added by just specifying type rather than passing the object.
This is a pattern which I want to replicate, but there seems to be a lot going on along with that system (Schedules, World-Subapp interaction, etc).

## Potential solution to the design problem
- We can create a get_renderer method, which can take a type as a generic argument.
- Every type in rust has a unique identity, which can be fetched using `TypeId::of::<Type>` expression.
- The renderers can be constructed in the instance on addition of on such renderer type through a method `add_renderer` in the instance. 
- This `add_renderer` method will require us to pass in a initialization method for the renderer. (We will need to call the new method of the renderer). It's potential solutions are
    - We can create a generic `new` method for all renderers in the renderer interface (For vulkan)
    - We can take in some type of builder (like Schedules in bevy) which get executed before construction. This routine can take in certain parameters (NEED to refine this approach, very vague right now)




# TODAY's GOALS


[x] Read Bevy Source code for solution to the below problem
[x] Explore interfacing common renderer functions (Designing the API for the engine) - Struggling heavily with this. 
    - The code design should have high decoupling between functionality and data
    - The VulkanInstance is state of the vulkan program, whereas renderer is a function which takes the instance as an input
    - Hence, there is no reason that the Renderer should be part of the Vulkan Instance.

[] Learn how to implement a cubemap
[] Model loading

[] Properly understand in-flight frames
[] Document code

# OPTIMIZATIONS
[x] In flight frames - 
[] Investigate why in-flight frames didn't give huge performance boost. 
    - Might be something to do with vsync



# FUTURE GOALS
[] Different types of lighting
[] Indirect drawing
[] Dynamic shader loading


### LOW PRIORITY
[x] Solve the global store problem in rust. - Will be solved using plugins (like in nest.js)

[] Modify the shader to take in the rotation axis and angle of rotation, translation and scale, then construct the model matrix in the shader itself.

[] Modify the load_image function to return the format of the image along with the data, which could then be used to create images of appropriate format for the image data. (This could be used to load images other than 8bit channel format)





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

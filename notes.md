
# DESCRIPTOR SETS 

Vulkan allows us to use push descriptors to send uniform data into the shader programs.
Unlike OpenGL, which used to require us to bind each variable using a single bind call to openGL,
What vulkan allows us to do is that it allows us to bind a group of data or variables into a single 
set, which are called descriptor sets.

Each binding in a descriptor set can represent a single data variable. Note that this variable can be
a custom defined structure which could potentially contain multiple elements, but in terms of binding calls,
This can be called a single variable.
So a single binding in a set is a single variable and a set can contain multiple bindings, pretty simple right?

Well, not really. See the thing is, each device only supports a limited number of sets based on the vulkan version
the device supports. In most cases (as recommended by vulkan), we can and should only consider using 4 descriptor 
sets since devices might not support sets more than that.

That being said, each descriptor set should also be used for different purposes.
Lower indexed descriptor sets should be used for uniforms with higher updation frequency as compared to the sets 
with a lower index. For this purpose, vulkan has the following types of descriptors

### 1. Push Descriptors
Sets which have a higher updation frequency should be updated using a **push descriptor**
A push descriptor can be used to bind data into an index at the time of building a command buffer rather than
having to create a descriptor and then bind the data set to the command buffer. This enables us to perform 
incremental updates to the uniforms without managing the lifecycle of the descriptor set data on the GPU. 
(Also, vulkano doesn't seem to have support for descriptor sets other than PersistentDescriptorSets, which are
basically fixed storage on the GPU)



### 2. Persistent Descriptors
A persistent descriptor set is a descriptor whose lifecycle is manually managed by us and which can be bound to a 
command buffer before executing the command buffer. This persistent descriptor set calls the vulkan call Vk 







## Questions
### 1. When to push constants vs push descriptor
 
Push constants only allow us to pass small ammounts of data into the shaders, but in an extremely fast manner (usually the size of the push constant buffer can be 128 bytes), so we should utilise this memory buffer whenever possible.
However push constants have some limitations. They cannot be used to pass data arrays and cannot be used for texture based data transfer due to size limitiations.
In such cases where large data needs to be frequently updated, we should instead 
use push descriptors as they are bound directly to the command buffer
instead of being seperately bound to a decscriptor sets

If some data is not changed very frequently, it is a good idea to 
place it in a persistent descriptor set.

### 2. What is the relevance of set index in terms of updation frequency?
This order of relevance is defined by us and not a in build thing in 
vulkan (I think so at least). We as developers can define which
descriptor set index is going to be a push descriptor and which is gonna
be a persistent descriptor.






# TEXTURE RENDERING

Textures can be mapped onto a geometry in vulkan similar to OpenGL. 
OpenGL dealt with textures with a texture image and a sampler, and vulkan is no different, but some steps do differ to make the process of data transfer more efficient. 
Vulkan allows us flexibility in binding texture data only once to a image view, which can then be written each frame without having to copy over the entire buffer data into the device image object (object storing the image data on the GPU) every frame, only the image view.
Here we are optimizing the data transfers by only passing image data again to the GPU when it is necessary to do so.

Following are the steps to bind a texture image in vulkan:

1. Create a sampler object with appropriate parameters.
2. Configure the push descriptor for texture data
    - Get the descriptor set for texture data from the set_layout
    - Add the push descriptor flag to this set layout
    - Designate a specific binding index as a immutable sampler field. 
    - Pass in the vec of samplers into this
3. Load the image data from the image file
4. Create a buffer object and load the image data into the buffer
5. Create an image object and bind it to an image view object. This image view object is then used to pass a push descriptor write into the texture parameter in the shader
6. Use the command buffer to copy the buffer data into the image if needed.
7. Use the image view object bound to the image to pass a push descriptor write in the drawing command buffer.
8. Configure the shader to take in the sampler and texture in a binding
9. Pass in the texture coords into the shaders.
10. Use the texture, sampler and tex_coord to render the image color in the fragment shader as follows:

        color = texture(sampler2D(texture_data, sampler_object), tex_coord)
       

However, one thing to note here is that the texture or sampler2D object in the shaders cannot be passed into a single binding. The texture or textureArray object and sampler both cannot be placed inside a block and must be a seperate binding in a shader

## Questions
1. How do we pass in an array of textures - Using textureArray instead of texture object. It requires some different configurations, which I am yet to explore (Refer the texture array example in the docs examples)

2. Can we modify the contents of an image after binding it to an image view object?
Guess - We should be able to do this since an image is just a location in memory, which is bound to a image view object. So image view should just contain the pointer to the image data. Us modifying the image data should not have any difference in the image view operation.



# DEFERRED RENDERING 

This is a technique used in 3D based graphics where the output of 
a single graphics pipeline is not immediately presented to the screen
but rather is given as an input into another render pass to get a 
new buffer output, which could then be used to render onto the screen 
or can be given as an input in another render pass.

To process a renderpass in a deferred manner, we must focus on the following components of the renderer:

1. **Preparing the Renderpass**
    - Define Renderpass, its attachments, and subpasses
    - Connect Subpass inputs, colors and depth stencil to appropriate attachments
    - Create the attachment buffers as Images, and create an access point through image views. The format for each attachment image must match the format defined for that attachment in the renderpass.
    - Attach the attachment image view into the framebuffer object that we create. The attachment buffers are a part of the frame, and hence the output of the shaders might be written to one of these buffers (based on ```color``` field definition of the subpass in render pass definition).
    ```
    NOTE: In case of a single pass, we used to attach the image of the swapchain which was to be displayed to the window in this framebuffer. Now, for multiple passes, we have the ability to bind multiple buffers into the framebuffer object. This buffer is allocated for every frame that we render.
    ```
    - Define the input attachment for the subpass if the subpass uses the output of a previous subpass. Subpass can write output to attachments which can then be used by a following subpass by taking that attachment as an input. To define the inputs in the subpass, we define the order of attachments in the `input` field of the subpass definition.



2. **Prepare the Shaders for each Subpass**
    - Write the shader with the required input. Note that the fragment shader must write the output to a defined attachment. The `layout` index of the fragment output variable is determined by the order attachments in the `color` field that we defined in the subpass definition while creating the renderpass.
    - Define the descriptor set in which the attachment will be taken as an input. This descriptor set will be used to bind and supply the attachment buffers in the shader to be used. This attachment will contain the output from the previous subpass which are to be used in the current subpass shaders

3. **Create the pipelines for each Subpass**
    - Create a subpass object for the pipeline using the render pass and the index of the current subpass in the renderpass.
    - Load the appropriate shader module. The input definitions in the shaders are important since they let us define push descriptors and contain descriptor set layouts required to create the pipeline.
    - Define the `color_blend_state` field while creating the GraphicsPipeline since it is important to define the number of color attachments that the subpass is writing to.
    - Define the `vertex_input_state` for the pipeline by taking the vertex input state definition from the vertex shader info
    ```
    NOTE: The vertex input state format has to be same for all the pipelines in a single subpass, since the fragment shader only runs for the fragments covered by the rasterizer, which covers the fragments only if they lie in the defined vertex bounds.
    ```
4. **Attach the Required buffer and attachments descriptor sets in Command Buffer**
    - Set the clear values for attachments at the beginning of the render pass (The format for clear values depends on the format defined for the attachments while creating the renderpass and while creating the attachment images).
    - Create a command buffer by binding the buffers in appropriate order (the vertex buffers must be in the format defined in `vertex_input_state` which we defined while creating the pipelines)
    - Create descriptor sets to pass in the appropriate data into the shaders. The frame buffer attachment which are to be used be used in the fragment shader must also be attached in this step. 
    - It is important to ensure that the attachment resources bound to the descriptor set are the same ones that were bound to the Framebuffer. Recreating the resources could lead to data being written to wrong buffers (buffers not known by the framebuffers, hence the next subpass will load data from the resource in the framebuffer, but since the last renderpass wrote to some other resource, the resource in the renderpass is still containing the clear value)
    - Once all the attachments have been supplied to the shader in descriptor sets, we can render the render pass to get the desired results.



### Overview model:

- Subpass 1
    - Graphics pipeline functions
        |
        output 1
- Subpass 2
        input1 = output 1
        |
    - Graphics pipeline functions
        |
        output 2

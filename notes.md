
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


Overview model:

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

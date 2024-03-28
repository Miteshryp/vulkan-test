
## Exploring the problem encountered in the vulkan-engine
ECS framework defined should help us to get better modularity in the project.
One of the current problems that I was facing was not able to attach a Renderer to the VulkanInstance after constructing the instance. The reason that was happening was that the renderer required references to data in the vulkan instance struct, since vulkan instance contained several global parameters necessary for all rendering processes (such as swapchain images, framebuffer indexes, logical device, etc).

This is because we cannot really store a reference to the vulkan instance object in the renderer because we cannot define a lifetime for it.
Furthermore, we are also not able to pass the entire instance object reference in the render() function call because we are storing the vulkan renderer inside the vulkan instance, and calling a method on the renderer partially borrows the instance object, hence the instance object cannot really be passed into the renderer. Hence, we run into this cyclic issue which is difficult to solve.

The problem in this scenario is high coupling of data and logic. If we take a step back and think about the goal of each component, we can simplify the components as  `VulkanInstance` containing the state of the vulkan instance, and the `DeferredRenderer` being a logical component which provides output as a function of an input which is this vulkan state. 
However, rust borrow checker forces us to design programs with modularity, which we are not doing by enclosing the renderer in the vulkan state itself.

Hence to solve this, we can take advantage of the ECS system that we have created to generalize this plugin pattern for all coupling problems. 
In this case, we can think of both state (VulkanInstance) and renderer and the renderer as 2 different components (actually resource because we only have single instance of the 2 things, but we'll implement that in the future in the ECS framework) where renderer is updated by taking this vulkan state as an input.

The problem that is gonna arise is that some changes in the vulkan state might need to trigger a change to resource. In this sense, these are dependent resource, hence we actually need an event system in the ECS system, which can automatically launch handlers for dependent resources.
If Resource B is dependent on Resource A, then any change to resource A should launch a handler on Resource B, which can change resource B as it sees fit by looking at the Resource A in the system. 
(This will have to be implemented using the event system. An event can have type arguments which defines anchor and dependent resources. Any change to the anchor resource should hence push an event with that resource as an anchor, and then the event should be processed with the anchor and dependent resource as parameter to the handler) 

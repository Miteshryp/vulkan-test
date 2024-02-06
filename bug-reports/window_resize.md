# BUG REPORT

### Cause
The bug is caused by continuous resizing of the window.

### Effect


### Debugging Analysis
1. Tried to refresh all appropriate parameters
    - Renderer pipelines
    - Swapchain 
    - render passes
    - surface

2. Renderdoc shows that the buffer being passed to the shader has some data in the instance which should not be there. This indicates the corruption of data being passed into the renderer. But while inspecting the parameters on the host side,
    every thing seems to be normal, i.e.:
    1. The size for host buffer seems to be correct 
        - vertex buffer - 1056 (same as in normal)
        - instance buffer - 40 (same as in normal)
        - index buffer - 144 (same as in normal)
    2. The size of the device buffer seems to be correct
        - vertex buffer - 46464 (same as in normal)
        - instance buffer - 800 (same as in normal)
        - index buffer - 576 (same as in normal)
    3. Renderdoc inspection reveals that there are unwanted instances being picked up by the pipeline.
    
    ![sdfsdf](./assets/window_resize_renderdoc.mkv)
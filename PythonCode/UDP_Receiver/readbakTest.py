from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties, GraphicsPipe, GraphicsOutput, Texture, OrthographicLens
import panda3d.core as p3d
from direct.filter.FilterManager import FilterManager
import numpy as np

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Configure off-screen buffer
        fb_props = FrameBufferProperties()
        fb_props.set_rgba_bits(8, 8, 8, 8)
        fb_props.set_depth_bits(24)
        winprops = p3d.WindowProperties()
        winprops.setSize(1024, 1024)
        self.win.requestProperties(winprops)
        graphics_pipe = self.pipe
        buffer = self.graphics_engine.make_output(
            graphics_pipe,
            "offscreen buffer",
            -1,
            fb_props,
            winprops,
            GraphicsPipe.BF_refuse_window,
            self.win.get_gsg(),
            self.win
        )
        #buffer = self.win.makeTextureBuffer("offscreen buffer2", 1024, 1024, to_ram=True, fbp=fb_props)

        # Create a render target texture
        render_target = Texture()
        render_target.setClearColor(p3d.Vec4(0, 0, 0, 0))
        render_target.clear()
        render_target.set_format(Texture.F_rgba)
        render_target.set_component_type(Texture.T_unsigned_byte)
        buffer.add_render_texture(
            render_target,
            GraphicsOutput.RTMBindOrCopy | GraphicsOutput.RTM_copy_ram,
            GraphicsOutput.RTPColor
        )
        # GraphicsOutput.RTMBindOrCopy | GraphicsOutput.RTM_copy_ram,
        buffer.setSort(0)

        # Set up the off-screen camera
        myScene = p3d.NodePath("fgRender")
        #offscreen_cam = self.make_camera(buffer)
        offscreen_cam = self.makeCamera(buffer, scene=myScene, lens=self.cam.node().getLens())
        #offscreen_cam.node().set_lens(OrthographicLens())
        #offscreen_cam.node().get_lens().set_film_size(10, 10)
        #offscreen_cam.node().get_lens().set_near_far(1, 1000)
        #offscreen_cam.set_pos(0, 0, 15)
        #offscreen_cam.reparent_to(self.cam)
        #offscreen_cam.node().set_scene(myScene)
        #offscreen_cam.node().set_active(False)
        #cm = p3d.CardMaker("card")
        #cm.setFrameFullscreenQuad()
        #card = self.render2d.attachNewNode(cm.generate())
        #card.setTexture(render_target)
        
        # Load a model and add it to the scene
        #model = self.loader.load_model("path/to/your/model")
        model = self.loader.loadModel('zup-axis')
        model.reparent_to(myScene)

        # Read back the texture data
        #self.graphics_engine.render_frame()
        #tex_data = render_target.get_ram_image_as("RGB")
        #print(tex_data)
        
        manager = FilterManager(self.win, self.cam)
        #tex = p3d.Texture()
        #tex.setup2dTexture(width, height, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
        self.quad = manager.renderSceneInto(colortex=None) # make dummy texture... for post processing...
        #mySvm.quad = manager.renderQuadInto(colortex=tex)
        self.quad.setShader(p3d.Shader.load(
                        p3d.Shader.SL_GLSL, vertex="post1_vs.glsl", fragment="post1_ps.glsl"))
        self.quad.setShaderInput("texPass0", buffer.getTexture(0))
        #self.quad.setShaderInput("texPass1", None)
        
        self.taskMgr.add(self.updatePreProcessShader,
                         "updatePreProcessShader")   
        
        # Manually render the scene using the offscreen camera
        #self.graphics_engine.render_frame()

        # Display the rendered result on the screen
        #cm = p3d.CardMaker('card')
        #cm.set_frame(-1, 1, -1, 1)
        #screen_card =self.render2d.attach_new_node(cm.generate())
        #screen_card.set_texture(render_target)

        self.render_target2 = Texture()
        self.render_target2.setClearColor(p3d.Vec4(1, 0, 0, 1))
        self.render_target2.clear()
        self.render_target2.set_format(Texture.F_rgba)
        self.render_target2.set_component_type(Texture.T_unsigned_byte)
        self.render_target2.setup2dTexture(1024, 1024, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
        
        self.buffer1 = buffer
        self.buffer1.set_active(False)
        
        # Create a TexturePeeker for asynchronous texture transfer
        self.texture_peeker = None
        self.peeker_ready = False
        
    def updatePreProcessShader(self, task):
        time = task.time
        self.buffer1.set_active(True)
        self.graphics_engine.render_frame()
        
        array = np.full((1024, 1024, 4), [0, 255, 255, 255], dtype=np.uint8)
        # Get the render target texture
        render_target = self.buffer1.get_texture(0)
        
        # Read the texture data into a PTAUchar
        tex_data = render_target.get_ram_image()
        
        if not tex_data.is_null():
            # Convert the PTAUchar to a NumPy array
            np_texture = np.frombuffer(tex_data, np.uint8)
            # Reshape the NumPy array to match the texture dimensions and number of channels
            np_texture = np_texture.reshape((render_target.get_y_size(), render_target.get_x_size(), 4))
            # Find the elements that match the condition (0, 0, 0, 0)
            condition = (np_texture == [105, 105, 105, 0]).all(axis=-1)
            # Update the elements that match the condition to (1, 1, 0, 1)
            array = np_texture.copy()
            array[condition] = [255, 0, 0, 255]
            
            # Process the texture data in the NumPy array
            self.render_target2.setRamImage(array)
            self.quad.setShaderInput("texPass1", self.render_target2)
        
        
        # Convert the PTAUchar to a NumPy array
        np_texture = np.frombuffer(tex_data, np.uint8)
        # Reshape the NumPy array to match the texture dimensions and number of channels
        np_texture = np_texture.reshape((render_target.get_y_size(), render_target.get_x_size(), 4))
        #print(np_texture.shape)
        
        #self.quad.setShaderInput("texPass1", render_target)
        self.buffer1.set_active(False)
        return task.cont

app = MyApp()
app.run()
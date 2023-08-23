import json
import os

import cv2 as cv
import numpy as np
import panda3d.core as p3d
from direct.filter.FilterManager import FilterManager
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Shader

with open("./data/source_video/Calibration_Simul.json") as f:
    calibration = json.load(f)

boat_breadth = calibration["boat_type"]["custom_breadth"] * 100
boat_length = calibration["boat_type"]["custom_length"] * 100

extrinsic_parameter = calibration["extrinsic_parameter"]["cameras"]
del extrinsic_parameter[3:5]
extrinsic_parameter[2], extrinsic_parameter[3] = extrinsic_parameter[3], extrinsic_parameter[2]
for extrinsic in extrinsic_parameter:
    extrinsic["location"]["translation_x"] *= 100
    extrinsic["location"]["translation_y"] *= 100
    extrinsic["location"]["translation_z"] *= 100

K1 = 1.281584985127447
K2 = 0.170043067138006
K3 = -0.023341058557079
K4 = 0.007690791651144
K5 = -0.001380968639013

img_width = 1920
img_height = 1080
fx = 345.12136354806347
fy = 346.09009197978003
cx = 959.5
cy = 539.5
skew_c = 0

winSizeX = 1024
winSizeY = 1024


class SurroundView(ShowBase):
    def __init__(self):
        super().__init__()

        winprops = p3d.WindowProperties()
        winprops.setSize(winSizeX, winSizeX)
        self.win.requestProperties(winprops)

        self.render.setAntialias(p3d.AntialiasAttrib.MAuto)
        self.cam.setPos(0, 0, -4000)
        self.cam.lookAt(p3d.LPoint3f(0, 0, 0), p3d.LVector3f(0, -1, 0))

        self.renderObj = p3d.NodePath("fgRender")
        self.renderSVM = p3d.NodePath("bgRender")

        # Set up the offscreen buffer
        fb_props = p3d.FrameBufferProperties()
        # fb_props.setRgbColor(True)
        fb_props.set_rgba_bits(32, 32, 32, 32)
        fb_props.setAuxRgba(2)  # Set up one auxiliary buffer
        fb_props.setDepthBits(24)

        # self.buffer1 = self.win.makeTextureBuffer("buffer1", winSizeX, winSizeY, to_ram=True, fbp=fb_props)
        self.buffer1 = self.graphics_engine.make_output(
            self.pipe,
            "offscreen buffer",
            -20,
            fb_props,
            winprops,
            p3d.GraphicsPipe.BF_refuse_window,
            self.win.get_gsg(),
            self.win,
        )  # DO NOT MAKE BUFFER via base.win.makeTextureBuffer for readback-purposed render target (F_rgba32)
        # base.win.makeTextureBuffer create a default render target buffer (F_rgba)
        self.buffer1.setClearColor(p3d.Vec4(0, 0, 0, 0))
        self.buffer1.setSort(10)

        # Create the render textures
        tex1 = p3d.Texture()  # self.buffer1.getTexture()
        tex2 = p3d.Texture()
        tex1.set_format(p3d.Texture.F_rgba32i)
        tex2.set_format(p3d.Texture.F_rgba32i)
        # tex1.set_component_type(p3d.Texture.T_unsigned_int)
        # tex2.set_component_type(p3d.Texture.T_unsigned_int)
        tex1.setClearColor(p3d.Vec4(0, 0, 0, 0))
        tex2.setClearColor(p3d.Vec4(0, 0, 0, 0))
        tex1.clear()
        tex2.clear()
        self.buffer1.addRenderTexture(tex1, p3d.GraphicsOutput.RTM_bind_or_copy | p3d.GraphicsOutput.RTM_copy_ram, p3d.GraphicsOutput.RTP_color)
        # I dont know why the RTP_aux_rgba_x with RTM_copy_ram (F_rgba32i?!) affects incorrect render-to-texture result.
        # so, tricky.. the tex2 only contains pos info (no need to readback to RAM)
        self.buffer1.addRenderTexture(
            tex2,  # p3d.GraphicsOutput.RTM_bind_or_copy |
            p3d.GraphicsOutput.RTM_bind_or_copy,
            p3d.GraphicsOutput.RTP_aux_rgba_0,
        )

        # Set up the offscreen camera
        self.cam1 = self.makeCamera(self.buffer1, scene=self.renderSVM, lens=self.cam.node().getLens())
        self.cam1.reparentTo(self.cam)

        # Set up a buffer for the first pass
        fb_props2 = p3d.FrameBufferProperties()
        fb_props2.setRgbColor(True)
        fb_props2.set_rgba_bits(8, 8, 8, 8)
        # fb_props2.setAuxRgba(1)  # Set up one auxiliary buffer
        fb_props2.setDepthBits(24)
        self.buffer2 = self.graphics_engine.make_output(
            self.pipe,
            "offscreen buffer2",
            -1,
            fb_props2,
            winprops,
            p3d.GraphicsPipe.BF_refuse_window,
            self.win.get_gsg(),
            self.win,
        )
        # self.buffer2 = self.win.makeTextureBuffer("buffer2", winSizeX, winSizeY) # this includes a default render target RTP_color
        texP0 = p3d.Texture()
        texP0.set_format(p3d.Texture.F_rgba)
        texP0.set_component_type(p3d.Texture.T_unsigned_byte)
        self.buffer2.addRenderTexture(texP0, p3d.GraphicsOutput.RTM_bind_or_copy, p3d.GraphicsOutput.RTP_color)

        self.buffer2.setClearColor(p3d.Vec4(0, 0, 0, 1))
        self.buffer2.setSort(0)

        # Set up a camera for the first pass
        self.cam2 = self.makeCamera(self.buffer2, scene=self.renderObj, lens=self.cam.node().getLens())
        self.cam2.reparentTo(self.cam)

        self.boat = self.loader.loadModel("avikus_boat.glb")
        self.boat.setHpr(0, -90, 180)

        bbox = self.boat.getTightBounds()
        scale = boat_length / (bbox[1].y - bbox[0].y)
        self.boat.setScale(scale)

        # bbox = self.boat.getTightBounds()
        # print(bbox)
        # self.boat.setPos(-bbox[0].z)
        self.boat.reparentTo(self.renderObj)

        self.axis = self.loader.loadModel("zup-axis")
        # self.axis.setPos(0, 0, 0)
        # self.axis.setHpr(180, 0, 0)
        self.axis.reparentTo(self.renderObj)

        manager = FilterManager(self.win, self.cam)
        self.quad = manager.renderSceneInto(colortex=None)  # make dummy texture... for post processing...
        # mySvm.quad = manager.renderQuadInto(colortex=tex)
        self.quad.setShader(Shader.load(Shader.SL_GLSL, vertex="post1_vs.glsl", fragment="svm_post1_ps_real.glsl"))
        self.quad.setShaderInput("texGeoInfo0", self.buffer1.getTexture(0))
        self.quad.setShaderInput("texGeoInfo1", self.buffer1.getTexture(1))
        self.quad.setShaderInput("texGeoInfo2", self.buffer2.getTexture(0))

        def GeneratePlaneNode(svmBase):
            # shader setting for SVM
            svmBase.planeShader = Shader.load(Shader.SL_GLSL, vertex="svm_vs.glsl", fragment="svm_ps_plane_real.glsl")
            vdata = p3d.GeomVertexData("triangle_data", p3d.GeomVertexFormat.getV3t2(), p3d.Geom.UHStatic)
            vdata.setNumRows(4)  # optional for performance enhancement!
            vertex = p3d.GeomVertexWriter(vdata, "vertex")
            texcoord = p3d.GeomVertexWriter(vdata, "texcoord")

            bbox = svmBase.boat.getTightBounds()
            print(bbox)
            self.waterZ = 10
            waterPlaneLength = 1000
            vertex.addData3(-waterPlaneLength, -waterPlaneLength, self.waterZ)
            vertex.addData3(waterPlaneLength, -waterPlaneLength, self.waterZ)
            vertex.addData3(waterPlaneLength, waterPlaneLength, self.waterZ)
            vertex.addData3(-waterPlaneLength, waterPlaneLength, self.waterZ)
            # not use...
            texcoord.addData2(0, 1)
            texcoord.addData2(1, 1)
            texcoord.addData2(1, 0)
            texcoord.addData2(0, 0)

            primTris = p3d.GeomTriangles(p3d.Geom.UHStatic)
            primTris.addVertices(0, 1, 2)
            primTris.addVertices(0, 2, 3)

            geom = p3d.Geom(vdata)
            geom.addPrimitive(primTris)
            geomNode = p3d.GeomNode("SVM Plane")
            geomNode.addGeom(geom)
            # note nodePath is the instance for node (obj resource)
            svmBase.plane = p3d.NodePath(geomNode)
            svmBase.plane.setTwoSided(True)
            svmBase.plane.setShader(svmBase.planeShader)

            # the following mat4 array does not work...
            # matViewProjs = [p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f()]
            # svmBase.plane.setShaderInput("matViewProjs", matViewProjs)

            # svmBase.planeTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
            for i in range(4):
                svmBase.plane.setShaderInput("matViewProj" + str(i), p3d.Mat4())
                svmBase.quad.setShaderInput("matViewProj" + str(i), p3d.Mat4())

            svmBase.planeTexArray = p3d.Texture()
            svmBase.planeTexArray.setup2dTextureArray(img_width, img_height, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
            svmBase.plane.setShaderInput("cameraImgs", svmBase.planeTexArray)
            svmBase.quad.setShaderInput("cameraImgs", svmBase.planeTexArray)

            svmBase.camPositions = [p3d.LVector4f(), p3d.LVector4f(), p3d.LVector4f(), p3d.LVector4f()]
            svmBase.plane.setShaderInput("camPositions", svmBase.camPositions)

            # initial setting like the above code! (for resource optimization)
            # svmBase.semanticTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
            svmBase.semanticTexArray = p3d.Texture()
            svmBase.semanticTexArray.setup2dTextureArray(img_width, img_height, 4, p3d.Texture.T_int, p3d.Texture.F_r32i)
            print(img_width, img_height)
            svmBase.plane.setShaderInput("semanticImgs", svmBase.semanticTexArray)
            svmBase.quad.setShaderInput("semanticImgs", svmBase.semanticTexArray)

        GeneratePlaneNode(self)
        self.plane.reparentTo(self.renderSVM)
        self.accept("r", self.shaderRecompile)

        self.bufferViewer.setPosition("llcorner")
        self.bufferViewer.setCardSize(0.5, 0)
        self.accept("v", self.bufferViewer.toggleEnable)

        self.taskMgr.add(self.readTextureData, "readTextureData")
        self.buffer1.set_active(False)
        self.buffer2.set_active(False)

    def readTextureData(self, task):
        self.buffer1.set_active(True)
        self.buffer2.set_active(True)
        self.win.set_active(False)
        self.graphics_engine.render_frame()

        # Get the render target texture
        tex0 = self.buffer1.getTexture(0)
        # Read the texture data into a PTAUchar
        tex0_data = tex0.get_ram_image()
        # Convert the PTAUchar to a NumPy array
        np_texture0 = np.frombuffer(tex0_data, np.uint32)
        # Reshape the NumPy array to match the texture dimensions and number of channels
        np_texture0 = np_texture0.reshape((tex0.get_y_size(), tex0.get_x_size(), 4))

        # to do with array0, array1
        array0 = np_texture0.copy()
        tex0.setRamImage(array0)
        self.quad.setShaderInput("texGeoInfo0", tex0)

        self.buffer1.set_active(False)
        self.buffer2.set_active(False)
        self.win.set_active(True)
        return task.cont

    def shaderRecompile(self):
        self.planeShader = Shader.load(Shader.SL_GLSL, vertex="svm_vs.glsl", fragment="svm_ps_plane_real.glsl")
        self.plane.setShader(self.planeShader)

        self.quad.setShader(Shader.load(Shader.SL_GLSL, vertex="post1_vs.glsl", fragment="svm_post1_ps_real.glsl"))


def make_extrinsic_matrix(extrinsic):
    defaultYaw = 0
    camera_position = int(extrinsic["camera_position"])
    if camera_position == 1:
        print("position is 1")
        defaultYaw = 0
    elif camera_position == 3 or camera_position == 5:
        print("position is 3 or 5")
        defaultYaw = -90
    elif camera_position == 2 or camera_position == 4:
        print("position is 2 or 4")
        defaultYaw = 90
    elif camera_position == 6:
        print("position is 6")
        defaultYaw = 180
    rotation = [-extrinsic["rotation"]["pitch"], extrinsic["rotation"]["roll"], -extrinsic["rotation"]["yaw"] - defaultYaw]
    translation = [extrinsic["location"]["translation_x"], extrinsic["location"]["translation_y"], extrinsic["location"]["translation_z"]]
    rotation_matrix0 = euler_to_matrix(rotation)
    rotation_matrix1 = euler_to_matrix([0, 0, -90])
    rotation_matrix = rotation_matrix1 @ rotation_matrix0
    return translation, rotation_matrix


def euler_to_matrix(euler_angle):
    """
    :param euler_angle: [x,y,z] in degree
    :return: rotation matrix
    """
    # calculate rotation about the x-axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.deg2rad(-euler_angle[0])), -np.sin(np.deg2rad(-euler_angle[0]))],
            [0.0, np.sin(np.deg2rad(-euler_angle[0])), np.cos(np.deg2rad(-euler_angle[0]))],
        ],
        dtype=float,
    )
    # calculate rotation about the y-axis
    R_y = np.array(
        [
            [np.cos(np.deg2rad(-euler_angle[1])), 0.0, np.sin(np.deg2rad(-euler_angle[1]))],
            [0.0, 1.0, 0.0],
            [-np.sin(np.deg2rad(-euler_angle[1])), 0.0, np.cos(np.deg2rad(-euler_angle[1]))],
        ],
        dtype=float,
    )
    # calculate rotation about the z-axis
    R_z = np.array(
        [
            [np.cos(np.deg2rad(-euler_angle[2])), -np.sin(np.deg2rad(-euler_angle[2])), 0.0],
            [np.sin(np.deg2rad(-euler_angle[2])), np.cos(np.deg2rad(-euler_angle[2])), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return R_z @ R_y @ R_x


def make_view_matrix(translation, rotation_matrix):
    # Create a 4x4 identity matrix
    extrinsic_matrix = np.eye(4)

    # Insert rotation_matrix into the extrinsic_matrix
    extrinsic_matrix[:3, :3] = rotation_matrix

    # Insert translation into the last row of the extrinsic_matrix
    extrinsic_matrix[3, :3] = translation

    # Return the inverse of the extrinsic_matrix to get the view matrix
    return np.linalg.inv(extrinsic_matrix)


def make_projection_matrix(fx, fy, skew_c, cx, cy, img_width, img_height, near_p, far_p):
    q = far_p / (near_p - far_p)
    qn = far_p * near_p / (near_p - far_p)

    projection_matrix = p3d.LMatrix4f()
    projection_matrix[0][0] = 2.0 * fx / img_width
    projection_matrix[1][0] = -2.0 * skew_c / img_width
    projection_matrix[2][0] = (img_width + 2.0 * 0 - 2.0 * cx) / img_width
    projection_matrix[3][0] = 0
    projection_matrix[0][1] = 0
    projection_matrix[1][1] = 2.0 * fy / img_height
    projection_matrix[2][1] = -(img_height + 2.0 * 0 - 2.0 * cy) / img_height
    projection_matrix[3][1] = 0
    projection_matrix[0][2] = 0
    projection_matrix[1][2] = 0
    projection_matrix[2][2] = q
    projection_matrix[3][2] = qn
    projection_matrix[0][3] = 0
    projection_matrix[1][3] = 0
    projection_matrix[2][3] = -1.0
    projection_matrix[3][3] = 0

    # projection_matrix.transposeInPlace()
    return projection_matrix


def InitSVM(base, imageWidth, imageHeight):
    matProj = make_projection_matrix(fx, fy, skew_c, cx, cy, imageWidth, imageHeight, 10, 10000)
    # projMat.transposeInPlace()

    matViews = []

    matViewFront = p3d.LMatrix4f(
        0.9992660293819391,
        0.03830607856152497,
        0.00021649132087547457,
        0,
        0.019186887951899295,
        -0.5053915229159182,
        0.8626768061652347,
        0,
        0.03315517838852627,
        -0.8620394729419105,
        -0.5057555548247171,
        0,
        19.805833783411012,
        -519.1335350240013,
        437.3066450005394,
        1,
    )
    matViews.append(matViewFront)

    matViewRight = p3d.LMatrix4f(
        -0.05054856147246779,
        0.7059372391096549,
        -0.706468298914609,
        0,
        0.9978433106243765,
        0.06535797412724426,
        -0.006087910985401715,
        0,
        0.04187565372919936,
        -0.7052524013828096,
        -0.7077185033390969,
        0,
        116.8902552996818,
        -251.0251450146386,
        38.053183655967726,
        1,
    )
    matViews.append(matViewRight)

    matViewRear = p3d.LMatrix4f(
        -0.9992873520600425,
        -0.008568458995732121,
        0.036760978268622464,
        0,
        -0.03140387017959493,
        0.7290592323411924,
        -0.6837297950768376,
        0,
        -0.020942419883356943,
        -0.6843969734361116,
        -0.7288087525550985,
        0,
        13.334953660706106,
        -529.6492087936449,
        239.03534181190214,
        1,
    )
    matViews.append(matViewRear)

    matViewLeft = p3d.LMatrix4f(
        0.006980197018203659,
        -0.7020491889150968,
        0.7120942446005599,
        0,
        -0.9998233295733476,
        -0.017328178353838954,
        -0.00728312266837638,
        0,
        0.0174524064372837,
        -0.7119176009754166,
        -0.7020461116842365,
        0,
        -95.89513094995147,
        -251.08578539176265,
        31.449200853016557,
        1.0000000000000002,
    )
    matViews.append(matViewLeft)

    matViewProjs = [p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f()]
    for i, extrinsic in enumerate(extrinsic_parameter):
        # translation, rotation_matrix = make_extrinsic_matrix(extrinsic)
        # matView = make_view_matrix(translation, rotation_matrix)
        # matView = p3d.LMatrix4f(*matView.flatten())
        # matViewProj = matView * matProj
        matViewProj = matViews[i] * matProj
        matViewProjs[i] = matViewProj

        base.plane.setShaderInput("matViewProj" + str(i), matViewProj)
        base.quad.setShaderInput("matViewProj" + str(i), matViewProj)

    base.planeTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
    base.plane.setShaderInput("cameraImgs", base.planeTexArray)

    base.semanticTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_int, p3d.Texture.F_r32i)
    base.plane.setShaderInput("semanticImgs", base.semanticTexArray)

    print("Texture Initialized!")


base_path = "./data/ws_segmenet"
camera_positions = ["front", "right", "rear", "left"]
image_paths = {position: sorted(os.listdir(os.path.join(base_path, position, "images"))) for position in camera_positions}
semantic_paths = {position: sorted(os.listdir(os.path.join(base_path, position, "pseudo_color_prediction"))) for position in camera_positions}
num_images = len(image_paths["front"])

current_idx = 0


def loadNextImage(task):
    global current_idx

    imgs = []
    semantics = []

    for position in camera_positions:
        # Load image
        img = cv.imread(os.path.join(base_path, position, "images", image_paths[position][current_idx]))
        img = cv.resize(img, (img_width, img_height))
        img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        imgs.append(img)

        # Load semantic image
        semantic = cv.imread(os.path.join(base_path, position, "pseudo_color_prediction", semantic_paths[position][current_idx]))
        semantic = cv.resize(semantic, (img_width, img_height))
        semantic = cv.cvtColor(semantic, cv.COLOR_BGR2GRAY)
        semantics.append(semantic)

    imgnpArray = np.array(imgs).astype(np.uint8)
    imgArray = imgnpArray.reshape((4, img_width, img_height, 4))
    cameraArray = imgArray[:, :, :, :].copy()
    semanticArray = np.array(semantics).astype(np.int32)

    mySvm.planeTexArray.setRamImage(cameraArray)
    mySvm.semanticTexArray.setRamImage(semanticArray)

    current_idx += 1
    if current_idx >= num_images:
        current_idx = 0  # Reset the index to loop back to the first image

    return Task.cont


if __name__ == "__main__":
    mySvm = SurroundView()

    width = mySvm.win.get_x_size()
    height = mySvm.win.get_y_size()
    print("init w {}, init h {}".format(width, height))

    InitSVM(mySvm, img_width, img_height)

    mySvm.taskMgr.add(loadNextImage, "loadNextImageTask", sort=1, uponDeath=exit)

    mySvm.run()

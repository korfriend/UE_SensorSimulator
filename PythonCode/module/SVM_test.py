import json
import math
import queue
import random
import socket
import struct
import threading
import time

import cv2 as cv
import keyboard
import numpy as np
import panda3d
import panda3d.core as p3d
from direct.filter.FilterManager import FilterManager
from direct.showbase.ShowBase import ShowBase
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from panda3d.core import Shader, ShaderAttrib

import DepthToPoint
import UDP_ReceiverSingle
from draw_sphere import draw_sphere

# from semantic_label_generator import get_semantic_label

color_map = [
    (50, 50, 50),
    (130, 120, 110),
    (255, 35, 35),
    (35, 255, 35),
    (35, 35, 255),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

colormap = cm.get_cmap("viridis", 1000)

winSizeX = 1024
winSizeY = 1024


############ Main Thread
class SurroundView(ShowBase):
    def __init__(self):
        super().__init__()
        winprops = p3d.WindowProperties()
        winprops.setSize(winSizeX, winSizeX)
        self.win.requestProperties(winprops)

        # ----------------카메라 프로퍼티
        self.render.setAntialias(p3d.AntialiasAttrib.MAuto)
        self.cam.setPos(0, 0, 3000)
        self.cam.lookAt(p3d.LPoint3f(0, 0, 0), p3d.LVector3f(0, 1, 0))
        self.camLens.setFov(60)

        self.renderObj = p3d.NodePath("fgRender")  # ?
        self.renderSVM = p3d.NodePath("bgRender")  # ?

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
        tex1 = p3d.Texture()  # self.buffer1.getTexture() #
        tex2 = p3d.Texture()
        tex1.set_format(p3d.Texture.F_rgba32i)
        tex2.set_format(p3d.Texture.F_rgba32i)
        # tex1.set_component_type(p3d.Texture.T_unsigned_int)
        # tex2.set_component_type(p3d.Texture.T_unsigned_int)
        tex1.setClearColor(p3d.Vec4(0, 0, 0, 0))
        tex2.setClearColor(p3d.Vec4(0, 0, 0, 0))
        tex1.clear()
        tex2.clear()
        self.buffer1.addRenderTexture(
            tex1,
            p3d.GraphicsOutput.RTM_bind_or_copy | p3d.GraphicsOutput.RTM_copy_ram,
            p3d.GraphicsOutput.RTP_color,
        )
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

        # 보트 로드
        self.boat = self.loader.loadModel("avikus_boat.glb")
        self.boat.setScale(p3d.Vec3(30, 30, 30))
        self.boat.set_hpr(0, 90, 90)
        bbox = self.boat.getTightBounds()
        print("boat bound : ", bbox)
        center = (bbox[0] + bbox[1]) * 0.5
        self.boat.setPos(-bbox[0].z)
        self.boat.reparentTo(self.renderObj)

        self.axis = self.loader.loadModel("zup-axis")
        self.axis.setPos(0, 0, 0)
        self.axis.setScale(100)
        self.axis.reparentTo(self.renderObj)

        self.isPointCloudSetup = False
        self.lidarRes = 0
        self.lidarChs = 0
        self.numLidars = 0

        draw_sphere(self, 500000, (0, 0, 0), (1, 1, 1, 1))
        self.sphereShader = Shader.load(
            Shader.SL_GLSL,
            vertex="./shaders/sphere_vs.glsl",
            fragment="./shaders/sphere_ps.glsl",
        )
        self.sphere.setShader(self.sphereShader)
        self.sphere.reparentTo(self.renderObj)

        manager = FilterManager(self.win, self.cam)
        self.quad = manager.renderSceneInto(colortex=None)  # make dummy texture... for post processing...
        # mySvm.quad = manager.renderQuadInto(colortex=tex)
        self.quad.setShader(
            Shader.load(
                Shader.SL_GLSL,
                vertex="./shaders/post1_vs.glsl",
                fragment="./shaders/svm_post1_ps.glsl",
            )
        )
        self.quad.setShaderInput("texGeoInfo0", self.buffer1.getTexture(0))
        self.quad.setShaderInput("texGeoInfo1", self.buffer1.getTexture(1))
        self.quad.setShaderInput("texGeoInfo2", self.buffer2.getTexture(0))
        self.quad.setShaderInput("img_w", 256)
        self.quad.setShaderInput("img_h", 256)

        def GeneratePlaneNode(svmBase):
            # shader setting for SVM
            svmBase.planeShader = Shader.load(
                Shader.SL_GLSL,
                vertex="./shaders/svm_vs.glsl",
                fragment="./shaders/svm_ps_plane.glsl",
            )
            vdata = p3d.GeomVertexData("triangle_data", p3d.GeomVertexFormat.getV3t2(), p3d.Geom.UHStatic)
            vdata.setNumRows(4)  # optional for performance enhancement!
            vertex = p3d.GeomVertexWriter(vdata, "vertex")
            texcoord = p3d.GeomVertexWriter(vdata, "texcoord")

            bbox = svmBase.boat.getTightBounds()
            print("boat bound SVM: ", bbox)
            self.waterZ = 110  # 0.2
            waterPlaneLength = 6000
            vertex.addData3(-waterPlaneLength, waterPlaneLength, self.waterZ)
            vertex.addData3(waterPlaneLength, waterPlaneLength, self.waterZ)
            vertex.addData3(waterPlaneLength, -waterPlaneLength, self.waterZ)
            vertex.addData3(-waterPlaneLength, -waterPlaneLength, self.waterZ)
            texcoord.addData2(0, 0)
            texcoord.addData2(1, 0)
            texcoord.addData2(1, 1)
            texcoord.addData2(0, 1)

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
            svmBase.plane.setShaderInput("img_w", 256)
            svmBase.plane.setShaderInput("img_h", 256)

            # the following mat4 array does not work...
            # matViewProjs = [p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f()]
            # svmBase.plane.setShaderInput("matViewProjs", matViewProjs)

            # svmBase.planeTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
            for i in range(4):
                svmBase.plane.setShaderInput("matViewProj" + str(i), p3d.Mat4())
                svmBase.sphere.setShaderInput("matViewProj" + str(i), p3d.Mat4())
                svmBase.quad.setShaderInput("matViewProj" + str(i), p3d.Mat4())

            svmBase.planeTexArray = p3d.Texture()
            svmBase.planeTexArray.setup2dTextureArray(256, 256, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
            svmBase.plane.setShaderInput("cameraImgs", svmBase.planeTexArray)
            svmBase.quad.setShaderInput("cameraImgs", svmBase.planeTexArray)
            svmBase.sphere.setShaderInput("cameraImgs", svmBase.planeTexArray)

            svmBase.camPositions = [
                p3d.LVector4f(),
                p3d.LVector4f(),
                p3d.LVector4f(),
                p3d.LVector4f(),
            ]
            svmBase.plane.setShaderInput("camPositions", svmBase.camPositions)

            # initial setting like the above code! (for resource optimization)
            # svmBase.semanticTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
            svmBase.semanticTexArray = p3d.Texture()
            svmBase.semanticTexArray.setup2dTextureArray(256, 256, 4, p3d.Texture.T_int, p3d.Texture.F_r32i)
            svmBase.plane.setShaderInput("semanticImgs", svmBase.semanticTexArray)
            svmBase.quad.setShaderInput("semanticImgs", svmBase.semanticTexArray)
            svmBase.sphere.setShaderInput("semanticImgs", svmBase.semanticTexArray)

            # zeros = np.ones((4, 256, 256), dtype=np.int32)
            # svmBase.semanticTexArray.setRamImage(zeros)

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
        self.planeShader = Shader.load(
            Shader.SL_GLSL,
            vertex="./shaders/svm_vs.glsl",
            fragment="./shaders/svm_ps_plane.glsl",
        )
        self.plane.setShader(mySvm.planeShader)
        self.sphereShader = Shader.load(
            Shader.SL_GLSL,
            vertex="./shaders/sphere_vs.glsl",
            fragment="./shaders/sphere_ps.glsl",
        )
        self.sphere.setShader(self.sphereShader)

        self.quad.setShader(
            Shader.load(
                Shader.SL_GLSL,
                vertex="./shaders/post1_vs.glsl",
                fragment="./shaders/svm_post1_ps.glsl",
            )
        )


def GeneratePointNode():
    svmBase = mySvm
    # note: use Geom.UHDynamic instead of Geom.UHStatic (resource setting for immutable or dynamic)
    vdata = p3d.GeomVertexData("point_data", p3d.GeomVertexFormat.getV3c4(), p3d.Geom.UHDynamic)
    numMaxPoints = svmBase.lidarRes * svmBase.lidarChs * svmBase.numLidars
    # 4 refers to the number of cameras
    vdata.setNumRows(numMaxPoints)
    vertex = p3d.GeomVertexWriter(vdata, "vertex")
    color = p3d.GeomVertexWriter(vdata, "color")
    # vertex.reserveNumRows(numMaxPoints)
    # color.reserveNumRows(numMaxPoints)
    vertex.setRow(0)
    color.setRow(0)
    # the following two points define the initial mix/max bounding box used for view-frustom culling
    # as the point cloud is supposed to be visible at all time, the min/max bounding box is set to large enough
    vertex.addData3f(-10000.0, -10000.0, -10000.0)
    vertex.addData3f(10000.0, 10000.0, 10000.0)
    color.addData4f(0, 0, 0, 0.0)
    color.addData4f(0, 0, 0, 0.0)
    for i in range(numMaxPoints):
        vertex.addData3f(10000.0, 10000.0, 10000.0)
        # https://docs.panda3d.org/1.9/cpp/programming/internal-structures/other-manipulation/more-about-reader-writer-rewriter
        # This allows you to store color components in the range 0.0 .. 1.0, and get the expected result (that is, the value is scaled into the range 0 .. 255). A similar conversion happens when data is read.
        color.addData4f(1, 0, 0, 0.0)

    primPoints = p3d.GeomPoints(p3d.Geom.UHDynamic)
    primPoints.addConsecutiveVertices(0, numMaxPoints)
    # primPoints.addConsecutiveVertices(0, len(points))

    geom = p3d.Geom(vdata)
    geom.addPrimitive(primPoints)
    geomNode = p3d.GeomNode("Lidar Points")
    geomNode.addGeom(geom)

    svmBase.pointsVertex = vertex
    svmBase.pointsColor = color
    svmBase.points = p3d.NodePath(geomNode)
    svmBase.points.setTwoSided(True)
    # self.points.setShader(svmBase.planeShader)
    # svmBase.points.reparentTo(svmBase.render)
    svmBase.points.reparentTo(svmBase.renderObj)
    svmBase.points.setTag("ObjScene", "True")

    # material = p3d.Material()
    # material.setShininess(1000)
    # svmBase.points.setMaterial(material)
    svmBase.points.set_render_mode_thickness(5)
    svmBase.isPointCloudSetup = True


def UpdateResource(task):
    PacketProcessing(mySvm.packetInit, mySvm.qQ)

    screenshot = mySvm.win.getScreenshot()  # get the screenshot
    img = np.frombuffer(screenshot.getRamImageAs("RGB").getData(), dtype=np.uint8)  # convert the screenshot to an array
    img = img.reshape((screenshot.getYSize(), screenshot.getXSize(), 3))  # reshape the array
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # convert the color space
    img = cv.flip(img, 0)  # flip the array
    mask = cv.inRange(img, (0, 0, 0), (0, 0, 0))
    dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
    cv.imshow("img", img)
    cv.imshow("mask", mask)
    cv.imshow("dst", dst)

    return task.cont


def InitSVM(base, numLidars, lidarRes, lidarChs, imageWidth, imageHeight, imgs, worldpointlist):
    if base.isInitializedUDP is True:
        return

    def createOglProjMatrix(fov, aspectRatio, n, f):
        tanHalfFovy = math.tan(fov / 2.0 * np.deg2rad(1))
        tanHalfFovx = tanHalfFovy * aspectRatio
        # col major in pand3d core but memory convention is based on the conventional row major
        # GLM_DEPTH_CLIP_SPACE == GLM_DEPTH_ZERO_TO_ONE version
        projMat = p3d.LMatrix4f(
            1.0 / tanHalfFovx,
            0,
            0,
            0,
            0,
            1.0 / tanHalfFovy,
            0,
            0,
            0,
            0,
            f / (n - f),
            -f * n / (f - n),
            0,
            0,
            -1,
            0,
        )
        projMat.transpose_in_place()
        return projMat

    def computeLookAt(camera_pos, camera_target, camera_up):
        forward = camera_pos - camera_target
        forward.normalize()
        right = forward.cross(camera_up)
        right.normalize()
        up = right.cross(forward)
        # print(("right {}").format(right))
        # print(("up {}").format(up))
        # print(("forward  {}").format(forward))
        # row major in pand3d core but memory convention is based on the conventional column major
        matLookAt = p3d.LMatrix4f(
            right[0],
            up[0],
            forward[0],
            0.0,
            right[1],
            up[1],
            forward[1],
            0.0,
            right[2],
            up[2],
            forward[2],
            0.0,
            -p3d.LVector3f.dot(right, camera_pos),
            -p3d.LVector3f.dot(up, camera_pos),
            -p3d.LVector3f.dot(forward, camera_pos),
            1.0,
        )
        return matLookAt

    base.isInitializedUDP = True

    print(("Num Lidars : {Num}").format(Num=numLidars))
    print(("Lidar Channels : {Num}").format(Num=lidarChs))
    print(("Camera Width : {Num}").format(Num=imageWidth))
    print(("Camera Height : {Num}").format(Num=imageHeight))

    base.lidarRes = lidarRes
    base.lidarChs = lidarChs
    base.numLidars = numLidars

    GeneratePointNode()

    camera_fov = packetInit["Fov"]  # 150
    verFoV = 2 * math.atan(math.tan(camera_fov * np.deg2rad(1) / 2) * (imageHeight / imageWidth)) * np.rad2deg(1)
    projMat = createOglProjMatrix(verFoV, imageWidth / imageHeight, 10, 100000)

    # LHS
    center_z = -0

    # sensor_pos_array = [
    #     p3d.Vec3(340, 0, 236 - center_z),
    #     p3d.Vec3(155, 140, 186 - center_z),
    #     p3d.Vec3(-400, 0, 186 - center_z),
    #     p3d.Vec3(155, -140, 146 - center_z),
    # ]
    sensor_pos_array = [
        p3d.Vec3(
            packetInit["CameraF_location_x"],
            packetInit["CameraF_location_y"],
            packetInit["CameraF_location_z"],
        ),
        p3d.Vec3(
            packetInit["CameraR_location_x"],
            packetInit["CameraR_location_y"],
            packetInit["CameraR_location_z"],
        ),
        p3d.Vec3(
            packetInit["CameraB_location_x"],
            packetInit["CameraB_location_y"],
            packetInit["CameraB_location_z"],
        ),
        p3d.Vec3(
            packetInit["CameraL_location_x"],
            packetInit["CameraL_location_y"],
            packetInit["CameraL_location_z"],
        ),
    ]

    sensor_rot_z_array = [0, 90, 180, -90]  # F R B L

    # sensor_rot_z_array = [90, -90, 0, 180]

    cam_pos = p3d.Vec3(0, 0, 250)

    # cam_rot_y_array = [-35, -43, -30, -43]
    cam_rot_y_array = [
        packetInit["CameraF_y"],
        packetInit["CameraR_y"],
        packetInit["CameraB_y"],
        packetInit["CameraL_y"],
    ]

    # LHS, RHS sameg
    sensorMatLHS_array = [
        p3d.LMatrix4f(),
        p3d.LMatrix4f(),
        p3d.LMatrix4f(),
        p3d.LMatrix4f(),
    ]
    imgIdx = 0

    for sensorPos, camPos in zip(sensor_pos_array, base.camPositions):
        camPos = sensorPos + cam_pos

    base.plane.setShaderInput("camPositions", base.camPositions)

    matViewProjs = [p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f()]
    for deg, pos, rot_y in zip(sensor_rot_z_array, sensor_pos_array, cam_rot_y_array):
        sensorMatRHS = p3d.LMatrix4f.rotateMat(deg, p3d.Vec3(0, 0, -1)) * p3d.LMatrix4f.translateMat(
            pos.x, -pos.y, pos.z
        )
        sensorMatLHS_array[imgIdx] = p3d.LMatrix4f.rotateMat(deg, p3d.Vec3(0, 0, 1)) * p3d.LMatrix4f.translateMat(
            pos.x, pos.y, pos.z
        )

        localCamMat = p3d.LMatrix4f.rotateMat(rot_y, p3d.Vec3(0, -1, 0)) * p3d.LMatrix4f.translateMat(cam_pos)
        camMat = localCamMat * sensorMatRHS
        # camMat3 = camMat.getUpper3()  # or, use xformVec instead

        # think... LHS to RHS...
        # also points...

        camPos = camMat.xformPoint(p3d.Vec3(0, 0, 0))
        view = camMat.xformVec(p3d.Vec3(1, 0, 0))
        up = camMat.xformVec(p3d.Vec3(0, 0, 1))
        viewMat = computeLookAt(camPos, camPos + view, up)

        # viewProjMat = p3d.LMatrix4f()
        viewProjMat = viewMat * projMat

        matViewProjs[imgIdx] = viewProjMat
        if imgIdx == 1:
            print(("camPos1 {}").format(camMat.xformPoint(p3d.Vec3(0, 0, 0))))
            print(("camPos2 {}").format(pos))
            print(("camDir {}").format(camMat.xform(p3d.Vec3(1, 0, 0))))
            print(("camUp  {}").format(camMat.xform(p3d.Vec3(0, 0, 1))))
            print("############")

        base.plane.setShaderInput("matViewProj" + str(imgIdx), viewProjMat)
        # print("plane.setShaderInput")
        base.quad.setShaderInput("matViewProj" + str(imgIdx), viewProjMat)
        # print("plane.setShaderInput")
        base.sphere.setShaderInput("matViewProj" + str(imgIdx), viewProjMat)
        # print("plane.setShaderInput")
        imgIdx += 1

    base.planeTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
    base.plane.setShaderInput("cameraImgs", base.planeTexArray)
    base.sphere.setShaderInput("cameraImgs", base.planeTexArray)

    base.semanticTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_int, p3d.Texture.F_r32i)
    base.plane.setShaderInput("semanticImgs", base.semanticTexArray)

    base.sensorMatLHS_array = sensorMatLHS_array

    base.plane.setShaderInput("img_w", imageWidth)
    base.plane.setShaderInput("img_h", imageHeight)

    base.quad.setShaderInput("img_w", imageWidth)
    base.quad.setShaderInput("img_h", imageHeight)

    print("Texture Initialized!")


def ProcSvmFromPackets(
    base,
    numLidars,
    lidarRes,
    lidarChs,
    imageWidth,
    imageHeight,
    imgs,
    segs,
    worldpointlist,
):
    InitSVM(
        base,
        numLidars,
        lidarRes,
        lidarChs,
        imageWidth,
        imageHeight,
        imgs,
        worldpointlist,
    )

    numMaxPoints = lidarRes * lidarChs * numLidars

    base.pointsVertex.setRow(0)
    base.pointsColor.setRow(0)

    # testpointcloud-------------------------------------------------------
    # for i in worldpointlist:
    #     posPoint = p3d.LPoint3f(i[0][0], i[0][1], i[0][2]) # 신기하게 이렇게 접근해야한다
    #     posPointWS = base.sensorMatLHS_array[0].xformPoint(posPoint)
    #     base.pointsVertex.setData3f(posPointWS)
    #     base.pointsColor.setData4f( 1,0,0,1)

    for i in range(numMaxPoints):
        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        z = random.randint(0, 1000)
        base.pointsVertex.setData3f(x, y, z)
        base.pointsColor.setData4f(0, 0, 0, 0)

    # for i in range(numMaxPoints - numProcessPoints):
    #     base.pointsVertex.setData3f(10000, 10000, 10000)
    #     base.pointsColor.setData4f(0, 0, 0, 0)
    # ------------------------------------------------------------------------------------------
    # EH_img = [imgs[0], imgs[1],imgs[2], imgs[3]] FBLR FLBR
    # EH_img = [imgs[0], imgs[2],imgs[1], imgs[3]]

    # F R B L
    EH_img = [imgs[0], imgs[1], imgs[2], imgs[3]]
    SM_img = [segs[0], segs[1], segs[2], segs[3]]

    imgs_ = np.array(EH_img)
    segs_ = np.array(SM_img).astype(np.uint32)
    segs_ = segs_.reshape((4, imageHeight, imageWidth, 4))
    # dst[:, :, 0] = src      # B-채널만 가져오기
    segs_1 = segs_[:, :, :, 0].copy()
    base.planeTexArray.setRamImage(imgs_)
    base.semanticTexArray.setRamImage(segs_1)


def PacketProcessing(packetInit: dict, q: queue):
    # packetNum = packetInit["packetNum"]
    # bytesPoints = packetInit["bytesPoints"]
    # bytesDepthmap = packetInit["bytesDepthmap"]
    # bytesRGBmap = packetInit["bytesRGBmap"]
    numLidars = packetInit["numLidars"]
    lidarRes = packetInit["lidarRes"]
    lidarChs = packetInit["lidarChs"]
    imageWidth = packetInit["imageWidth"]
    imageHeight = packetInit["imageHeight"]

    if q.empty():
        # print("텅텅")
        time.sleep(0.02)
        # continue
    frameData = q.get()
    worldpointList = frameData[0]
    imgs = frameData[1]
    segImg = frameData[2]
    segRaw = frameData[3]
    # for img in imgs:
    #     img = img[:, :, :3]
    #     print(img.shape)
    #     segImg.append(get_semantic_label(img))
    #     time.sleep(1)
    # F, R , B  L camera 순서로 들어온다

    cv.namedWindow("img 0", cv.WINDOW_GUI_NORMAL)
    cv.namedWindow("img 1", cv.WINDOW_GUI_NORMAL)
    cv.namedWindow("img 2", cv.WINDOW_GUI_NORMAL)
    cv.namedWindow("img 3", cv.WINDOW_GUI_NORMAL)
    cv.imshow("img 0", imgs[0])
    cv.imshow("img 1", imgs[1])
    cv.imshow("img 2", imgs[2])
    cv.imshow("img 3", imgs[3])

    cv.moveWindow("img 0", 960, 600)
    cv.moveWindow("img 1", 1165, 600)
    cv.moveWindow("img 2", 1370, 600)
    cv.moveWindow("img 3", 1575, 600)
    cv.resizeWindow("img 0", 200, int(200 * 1280 / 1280.0))
    cv.resizeWindow("img 1", 200, int(200 * 1280 / 1280.0))
    cv.resizeWindow("img 2", 200, int(200 * 1280 / 1280.0))
    cv.resizeWindow("img 3", 200, int(200 * 1280 / 1280.0))

    cv.namedWindow("s0", cv.WINDOW_GUI_NORMAL)
    cv.namedWindow("s1", cv.WINDOW_GUI_NORMAL)
    cv.namedWindow("s2", cv.WINDOW_GUI_NORMAL)
    cv.namedWindow("s3", cv.WINDOW_GUI_NORMAL)
    cv.imshow("s0", segImg[0])
    cv.imshow("s1", segImg[1])
    cv.imshow("s2", segImg[2])
    cv.imshow("s3", segImg[3])

    cv.moveWindow("s0", 960, 800)
    cv.moveWindow("s1", 1165, 800)
    cv.moveWindow("s2", 1370, 800)
    cv.moveWindow("s3", 1575, 800)
    cv.resizeWindow("s0", 200, int(200 * 1280 / 1280.0))
    cv.resizeWindow("s1", 200, int(200 * 1280 / 1280.0))
    cv.resizeWindow("s2", 200, int(200 * 1280 / 1280.0))
    cv.resizeWindow("s3", 200, int(200 * 1280 / 1280.0))

    cv.waitKey(1)

    ProcSvmFromPackets(
        mySvm,
        numLidars,
        lidarRes,
        lidarChs,
        imageWidth,
        imageHeight,
        imgs,
        segRaw,
        worldpointList,
    )


if __name__ == "__main__":
    mySvm = SurroundView()
    mySvm.isInitializedUDP = False

    width = mySvm.win.get_x_size()
    height = mySvm.win.get_y_size()
    print("init w {}, init h {}".format(width, height))

    packetInit = dict()
    q = queue.Queue(maxsize=10)

    mySvm.qQ = q
    mySvm.packetInit = packetInit
    mySvm.taskMgr.add(UpdateResource, "UpdateResource", sort=0)
    print("UDP server up and listening")
    t1 = threading.Thread(target=UDP_ReceiverSingle.ReceiveData, args=(packetInit, q))

    t1.start()

    while True:
        if len(packetInit) != 0:
            print("run")
            mySvm.run()
            break

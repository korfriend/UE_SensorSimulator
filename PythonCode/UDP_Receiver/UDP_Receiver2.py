import threading
import socket
import struct

import numpy as np
import math
import cv2 as cv
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Shader, ShaderAttrib
import panda3d.core as p3d
import panda3d
from draw_sphere import draw_sphere
from direct.filter.FilterManager import FilterManager

import json
import keyboard

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
colormap = cm.get_cmap('viridis', 1000)

print('Pandas Version :', panda3d.__version__)

winSizeX = 1024
winSizeY = 1024
############ Main Thread
class SurroundView(ShowBase):
    def __init__(self):
        super().__init__()
        
        winprops = p3d.WindowProperties()
        winprops.setSize(winSizeX, winSizeX)
        self.win.requestProperties(winprops)
        
        # https://docs.panda3d.org/1.10/python/programming/render-attributes/antialiasing
        self.render.setAntialias(p3d.AntialiasAttrib.MAuto)
        # 카메라 위치 조정 
        self.cam.setPos(0, 0, 4000)
        self.cam.lookAt(p3d.LPoint3f(0, 0, 0), p3d.LVector3f(0, 1, 0))
        #self.trackball.node().setMat(self.cam.getMat())
        #self.trackball.node().reset()
        #self.trackball.node().set_pos(0, 30, 0)
        
        self.renderObj = p3d.NodePath("fgRender")
        self.renderSVM = p3d.NodePath("bgRender")
        
        # Set up the offscreen buffer
        fb_props = p3d.FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.set_rgba_bits(32, 32, 32, 32) 
        fb_props.setAuxRgba(3)  # Set up one auxiliary buffer
        fb_props.setDepthBits(24)

        
        self.buffer1 = self.win.makeTextureBuffer("buffer1", winSizeX, winSizeY, to_ram=True, fbp=fb_props)
        #self.buffer1 = self.win.makeTextureBuffer("buffer1", winSizeX, winSizeY)
        self.buffer1.setClearColor(p3d.Vec4(0, 0, 0, 1))
        self.buffer1.setSort(10)
        
        # Create the render textures
        tex1 = p3d.Texture() # self.buffer1.getTexture() # 
        tex2 = p3d.Texture()
        tex1.set_format(p3d.Texture.F_rgba32)
        tex2.set_format(p3d.Texture.F_rgba32)
        tex1.setClearColor(p3d.Vec4(0, 0, 0, 0))
        tex2.setClearColor(p3d.Vec4(0, 0, 0, 0))
        tex1.clear()
        tex2.clear()
        #tex1.setup2dTexture(winSizeX, winSizeY, p3d.Texture.T_int, p3d.Texture.F_rgba32)
        #tex2.setup2dTexture(winSizeX, winSizeY, p3d.Texture.T_int, p3d.Texture.F_rgba32)

        self.buffer1.addRenderTexture(tex1, p3d.GraphicsOutput.RTM_bind_or_copy, p3d.GraphicsOutput.RTP_aux_rgba_0)
        self.buffer1.addRenderTexture(tex2, p3d.GraphicsOutput.RTM_bind_or_copy, p3d.GraphicsOutput.RTP_aux_rgba_1)

        #cm = p3d.CardMaker("card")
        #cm.setFrameFullscreenQuad()
        #card = self.render2d.attachNewNode(cm.generate())
        #card.setTexture(tex1)
        
        # Create a fullscreen quad to display the output
        #cm = p3d.CardMaker("card")
        #cm.setFrameFullscreenQuad()
        #card = self.render2d.attachNewNode(cm.generate())
        #card.setTexture(tex1)
        
        # Set up the offscreen camera
        self.cam1 = self.makeCamera(self.buffer1, scene=self.renderSVM, lens=self.cam.node().getLens())
        self.cam1.reparentTo(self.cam)

        # Set up a buffer for the first pass
        self.buffer2 = self.win.makeTextureBuffer("buffer2", winSizeX, winSizeY)
        self.buffer2.setClearColor(p3d.Vec4(0, 0, 0, 1))
        self.buffer2.setSort(0)

        # Set up a camera for the first pass
        self.cam2 = self.makeCamera(self.buffer2, scene=self.renderObj, lens=self.cam.node().getLens())
        self.cam2.reparentTo(self.cam)

        
        #보트 로드
        self.boat = self.loader.loadModel("avikus_boat.glb")
        self.boat.setScale(p3d.Vec3(30, 30, 30))
        self.boat.set_hpr(0, 90, 90)
        bbox = self.boat.getTightBounds()
        print(bbox)
        center = (bbox[0] + bbox[1]) * 0.5
        self.boat.setPos(-bbox[0].z)
        self.boat.reparentTo(self.renderObj)
        self.boat.setTag("ObjScene", "True")
        
        self.axis = self.loader.loadModel('zup-axis')
        self.axis.setPos(0, 0, 0)
        self.axis.setScale(100)
        self.axis.reparentTo(self.renderObj)
        self.axis.setTag("ObjScene", "True")
        
        self.isPointCloudSetup = False
        self.lidarRes = 0
        self.lidarChs = 0
        self.numLidars = 0
        
        draw_sphere(self, 500000, (0,0,0), (1,1,1,1))
        self.sphereShader = Shader.load(
            Shader.SL_GLSL, vertex="sphere_vs.glsl", fragment="sphere_ps.glsl")
        self.sphere.setShader(self.sphereShader)
        self.sphere.reparentTo(self.renderObj)
        self.sphere.setTag("ObjScene", "True")
        
        manager = FilterManager(self.win, self.cam)
        #tex = p3d.Texture()
        #tex.setup2dTexture(width, height, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
        self.quad = manager.renderSceneInto(colortex=None) # make dummy texture... for post processing...
        #mySvm.quad = manager.renderQuadInto(colortex=tex)
        self.quad.setShader(Shader.load(
                        Shader.SL_GLSL, vertex="post1_vs.glsl", fragment="svm_post1_ps.glsl"))
        self.quad.setShaderInput("texGeoInfo0", self.buffer1.getTexture(1))
        self.quad.setShaderInput("texGeoInfo1", self.buffer1.getTexture(2))
        self.quad.setShaderInput("texGeoInfo2", self.buffer2.get_texture())
        
        def GeneratePlaneNode(svmBase):
            #shader setting for SVM
            svmBase.planeShader = Shader.load(
                Shader.SL_GLSL, vertex="svm_vs.glsl", fragment="svm_ps_plane.glsl")
            vdata = p3d.GeomVertexData(
                'triangle_data', p3d.GeomVertexFormat.getV3t2(), p3d.Geom.UHStatic)
            vdata.setNumRows(4)  # optional for performance enhancement!
            vertex = p3d.GeomVertexWriter(vdata, 'vertex')
            texcoord = p3d.GeomVertexWriter(vdata, 'texcoord')

            bbox = svmBase.boat.getTightBounds()
            print(bbox)
            self.waterZ = 0 # 0.2
            waterPlaneLength = 2500
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

            # the following mat4 array does not work... 
            #matViewProjs = [p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f()]
            #svmBase.plane.setShaderInput("matViewProjs", matViewProjs)
            
            #svmBase.planeTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
            for i in range(4):
                svmBase.plane.setShaderInput('matViewProj' + str(i), p3d.Mat4())
                svmBase.sphere.setShaderInput('matViewProj' + str(i), p3d.Mat4())
                svmBase.quad.setShaderInput('matViewProj' + str(i), p3d.Mat4())

            svmBase.planeTexArray = p3d.Texture()
            svmBase.planeTexArray.setup2dTextureArray(256, 256, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
            svmBase.plane.setShaderInput('cameraImgs', svmBase.planeTexArray)
            svmBase.quad.setShaderInput("cameraImgs", svmBase.planeTexArray)
            svmBase.sphere.setShaderInput('cameraImgs', svmBase.planeTexArray)

            svmBase.camPositions = [p3d.LVector4f(), p3d.LVector4f(),
                  p3d.LVector4f(), p3d.LVector4f()]
            svmBase.plane.setShaderInput("camPositions", svmBase.camPositions)

            
            # initial setting like the above code! (for resource optimization)
            # svmBase.semanticTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
            svmBase.semanticTexArray = p3d.Texture()
            svmBase.semanticTexArray.setup2dTextureArray(256, 256, 4, p3d.Texture.T_int, p3d.Texture.F_r32i)
            svmBase.plane.setShaderInput('semanticImgs', svmBase.semanticTexArray)
            svmBase.quad.setShaderInput("semanticImgs", svmBase.semanticTexArray)
            svmBase.sphere.setShaderInput('semanticImgs', svmBase.semanticTexArray)

        GeneratePlaneNode(self)
        self.plane.reparentTo(self.renderSVM)
        self.plane.setTag("SvmScene", "True")
        self.accept('r', self.shaderRecompile)
        
        self.cam1.node().setInitialState(self.renderObj.getState())
        self.cam1.node().setTagStateKey("ObjScene")
        self.cam1.node().setTagState("True", self.renderObj.getState())

        self.cam2.node().setInitialState(self.renderSVM.getState())
        self.cam2.node().setTagStateKey("SvmScene")
        self.cam2.node().setTagState("True", self.renderSVM.getState())
        
        self.bufferViewer.setPosition("llcorner")
        self.bufferViewer.setCardSize(0.5, 0)
        self.accept("v", self.bufferViewer.toggleEnable)

    def shaderRecompile(self):
        self.planeShader = Shader.load(
            Shader.SL_GLSL, vertex="svm_vs.glsl", fragment="svm_ps_plane.glsl")
        self.plane.setShader(mySvm.planeShader)
        self.sphereShader = Shader.load(
            Shader.SL_GLSL, vertex="sphere_vs.glsl", fragment="sphere_ps.glsl")
        self.sphere.setShader(self.sphereShader)
        
        self.quad.setShader(Shader.load(
                Shader.SL_GLSL, vertex="post1_vs.glsl", fragment="svm_post1_ps.glsl"))   

mySvm = SurroundView()
width = mySvm.win.get_x_size()
height = mySvm.win.get_y_size()
print('init w {}, init h {}'.format(width, height))
#mySvm.bufferViewer.enable(1)

#vertices = np.random.uniform(-30.0, 30.0, size=(12800, 3)).astype(np.float32)
#colors = np.random.uniform(0.0, 1.0, size=(12800, 3)).astype(np.float32) # 무작위 색상
#points = [p3d.LPoint3f(p[0], p[1], p[2]) for p in vertices]
#print(len(points))

def GeneratePointNode(task):
    svmBase = mySvm
    if svmBase.lidarRes == 0 or svmBase.lidarChs == 0 or svmBase.numLidars == 0:
        return task.cont

    # note: use Geom.UHDynamic instead of Geom.UHStatic (resource setting for immutable or dynamic)
    vdata = p3d.GeomVertexData(
        'point_data', p3d.GeomVertexFormat.getV3c4(), p3d.Geom.UHDynamic)
    numMaxPoints = svmBase.lidarRes * svmBase.lidarChs * svmBase.numLidars
    # 4 refers to the number of cameras
    vdata.setNumRows(numMaxPoints)
    vertex = p3d.GeomVertexWriter(vdata, 'vertex')
    color = p3d.GeomVertexWriter(vdata, 'color')
    #vertex.reserveNumRows(numMaxPoints)
    #color.reserveNumRows(numMaxPoints)
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
        color.addData4f(0, 0, 0, 0.0)

    #vertex.setRow(0)
    #color.setRow(0)
    #for point, inputColor in zip(points, colors):
    #    vertex.addData3f(point)
    #    color.addData4f(inputColor[0], inputColor[1], inputColor[2], 1.0)

    primPoints = p3d.GeomPoints(p3d.Geom.UHDynamic)
    primPoints.addConsecutiveVertices(0, numMaxPoints)
    #primPoints.addConsecutiveVertices(0, len(points))

    geom = p3d.Geom(vdata)
    geom.addPrimitive(primPoints)
    geomNode = p3d.GeomNode("Lidar Points")
    geomNode.addGeom(geom)

    svmBase.pointsVertex = vertex
    svmBase.pointsColor = color
    svmBase.points = p3d.NodePath(geomNode)
    svmBase.points.setTwoSided(True)
    #self.points.setShader(svmBase.planeShader)
    #svmBase.points.reparentTo(svmBase.render)
    svmBase.points.reparentTo(svmBase.renderObj)
    svmBase.points.setTag("ObjScene", "True")

    #material = p3d.Material()
    #material.setShininess(1000)
    #svmBase.points.setMaterial(material)
    svmBase.points.set_render_mode_thickness(5)

    svmBase.isPointCloudSetup = True
    return task.done  # remove this task

def UpdateResource(task):
    return task.cont

####################### UDP Thread

localIP = "127.0.0.1"
localPort = 12000
bufferSize = 60000 

#msgFromServer = "Hello UDP Client"

bytesToSend = b'17'# str.encode(msgFromServer)

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)


# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))


print("UDP server up and listening")

mySvm.isInitializedUDP = True

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

def ProcSvmFromPackets(base, fullPackets, packetNum,
                       bytesPoints, bytesDepthmap, bytesRGBmap, 
                       numLidars, lidarRes, lidarChs, 
                       imageWidth, imageHeight):
    if base.isInitializedUDP == True:
        print(("Num Packets : {Num}").format(Num=packetNum))
        print(("Bytes of Points : {Num}").format(Num=bytesPoints))
        print(("Bytes of RGB map : {Num}").format(Num=bytesRGBmap))
        print(("Bytes of Depth map : {Num}").format(Num=bytesDepthmap))
        print(("Num Lidars : {Num}").format(Num=numLidars))
        print(("Lidar Resolution : {Num}").format(Num=lidarRes))
        print(("Lidar Channels : {Num}").format(Num=lidarChs))
        print(("Camera Width : {Num}").format(Num=imageWidth))
        print(("Camera Height : {Num}").format(Num=imageHeight))
        base.isInitializedUDP = False

        base.lidarRes = lidarRes
        base.lidarChs = lidarChs
        base.numLidars = numLidars
        base.taskMgr.add(GeneratePointNode, "GeneratePointNode")
        
        # Create a opengl convention projection matrix
        def createOglProjMatrix(fov, aspectRatio, n, f):
            tanHalfFovy = math.tan(fov / 2.0 * np.deg2rad(1))
            tanHalfFovx = tanHalfFovy * aspectRatio
            # col major in pand3d core but memory convention is based on the conventional row major
            # GLM_DEPTH_CLIP_SPACE == GLM_DEPTH_ZERO_TO_ONE version
            projMat = p3d.LMatrix4f(1.0 / tanHalfFovx, 0, 0, 0,
                                    0, 1.0 / tanHalfFovy, 0, 0,
                                    0, 0, f / (n - f), -f * n / (f - n),
                                    0, 0, -1, 0)
            projMat.transpose_in_place()
            return projMat

        def computeLookAt(camera_pos, camera_target, camera_up):
            forward = camera_pos - camera_target
            forward.normalize()
            right = forward.cross(camera_up)
            right.normalize()
            up = right.cross(forward)
            #print(("right {}").format(right))
            #print(("up {}").format(up))
            #print(("forward  {}").format(forward))
            # row major in pand3d core but memory convention is based on the conventional column major
            matLookAt = p3d.LMatrix4f(
                right[0], up[0], forward[0], 0.0,
                right[1], up[1], forward[1], 0.0,
                right[2], up[2], forward[2], 0.0,
                -p3d.LVector3f.dot(right, camera_pos),
                -p3d.LVector3f.dot(up, camera_pos),
                -p3d.LVector3f.dot(forward, camera_pos), 1.0)
            return matLookAt
        
        verFoV = 2 * math.atan(math.tan(150 * np.deg2rad(1) / 2) * (imageHeight/imageWidth)) * np.rad2deg(1)
        projMat = createOglProjMatrix(verFoV, imageWidth/imageHeight, 10, 100000)

        # LHS
        center_z = -95
        sensor_pos_array = [
            p3d.Vec3(30, 0, 40 - center_z), 
            p3d.Vec3(0, 60, 40 - center_z), 
            p3d.Vec3(-40, 0, 40 - center_z), 
            p3d.Vec3(0, -80, 40 - center_z)
            ]
        sensor_rot_z_array = [0, 90, 180, -90]
        cam_pos = p3d.Vec3(0, 0, 50)
        cam_rot_y = -10

        # LHS, RHS same
        localCamMat = p3d.LMatrix4f.rotateMat(cam_rot_y, p3d.Vec3(0, -1, 0)) * p3d.LMatrix4f.translateMat(cam_pos)
        sensorMatLHS_array = [p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f()]
        imgIdx = 0
        
        for sensorPos, camPos in zip(sensor_pos_array, base.camPositions):
            camPos = sensorPos + cam_pos
            
        base.plane.setShaderInput("camPositions", base.camPositions)

        matViewProjs = [p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f(), p3d.LMatrix4f()]
        for deg, pos in zip(sensor_rot_z_array, sensor_pos_array):
            sensorMatRHS = p3d.LMatrix4f.rotateMat(deg, p3d.Vec3(0, 0, -1)) * p3d.LMatrix4f.translateMat(pos.x, -pos.y, pos.z)
            sensorMatLHS_array[imgIdx] = p3d.LMatrix4f.rotateMat(deg, p3d.Vec3(0, 0, 1)) * p3d.LMatrix4f.translateMat(pos.x, pos.y, pos.z)
            
            camMat = localCamMat * sensorMatRHS
            #camMat3 = camMat.getUpper3()  # or, use xformVec instead
            
            # think... LHS to RHS...
            # also points...
            
            camPos = camMat.xformPoint(p3d.Vec3(0, 0, 0))
            view = camMat.xformVec(p3d.Vec3(1, 0, 0))
            up = camMat.xformVec(p3d.Vec3(0, 0, 1))
            viewMat = computeLookAt(camPos, camPos + view, up)

            #viewProjMat = p3d.LMatrix4f()
            #viewProjMat = viewMat * projMat
            viewProjMat = viewMat * projMat
            
            matViewProjs[imgIdx] = viewProjMat
            if imgIdx == 1:
                print(("camPos1 {}").format(camMat.xformPoint(p3d.Vec3(0, 0, 0))))
                print(("camPos2 {}").format(pos))
                print(("camDir {}").format(camMat.xform(p3d.Vec3(1, 0, 0)))) 
                print(("camUp  {}").format(camMat.xform(p3d.Vec3(0, 0, 1))))
                print("############")
                #print(("test pos1  {}").format(viewMat.xformPoint(p3d.Vec3(1000, 0, 0)))) 
                #print(("test pos2  {}").format(viewProjMat.xform(p3d.Vec4(1000, 0, 0, 1))))
                #print(("test pos3  {}").format(viewProjMat.xform(p3d.Vec4(30.15, 0, 0, 1))))
                #print(("test pos3  {}").format(viewProjMat.xform(p3d.Vec4(-1000, 0, 0, 1))))
                #base.plane.setShaderInput("matTest0", viewMat)
                #base.plane.setShaderInput("matTest1", projMat)

            base.plane.setShaderInput("matViewProj" + str(imgIdx), viewProjMat)
            base.quad.setShaderInput("matViewProj" + str(imgIdx), viewProjMat)
            base.sphere.setShaderInput("matViewProj" + str(imgIdx), viewProjMat)
            imgIdx += 1

        #base.plane.setShaderInput("matViewProjs", matViewProjs)
        #base.planePnm = p3d.PNMImage()
        #for i in range(4):
        #    base.planeTexs[i].setup2dTexture(
        #        imageWidth, imageHeight, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
        #    base.plane.setShaderInput(
        #        'myTexture' + str(i), base.planeTexs[i])

        base.planeTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
        base.plane.setShaderInput('cameraImgs', base.planeTexArray)
        base.sphere.setShaderInput('cameraImgs', base.planeTexArray)
        
        if bytesRGBmap > imageWidth * imageHeight * 4 * numLidars:
            # use this branch logic to add custom data in this scene
            print("Custom case for Semantic images")
            
        #for i in range(4):
        #    base.semanticTexs[i].setup2dTexture(
        #        imageWidth, imageHeight, p3d.Texture.T_unsigned_byte, p3d.Texture.F_red)
        #    base.plane.setShaderInput(
        #        'semanticTex' + str(i), base.semanticTexs[i])

        base.semanticTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_int, p3d.Texture.F_r32i)
        base.plane.setShaderInput('semanticImgs', base.semanticTexArray)

        base.sensorMatLHS_array = sensorMatLHS_array
        print("Texture Initialized!")
        
    if base.isPointCloudSetup == True:
        #print("Point Clout Update!!")
        # point cloud buffer : fullPackets[0:bytesPoints]
        numMaxPoints = lidarRes * lidarChs * numLidars
        numProcessPoints = 0
        
        base.pointsVertex.setRow(0)
        base.pointsColor.setRow(0)
        offsetPoints = 0
        for i in range(4):
            numSingleLidarPoints = int.from_bytes(fullPackets[offsetPoints: 4 + offsetPoints], "little")
            #print(("Num Process Points : {Num}").format(Num=numSingleLidarPoints))

            matSensorLHS = base.sensorMatLHS_array[i]
            offsetPoints += 4
            numProcessPoints += numSingleLidarPoints
            for j in range(numSingleLidarPoints):
                pX = struct.unpack('<f', fullPackets[0 + offsetPoints : 4 + offsetPoints])[0]
                pY = struct.unpack('<f', fullPackets[4 + offsetPoints : 8 + offsetPoints])[0]
                pZ = struct.unpack('<f', fullPackets[8 + offsetPoints: 12 + offsetPoints])[0]
                #cR = np.frombuffer(fullPackets[12 + offsetPoints, 13 + offsetPoints], dtype=np.int8)[0]
                cR = int.from_bytes(fullPackets[12 + offsetPoints : 13 + offsetPoints], "little")
                cG = int.from_bytes(fullPackets[13 + offsetPoints : 14 + offsetPoints], "little")
                cB = int.from_bytes(fullPackets[14 + offsetPoints : 15 + offsetPoints], "little")
                cA = int.from_bytes(fullPackets[15 + offsetPoints : 16 + offsetPoints], "little")
                #if j == 17 :
                #    print(("pos : {}, {}, {}, {}, {}").format(i, offsetPoints, pX, pY, pZ))
                #    print(("clr : {}, {}, {}, {}, {}, {}").format(i, offsetPoints, cR, cG, cB, cA))
                offsetPoints += 16
                posPoint = p3d.LPoint3f(pX, pY, pZ)
                
                # to do : transform posPoint (local) to world
                posPointWS = matSensorLHS.xformPoint(posPoint)
                posPointWS.y *= -1
                #y = posPointWS.y
                #posPointWS.y = posPointWS.x
                #posPointWS.x = y 
                #if posPointWS.z > base.waterZ + 1:
                
                color = colormap(posPointWS.z/150)
                cR = color[0] * 255.0
                cG = color[1] * 255.0
                cB = color[2] * 255.0
                
                base.pointsVertex.setData3f(posPointWS)
                base.pointsColor.setData4f(cB / 255.0, cG / 255.0, cR / 255.0, cA / 255.0)
                
                #if i == 0:
                #    base.pointsColor.setData4f(0, 0, 1, 1)
                #elif i == 1:
                #    base.pointsColor.setData4f(0, 1, 0, 1)
                #elif i == 2:
                #    base.pointsColor.setData4f(1, 0, 0, 1)
                #elif i == 3:
                #    base.pointsColor.setData4f(0, 1, 1, 1)



                #base.pointsColor.setData4f(1.0, 0, 0, 1.0)
            #print(("Num Process Points : {Num}").format(Num=numProcessPoints))

        for i in range(numMaxPoints - numProcessPoints):
            base.pointsVertex.setData3f(10000, 10000, 10000)
            base.pointsColor.setData4f(0, 0, 0, 0)
            
        #base.pointsVertex.setRow(0)
        #base.pointsColor.setRow(0)
        #for point, inputColor in zip(points, colors):
        #    base.pointsVertex.addData3f(point)
        #    base.pointsColor.addData4f(inputColor[0], inputColor[1], inputColor[2], 1.0)
        #print(("Total Points : {Num}").format(Num=numProcessPoints))
        #print(("Remaining Points : {Num}").format(Num=numMaxPoints - numProcessPoints))

        #print(("Is Position End : {End}").format(End=base.pointsVertex.isAtEnd()))
        #print(("Is Color End : {End}").format(End=base.pointsColor.isAtEnd()))
            
    # depth buffer : fullPackets[bytesPoints:bytesPoints + bytesDepthmap] ... 4 of (lidarRes * lidarchs * 4 bytes) 

    offsetColor = bytesPoints + bytesDepthmap
    imgBytes = imageWidth * imageHeight * 4
    imgs = []
    semantics = []
    # use this branch logic to add custom data in this scene
    isCustomImgs = bytesRGBmap > imageWidth * imageHeight * 4 * numLidars
    
    for i in range(4):
        #print(("AAA-1 {aa}, {bb}, {cc}").format(aa=imgBytes, bb=offsetColor, cc=i))
        if not isCustomImgs:
            imgnp = np.array(
                fullPackets[offsetColor + imgBytes * i: offsetColor + imgBytes * (i + 1)], dtype=np.uint8)

            #print("AAA-2")
            img = imgnp.reshape((imageHeight, imageWidth, 4))
            
            #print("AAA-3")
                # https://docs.panda3d.org/1.10/python/programming/texturing/simple-texturing
                # https://docs.panda3d.org/1.10/cpp/programming/advanced-loading/loading-resources-from-memory
            imgs.append(img)
        else :
            # if i % 2 == 0:  # imgs
                # imgs.append(img)
                #print("AAA-4")
                #base.planeTexs[int(i / 2)].setRamImage(img)

                #base.planeTexs[i].setRamImage(img)
                #base.planeTexs[i].setup2dTexture(
                #imageWidth, imageHeight, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
                #base.plane.setShaderInput(
                #    'myTexture' + str(i), base.planeTexs[i])
                #print(("Plane Texture : {Num}").format(Num=i))
            # else:  # semantics
                # semantic = np.zeros_like(img).astype(np.uint8)
                # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                # for j, color in enumerate(color_map):
                #     for k in range(3):
                #         semantic[:, :, k][img[:, :] == j] = color[k]
                # semantics.append(semantic)
                # use sampler2DArray in glsl rather than sampler2D
                # refer to 'sampler2DArray cameraImgs' and 'base.planeTexArray.setRamImage(imgArray)'
                # base.semanticTexs[int(i / 2)].setRamImage(img)
            imgnp = np.array(
                fullPackets[offsetColor + imgBytes * (i * 2) : offsetColor + imgBytes * (i * 2 + 1)], dtype=np.uint8)
            semanticnp = np.array(
                fullPackets[offsetColor + imgBytes * (i * 2 + 1) : offsetColor + imgBytes * (i * 2 + 2)], dtype=np.uint8)
            
            img = imgnp.reshape((imageHeight, imageWidth, 4))
            semantic = semanticnp.reshape((imageHeight, imageWidth, 4))
            color_img = np.zeros_like(semantic).astype(np.uint8)
            for j, color in enumerate(color_map):
                for k in range(3):
                    color_img[:, :, k][semantic[:, :, 0] == j] = color[k]

            imgs.append(img)
            semantics.append(color_img)

    if not isCustomImgs:
        imgnpArray = np.array(
            fullPackets[offsetColor + imgBytes * 0 : offsetColor + imgBytes * (3 + 1)], dtype=np.uint8)
        imgArray = imgnpArray.reshape((4, imageHeight, imageWidth, 4))
        base.planeTexArray.setRamImage(imgArray)
    else:
        imgnpArray = np.array(
            fullPackets[offsetColor + imgBytes * 0 : offsetColor + imgBytes * 8], dtype=np.uint8)
        imgArray = imgnpArray.reshape((8, imageHeight, imageWidth, 4))
        cameraArray = imgArray[::2, :, :, :].copy()
        semanticArray = imgArray[1::2, :, :, :].copy()
        
        semanticArray1 = semanticArray[..., 0]
        semanticArray1_2 = semanticArray1.astype(np.uint32)
        
        base.planeTexArray.setRamImage(cameraArray)
        base.semanticTexArray.setRamImage(semanticArray1_2)

    cv.imshow('image_deirvlon 0', imgs[0])
    cv.imshow('image_deirvlon 1', imgs[1])
    cv.imshow('image_deirvlon 2', imgs[2])
    cv.imshow('image_deirvlon 3', imgs[3])
    if isCustomImgs:
        cv.imshow("semantic_deirvlon 0", semantics[0])
        cv.imshow("semantic_deirvlon 1", semantics[1])
        cv.imshow("semantic_deirvlon 2", semantics[2])
        cv.imshow("semantic_deirvlon 3", semantics[3])
    cv.waitKey(1)
    
timeout = 2
UDPServerSocket.settimeout(timeout)
def ReceiveData():
    # Listen for incoming datagrams
    while(True):

        #print("listening")
        try:
            bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            #print("got it")
        
            packetInit = bytesAddressPair[0]
            addressInit = bytesAddressPair[1]

            #clientMsg = "Message from Client:{}".format(message)
            clientIP = "Client IP Address:{}".format(addressInit)

            index = int.from_bytes(packetInit[0:4], "little")
            
            if index == 0xffffffff:
                #print("packet start")
                packetNum = int.from_bytes(packetInit[4:8], "little")
                bytesPoints = int.from_bytes(packetInit[8:12], "little")
                bytesDepthmap = int.from_bytes(packetInit[12:16], "little")
                bytesRGBmap = int.from_bytes(packetInit[16:20], "little")
                # check code for semantic map for experimental
                numLidars = int.from_bytes(packetInit[20:24], "little")
                lidarRes = int.from_bytes(packetInit[24:28], "little")
                lidarChs = int.from_bytes(packetInit[28:32], "little")
                imageWidth = int.from_bytes(packetInit[32:36], "little")
                imageHeight = int.from_bytes(packetInit[36:40], "little")
            
                if packetNum == 0:
                    UDPServerSocket.sendto(bytesToSend, addressInit)
                    continue
                
                UDPServerSocket.sendto(bytesToSend, addressInit)

                fullPackets = bytearray(b'')
                #fullPackets = bytes(b'')
                packetIndex = 0
                # the following 'while' loop receives the entire packet data of each frame
                while(packetIndex < packetNum):
                    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
                    packet = bytesAddressPair[0]
                    address = bytesAddressPair[1]
                    UDPServerSocket.sendto(bytesToSend, addressInit)
                    index = int.from_bytes(packet[0:4], "little")
                    if (index != packetIndex) :
                        print(("Error {id}").format(id = index))
                    packetIndex += 1
                    fullPackets += packet[4:]
                    #print(("index {i}, num bytes {b}").format(i=index, b=len(packet)))
                
                # now one frame's full packet is completed
                data = {
                    "packetNum" : packetNum,
                    "bytesPoints" : bytesPoints,
                    "bytesDepthmap" : bytesDepthmap,
                    "bytesRGBmap" : bytesRGBmap,
                    "numLidars" : numLidars,
                    "lidarRes" : lidarRes,
                    "lidarChs" : lidarChs,
                    "imageWidth" : imageWidth,
                    "imageHeight" : imageHeight
                }
                with open("params.json", "w") as json_file:
                    json.dump(data, json_file, indent=4)
                
                with open('prevFrame.bin', 'wb') as f:
                    f.write(fullPackets)
                
                print(("Bytes of All Packets: {d}").format(d=len(fullPackets)))
                # to do 
                ProcSvmFromPackets(mySvm, fullPackets, packetNum,
                        bytesPoints, bytesDepthmap, bytesRGBmap, 
                        numLidars, lidarRes, lidarChs, 
                        imageWidth, imageHeight)
            
        except socket.timeout:
            print("No data received after {} seconds. Timed out.".format(timeout))
            with open("params.json", "r") as json_file:
                restored_data = json.load(json_file)

            packetNum = restored_data["packetNum"]
            bytesPoints = restored_data["bytesPoints"]
            bytesDepthmap = restored_data["bytesDepthmap"]
            bytesRGBmap = restored_data["bytesRGBmap"]
            numLidars = restored_data["numLidars"]
            lidarRes = restored_data["lidarRes"]
            lidarChs = restored_data["lidarChs"]
            imageWidth = restored_data["imageWidth"]
            imageHeight = restored_data["imageHeight"]
            
            with open('prevFrame.bin', 'rb') as f:
                fullPackets = bytearray(f.read())
                
            print(("Bytes of All Packets: {d}").format(d=len(fullPackets)))
                
            ProcSvmFromPackets(mySvm, fullPackets, packetNum,
                       bytesPoints, bytesDepthmap, bytesRGBmap, 
                       numLidars, lidarRes, lidarChs, 
                       imageWidth, imageHeight)
            
            keyboard.wait('space')
            continue
            

            

    #print(("Packets : {p0}, {p1}, {p2}, {p3}").format(p0=packet[0], p1=packet[1], p2=packet[2], p3=packet[3]))
    #print(index)
    #print(clientIP)

    # Sending a reply to client

if __name__ == "__main__":
    t = threading.Thread(target=ReceiveData, args=())
    t.start()

    print("SVM Start!")
    mySvm.run()

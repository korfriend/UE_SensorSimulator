import math

import cv2 as cv
import numpy as np
import panda3d.core as p3d
from direct.filter.FilterManager import FilterManager
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Shader

K1 = 1.281584985127447
K2 = 0.170043067138006
K3 = -0.023341058557079
K4 = 0.007690791651144
K5 = -0.001380968639013

img_width = 1920
img_height = 1080
fx = 345.12136354806347
fy = 346.09009197978003
cx = 959.5  # - img_width / 2
cy = 539.5  # - img_height / 2
skew_c = 0

extrinsic_parameters = [
    {  # front
        "pos_x": 0.0,  # -0.1,
        "pos_y": -640.0,  # -6.45,
        "pos_z": -227.0,  # -1.82,
        "euler_x": 59.6,
        "euler_y": 1.9,
        "euler_z": 1.1,
    },
    {  # right
        "pos_x": 210.0,  # 1.4,
        "pos_y": -100.0,  # 1.55,
        "pos_z": -155.0,  # -2.16,
        "euler_x": 44.9,
        "euler_y": 2.4001,
        "euler_z": 92.9,
    },
    {  # rear
        "pos_x": 0.0,  # 0.0,
        "pos_y": 550.5,  # 3.99,
        "pos_z": -188.0,  # -1.25,
        "euler_x": 43.2,
        "euler_y": -1.2,
        "euler_z": 181.8,
    },
    {  # left
        "pos_x": -198.0,  # -1.4,
        "pos_y": -100.0,  # 1.55,
        "pos_z": -155.0,  # -2.16,
        "euler_x": 45.0,
        "euler_y": 1,
        "euler_z": -89.6,
    },
]

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

        # self.disableMouse()

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
        self.buffer1.addRenderTexture(
            tex1, p3d.GraphicsOutput.RTM_bind_or_copy | p3d.GraphicsOutput.RTM_copy_ram, p3d.GraphicsOutput.RTP_color
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

        self.boat = self.loader.loadModel("avikus_boat.glb")
        self.boat.setHpr(0, -90, 180)

        bbox = self.boat.getTightBounds()
        scale = abs(extrinsic_parameters[0]["pos_y"] - extrinsic_parameters[2]["pos_y"]) / (bbox[1].y - bbox[0].y)
        self.boat.setScale(scale)

        bbox = self.boat.getTightBounds()
        origin = (bbox[0] + bbox[1]) / 2

        pos_x = 0
        for extrinsic_parameter in extrinsic_parameters:
            pos_x += extrinsic_parameter["pos_x"]
        pos_x /= len(extrinsic_parameters)

        pos_y = 0
        for extrinsic_parameter in extrinsic_parameters:
            pos_y += extrinsic_parameter["pos_y"]
        pos_y /= len(extrinsic_parameters)

        pos_z = 0
        for extrinsic_parameter in extrinsic_parameters:
            pos_z += extrinsic_parameter["pos_z"]
        pos_z /= len(extrinsic_parameters)

        self.boat.setPos(pos_x - origin.x, pos_y - origin.y, pos_z - origin.z)

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
            svmBase.planeTexArray.setup2dTextureArray(
                img_width, img_height, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba
            )
            svmBase.plane.setShaderInput("cameraImgs", svmBase.planeTexArray)
            svmBase.quad.setShaderInput("cameraImgs", svmBase.planeTexArray)

            svmBase.camPositions = [p3d.LVector4f(), p3d.LVector4f(), p3d.LVector4f(), p3d.LVector4f()]
            svmBase.plane.setShaderInput("camPositions", svmBase.camPositions)

            # initial setting like the above code! (for resource optimization)
            # svmBase.semanticTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
            svmBase.semanticTexArray = p3d.Texture()
            svmBase.semanticTexArray.setup2dTextureArray(
                img_width, img_height, 4, p3d.Texture.T_int, p3d.Texture.F_r32i
            )
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


def get_projection_matrix(fx, fy, skew_c, cx, cy, img_width, img_height, near_p, far_p):
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


def computeLookAtMatrix(eye, target, up):
    forward = target - eye
    forward.normalized()

    right = forward.cross(up)
    right.normalized()

    up_actual = right.cross(forward)
    up_actual.normalized()

    viewMatrix = p3d.LMatrix4f()
    viewMatrix[0][0] = right.x
    viewMatrix[1][0] = right.y
    viewMatrix[2][0] = right.z
    viewMatrix[0][1] = up_actual.x
    viewMatrix[1][1] = up_actual.y
    viewMatrix[2][1] = up_actual.z
    viewMatrix[0][2] = -forward.x
    viewMatrix[1][2] = -forward.y
    viewMatrix[2][2] = -forward.z
    viewMatrix[3][0] = -right.dot(eye)
    viewMatrix[3][1] = -up_actual.dot(eye)
    viewMatrix[3][2] = forward.dot(eye)

    # viewMatrix.transposeInPlace()
    return viewMatrix


def MakeQt(euler):
    x = math.radians(euler.x)
    y = math.radians(euler.y)
    z = math.radians(euler.z)

    # http://www.mathworks.com/matlabcentral/fileexchange/
    # 	20696-function-to-convert-between-dcm-euler-angles-quaternions-and-euler-vectors/
    # 	content/SpinCalc.m

    c1 = math.cos(x / 2)
    c2 = math.cos(y / 2)
    c3 = math.cos(z / 2)

    s1 = math.sin(x / 2)
    s2 = math.sin(y / 2)
    s3 = math.sin(z / 2)

    q = p3d.LQuaternionf()
    q.x = s1 * c2 * c3 + c1 * s2 * s3
    q.y = c1 * s2 * c3 - s1 * c2 * s3
    q.z = c1 * c2 * s3 + s1 * s2 * c3
    q.w = c1 * c2 * c3 - s1 * s2 * s3

    return q


def get_view_matrix(pos_x, pos_y, pos_z, euler_x, euler_y, euler_z):
    # euler_x = math.radians(euler_x)
    # euler_y = math.radians(euler_y)
    # euler_z = math.radians(euler_z)

    translation_matrix = p3d.LMatrix4f.translateMat(pos_x, pos_y, pos_z)
    S = p3d.LMatrix4f.scaleMat(1, 1, -1)

    q = MakeQt(p3d.LVector3f(euler_x, euler_y, euler_z))
    matrix = p3d.Mat4()
    # matrix.setQuat(q)
    q.extract_to_matrix(matrix)
    obj = p3d.NodePath("MyTest")

    # Rotate the object using Euler angles
    heading = euler_y  # Rotation around the Y-axis (yaw)
    pitch = euler_x  # Rotation around the X-axis
    roll = euler_z  # Rotation around the Z-axis

    obj.setHpr(heading, pitch, roll)
    matR = p3d.LMatrix4f(obj.getMat())

    # RS = matrix * R
    # RS.invertInPlace()

    world_matrix = matR * translation_matrix
    view = world_matrix.xformVec(p3d.LVector3f(0, 0, 1))
    pos = world_matrix.xformPoint(p3d.LVector3f(0, 0, 0))
    up = world_matrix.xformVec(p3d.LVector3f(0, 1, 0))

    return computeLookAtMatrix(pos, pos + view, up)


# def get_view_matrix(pos_x, pos_y, pos_z, euler_x, euler_y, euler_z):
#
#    # Create the rotation matrix
#    mat = p3d.Mat3()
#    mat.set_hpr(euler_x, euler_y, euler_z)
#
#    # Convert to a 4x4 matrix
#    mat4 = p3d.Mat4(mat)
#    mat4.set_row(3, (0, 0, 0, 1))  # Set the bottom row
#
#    view_matrix = translation_matrix * mat4
#    view_matrix.invertInPlace()
#


def InitSVM(base, imageWidth, imageHeight):
    projMat = get_projection_matrix(fx, fy, skew_c, cx, cy, imageWidth, imageHeight, 10, 10000)
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
    for i, extrinsic in enumerate(extrinsic_parameters):
        # viewMat = get_view_matrix(**extrinsic)
        # viewProjMat = viewMat * projMat
        viewProjMat = matViews[i] * projMat

        matViewProjs[i] = viewProjMat

        base.plane.setShaderInput("matViewProj" + str(i), viewProjMat)
        # print("plane.setShaderInput")
        base.quad.setShaderInput("matViewProj" + str(i), viewProjMat)
        # print("plane.setShaderInput")

    # base.plane.setShaderInput("matViewProjs", matViewProjs)
    # base.planePnm = p3d.PNMImage()
    # for i in range(4):
    #    base.planeTexs[i].setup2dTexture(
    #        imageWidth, imageHeight, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
    #    base.plane.setShaderInput(
    #        'myTexture' + str(i), base.planeTexs[i])

    base.planeTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
    base.plane.setShaderInput("cameraImgs", base.planeTexArray)

    base.semanticTexArray.setup2dTextureArray(imageWidth, imageHeight, 4, p3d.Texture.T_int, p3d.Texture.F_r32i)
    base.plane.setShaderInput("semanticImgs", base.semanticTexArray)

    print("Texture Initialized!")


def undistort_image(image):
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([K2, K3, K4, K5]) / K1

    h, w = image.shape[:2]
    R = np.eye(3)
    P = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), R, balance=1.0)
    mapx, mapy = cv.fisheye.initUndistortRectifyMap(K, D, R, P, (w, h), cv.CV_16SC2)
    undistorted = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)

    return undistorted


if __name__ == "__main__":
    mySvm = SurroundView()

    width = mySvm.win.get_x_size()
    height = mySvm.win.get_y_size()
    print("init w {}, init h {}".format(width, height))

    InitSVM(mySvm, img_width, img_height)

    front = cv.imread("./front.png")
    right = cv.imread("./right.png")
    rear = cv.imread("./rear.png")
    left = cv.imread("./left.png")
    front = cv.resize(front, (1920, 1080))
    front4 = cv.cvtColor(
        front,
        cv.COLOR_BGR2BGRA,
    )
    # cv.imshow("TEST",front)
    # cv.waitKey(1)
    right = cv.resize(right, (1920, 1080))
    right4 = cv.cvtColor(
        right,
        cv.COLOR_BGR2BGRA,
    )
    rear = cv.resize(rear, (1920, 1080))
    rear4 = cv.cvtColor(
        rear,
        cv.COLOR_BGR2BGRA,
    )
    left = cv.resize(left, (1920, 1080))
    left4 = cv.cvtColor(
        left,
        cv.COLOR_BGR2BGRA,
    )

    # ufront = undistort_image(front)
    # uright = undistort_image(right)
    # urear = undistort_image(rear)
    # uleft = undistort_image(left)

    # cv.imwrite("front_undistort.png", ufront)
    # cv.imwrite("right_undistort.png", uright)

    front_semantic = cv.imread("./front_semantic.png")
    right_semantic = cv.imread("./right_semantic.png")
    rear_semantic = cv.imread("./rear_semantic.png")
    left_semantic = cv.imread("./left_semantic.png")
    front_semantic = cv.resize(front_semantic, (1920, 1080))
    front_semantic = cv.cvtColor(
        front_semantic,
        cv.COLOR_BGR2GRAY,
    )
    right_semantic = cv.resize(right_semantic, (1920, 1080))
    right_semantic = cv.cvtColor(
        right_semantic,
        cv.COLOR_BGR2GRAY,
    )
    rear_semantic = cv.resize(rear_semantic, (1920, 1080))
    rear_semantic = cv.cvtColor(
        rear_semantic,
        cv.COLOR_BGR2GRAY,
    )
    left_semantic = cv.resize(left_semantic, (1920, 1080))
    left_semantic = cv.cvtColor(
        left_semantic,
        cv.COLOR_BGR2GRAY,
    )

    imgnpArray = np.array([front4, right4, rear4, left4]).astype(np.uint8)
    imgArray = imgnpArray.reshape((4, 1920, 1080, 4))
    cameraArray = imgArray[:, :, :, :].copy()
    semanticArray = np.array([front_semantic, right_semantic, rear_semantic, left_semantic]).astype(np.int32)

    mySvm.planeTexArray.setRamImage(cameraArray)
    mySvm.semanticTexArray.setRamImage(semanticArray)

    mySvm.run()

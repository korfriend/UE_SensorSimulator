import threading
import socket
import struct

import numpy as np
import cv2 as cv
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from panda3d.core import Shader
import panda3d.core as p3d
import panda3d

print('Pandas Version :', panda3d.__version__)

############ Main Thread
class SurroundView(ShowBase):
    def __init__(self):
        super().__init__()
        
        # 카메라 위치 조정 
        self.cam.setPos(0, 0, 80)
        self.cam.lookAt(p3d.LPoint3f(0, 0, 0), p3d.LVector3f(0, 1, 0))
        #self.trackball.node().setMat(self.cam.getMat())
        #self.trackball.node().reset()
        #self.trackball.node().set_pos(0, 30, 0)
        
        #보트 로드
        self.boat = self.loader.loadModel("avikus_boat.glb")
        bbox = self.boat.getTightBounds()
        print(bbox)
        center = (bbox[0] + bbox[1]) * 0.5
        self.boat.setPos(-center)
        self.boat.reparentTo(self.render)
        
        self.axis = self.loader.loadModel('zup-axis')
        self.axis.setPos(0, 0, 0)
        #self.axis.setScale(.001)
        self.axis.reparentTo(base.render)
        
    def GeneratePlaneNode(self):
        #shader setting for SVM
        self.planeShader = Shader.load(
            Shader.SL_GLSL, vertex="svm_vs.glsl", fragment="svm_ps.glsl")
        vdata = p3d.GeomVertexData(
            'triangle_data', p3d.GeomVertexFormat.getV3t2(), p3d.Geom.UHStatic)
        vdata.setNumRows(4)  # optional for performance enhancement!
        vertex = p3d.GeomVertexWriter(vdata, 'vertex')
        texcoord = p3d.GeomVertexWriter(vdata, 'texcoord')

        bbox = self.boat.getTightBounds()
        waterZ = bbox[0].z + (bbox[1].z - bbox[0].z) * 0.2
        waterPlaneLength = 20
        vertex.addData3(-waterPlaneLength, waterPlaneLength, waterZ)
        vertex.addData3(waterPlaneLength, waterPlaneLength, waterZ)
        vertex.addData3(waterPlaneLength, -waterPlaneLength, waterZ)
        vertex.addData3(-waterPlaneLength, -waterPlaneLength, waterZ)
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
        self.plane = p3d.NodePath(geomNode)
        self.plane.setTwoSided(True)
        self.plane.setShader(self.planeShader)
        self.plane.reparentTo(self.render)
        
    def GeneratePointNode(self, lidarRes: int, lidarchs: int, numLidars: int):
        # note: use Geom.UHDynamic instead of Geom.UHStatic (resource setting for immutable or dynamic)
        vdata = p3d.GeomVertexData(
            'point_data', p3d.GeomVertexFormat.getV3c4(), p3d.Geom.UHDynamic)
        numMaxPoints = lidarRes * lidarchs * numLidars
        # 4 refers to the number of cameras
        vdata.setNumRows(numMaxPoints)
        vertex = p3d.GeomVertexWriter(vdata, 'vertex')
        color = p3d.GeomVertexWriter(vdata, 'color')
        vertex.reserveNumRows(numMaxPoints)
        color.reserveNumRows(numMaxPoints)
        vertex.setRow(0)
        color.setRow(0)
        for i in range(numMaxPoints):
            vertex.addData3f(0, 0, 0)
            color.addData1i(0)
        primPoints = p3d.GeomPoints(p3d.Geom.UHDynamic)
        #prim.addConsecutiveVertices(0, len(points))
        
        geom = p3d.Geom(vdata)
        geom.addPrimitive(primPoints)
        geomNode = p3d.GeomNode("Lidar Points")
        geomNode.addGeom(geom)

        self.pointsVertex = vertex
        self.pointsColor = color
        self.points = p3d.NodePath(geomNode)
        self.points.setTwoSided(True)
        #self.points.setShader(self.planeShader)
        self.points.reparentTo(self.render)
        
mySvm = SurroundView()

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

def ReceiveData():
    # Listen for incoming datagrams
    while(True):

        print("listening")
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        print("got it")
        
        packetInit = bytesAddressPair[0]
        addressInit = bytesAddressPair[1]

        #clientMsg = "Message from Client:{}".format(message)
        clientIP = "Client IP Address:{}".format(addressInit)

        index = int.from_bytes(packetInit[0:4], "little")
        
        if index == 0xffffffff:
            print("packet start")
            packetNum = int.from_bytes(packetInit[4:8], "little")
            bytesPoints = int.from_bytes(packetInit[8:12], "little")
            bytesDepthmap = int.from_bytes(packetInit[12:16], "little")
            bytesRGBmap = int.from_bytes(packetInit[16:20], "little")
            print(("Num Packets : {Num}").format(Num=packetNum))
            print(("Bytes of Points : {Num}").format(Num=bytesPoints))
            print(("Bytes of RGB map : {Num}").format(Num=bytesRGBmap))
            print(("Bytes of Depth map : {Num}").format(Num=bytesDepthmap))
            
            if packetNum == 0:
                UDPServerSocket.sendto(bytesToSend, addressInit)
                continue
            
            UDPServerSocket.sendto(bytesToSend, addressInit)

            fullPackets = bytearray(b'')
            packetIndex = 0
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
                
            #print(("Bytes of All Packets: {d}").format(d=len(fullPackets)))
            # to do 
            # those information will be loaded from the UDP packet
            numPoints = bytesPoints / 16
            numLidars = 4
            lidarRes = 100
            lidarchs = 32
            imageWidth = 256
            imageHeight = 256

            if mySvm.isInitializedUDP == True:
                mySvm.isInitializedUDP = False

                mySvm.GeneratePlaneNode()
                mySvm.GeneratePointNode(lidarRes, lidarchs, numLidars)
                
                #mySvm.planePnm = p3d.PNMImage()
                mySvm.planeTexs = [p3d.Texture(), p3d.Texture(), p3d.Texture(), p3d.Texture()]
                for i in range(4):
                    mySvm.planeTexs[i].setup2dTexture(
                        imageWidth, imageHeight, p3d.Texture.T_unsigned_byte, p3d.Texture.F_rgba)
                    mySvm.plane.setShaderInput(
                        'myTexture' + str(i), mySvm.planeTexs[i])
                
                print("Texture Initialized!")
                
            # point cloud buffer : fullPackets[0:bytesPoints]
            numMaxPoints = lidarRes * lidarchs * numLidars
            mySvm.pointsVertex.setRow(0)
            mySvm.pointsColor.setRow(0)
            for i in range(numMaxPoints):
                if i < numPoints:
                    offset = 16 * i
                    pX = struct.unpack('<f', fullPackets[0 + offset : 4 + offset])[0]
                    pY = struct.unpack('<f', fullPackets[4 + offset : 8 + offset])[0]
                    pZ = struct.unpack('<f', fullPackets[8 + offset : 12 + offset])[0]
                    pC =  int.from_bytes(fullPackets[12 + offset : 16 + offset], "little")
                    mySvm.pointsVertex.setData3f(p3d.LPoint3f(pX, pY, pZ))
                    mySvm.pointsColor.setData1i(pC)
                else :
                    mySvm.pointsVertex.setData3f(0, 0, 0)
                    mySvm.pointsColor.setData1i(0)
                    
            # depth buffer : fullPackets[bytesPoints:bytesPoints + bytesDepthmap] ... 4 of (lidarRes * lidarchs * 4 bytes) 

            offsetColor = bytesPoints + bytesDepthmap
            imgBytes = imageWidth * imageHeight * 4
            imgs = []
            for i in range(4):
                imgnp = np.array(
                    fullPackets[offsetColor + imgBytes * i: offsetColor + imgBytes * (i + 1)], dtype=np.uint8)
                img = imgnp.reshape((imageWidth, imageHeight, 4))
                imgs.append(img)
                # https://docs.panda3d.org/1.10/python/programming/texturing/simple-texturing
                # https://docs.panda3d.org/1.10/cpp/programming/advanced-loading/loading-resources-from-memory
                mySvm.planeTexs[i].setRamImage(img)

            cv.imshow('image_deirvlon 0', imgs[0])
            cv.imshow('image_deirvlon 1', imgs[1])
            cv.imshow('image_deirvlon 2', imgs[2])
            cv.imshow('image_deirvlon 3', imgs[3])
            cv.waitKey(1)
            

    #print(("Packets : {p0}, {p1}, {p2}, {p3}").format(p0=packet[0], p1=packet[1], p2=packet[2], p3=packet[3]))
    #print(index)
    #print(clientIP)

    # Sending a reply to client

t = threading.Thread(target=ReceiveData, args=())
t.start()

print("\n\nSVM Start!")
mySvm.run()

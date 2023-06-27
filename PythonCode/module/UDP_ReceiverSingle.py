import math
import queue
import socket
import time

import cv2 as cv
import DepthToPoint
import numpy as np

color_map = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
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


def ReceiveData(packetInit: dict, q: queue):
    localIP = "127.0.0.1"
    localPort = 12000
    bufferSize = 60000

    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    UDPServerSocket.bind((localIP, localPort))
    # timeout = 5
    # UDPServerSocket.settimeout(timeout)

    packetDict = {}

    while True:
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)

        packet = bytesAddressPair[0]

        frame = int.from_bytes(packet[0:4], "little")
        count = int.from_bytes(packet[4:8], "little")

        if frame == 0xFFFFFFFF:  # initial packet
            # to do
            # modify initial packet
            packetInit["packetNum"] = int.from_bytes(packet[4:8], "little")
            packetInit["bytesPoints"] = int.from_bytes(packet[8:12], "little")
            packetInit["bytesDepthmap"] = int.from_bytes(packet[12:16], "little")
            packetInit["bytesRGBmap"] = int.from_bytes(packet[16:20], "little")
            packetInit["numLidars"] = int.from_bytes(packet[20:24], "little")
            packetInit["lidarRes"] = int.from_bytes(packet[24:28], "little")
            packetInit["lidarChs"] = int.from_bytes(packet[28:32], "little")
            packetInit["imageWidth"] = int.from_bytes(packet[32:36], "little")
            packetInit["imageHeight"] = int.from_bytes(packet[36:40], "little")
            packetInit["Fov"] = int.from_bytes(packet[40:44], "little")

            packetInit["CameraF_y"] = int.from_bytes(packet[44:48], "little", signed=True)
            packetInit["CameraR_y"] = int.from_bytes(packet[48:52], "little", signed=True)
            packetInit["CameraB_y"] = int.from_bytes(packet[52:56], "little", signed=True)
            packetInit["CameraL_y"] = int.from_bytes(packet[56:60], "little", signed=True)

            packetInit["CameraF_location_x"] = int.from_bytes(packet[60:64], "little", signed=True)
            packetInit["CameraR_location_x"] = int.from_bytes(packet[64:68], "little", signed=True)
            packetInit["CameraB_location_x"] = int.from_bytes(packet[68:72], "little", signed=True)
            packetInit["CameraL_location_x"] = int.from_bytes(packet[72:76], "little", signed=True)

            packetInit["CameraF_location_y"] = int.from_bytes(packet[76:80], "little", signed=True)
            packetInit["CameraR_location_y"] = int.from_bytes(packet[80:84], "little", signed=True)
            packetInit["CameraB_location_y"] = int.from_bytes(packet[84:88], "little", signed=True)
            packetInit["CameraL_location_y"] = int.from_bytes(packet[88:92], "little", signed=True)

            packetInit["CameraF_location_z"] = int.from_bytes(packet[92:96], "little", signed=True)
            packetInit["CameraR_location_z"] = int.from_bytes(packet[96:100], "little", signed=True)
            packetInit["CameraB_location_z"] = int.from_bytes(packet[100:104], "little", signed=True)
            packetInit["CameraL_location_z"] = int.from_bytes(packet[104:108], "little", signed=True)

            packetInit["isFisheye"] = int.from_bytes(packet[108:112], "little", signed=True)

            print("Num Packets : {}".format(packetInit["packetNum"]))
            print("Bytes of Points : {}".format(packetInit["bytesPoints"]))
            print("Bytes of RGB map : {}".format(packetInit["bytesRGBmap"]))
            print("Bytes of Depth map : {}".format(packetInit["bytesDepthmap"]))
            print("Num Lidars : {}".format(packetInit["numLidars"]))
            print("Lidar Resolution : {}".format(packetInit["lidarRes"]))
            print("Lidar Channels : {}".format(packetInit["lidarChs"]))
            print("Camera Width : {}".format(packetInit["imageWidth"]))
            print("Camera Height : {}".format(packetInit["imageHeight"]))
            print("Camera Fov : {}".format(packetInit["Fov"]))
            print("Camera rotate y: {}, {}, {}, {}".format(packetInit["CameraF_y"], packetInit["CameraR_y"], packetInit["CameraB_y"], packetInit["CameraL_y"]))
            print("CameraF location: {}, {}, {}".format(packetInit["CameraF_location_x"], packetInit["CameraF_location_y"], packetInit["CameraF_location_z"]))
            print("CameraR location: {}, {}, {}".format(packetInit["CameraR_location_x"], packetInit["CameraR_location_y"], packetInit["CameraR_location_z"]))
            print("CameraB location: {}, {}, {}".format(packetInit["CameraB_location_x"], packetInit["CameraB_location_y"], packetInit["CameraB_location_z"]))
            print("CameraL location: {}, {}, {}".format(packetInit["CameraL_location_x"], packetInit["CameraL_location_y"], packetInit["CameraL_location_z"]))


        else:
            if not packetInit:
                continue
            # print("frame : ", frame)
            # print("packet count : ", count)
            if frame not in packetDict:
                packetDict[frame] = {}
            packetDict[frame][count] = packet[8:]
            # print("count sum : ", len(packetDict[frame]))
            for key in list(packetDict.keys()):
                packetNum = packetInit["packetNum"]
                # bytesPoints = packetInit["bytesPoints"]
                # bytesDepthmap = packetInit["bytesDepthmap"]
                # bytesRGBmap = packetInit["bytesRGBmap"]
                # numLidars = packetInit["numLidars"]
                lidarRes = packetInit["lidarRes"]
                lidarChs = packetInit["lidarChs"]
                imageWidth = packetInit["imageWidth"]
                imageHeight = packetInit["imageHeight"]
                fov = packetInit["Fov"]
                isFisheye = packetInit["isFisheye"]

                if len(packetDict[key]) == packetNum:
                    fullPackets = b"".join([packetDict[frame][i] for i in range(packetNum)])
                    fullPackets = bytearray(fullPackets)
                    imgs = []
                    segs = []
                    segr = []
                    offset = 0
                    depthMapBytes = lidarRes * lidarChs * 4

                    depthmapnp = np.array(fullPackets[offset : offset + lidarRes * lidarChs], dtype=np.float32)
                    depthmapnp = depthmapnp.reshape((lidarChs, lidarRes, 1))
                    worldpointList = DepthToPoint.toPoints(lidarChs, lidarRes, 30, 360, depthmapnp)

                    offsetImg = depthMapBytes + offset
                    imgBytes = imageWidth * imageHeight * 4
                    dummyByte = 0

                    for i in range(4):
                        imgnp = np.array(
                            fullPackets[offsetImg + dummyByte : offsetImg + dummyByte + imgBytes], dtype=np.uint8
                        )
                        segnp = np.array(
                            fullPackets[offsetImg + dummyByte + imgBytes : offsetImg + dummyByte + imgBytes + imgBytes],
                            dtype=np.uint8,
                        )
                        imgnp = imgnp.reshape((imageHeight, imageWidth, 4))

                        # if isFisheye == 1:
                        #     fx = imageWidth / math.tan(0.5 * fov) * 2
                        #     fy = imageHeight / math.tan(0.5 * fov) * 2
                        #     cx = imageWidth / 2
                        #     cy = imageHeight / 2

                        #     K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                        #     K1 = 0.0
                        #     K2 = 0.0
                        #     K3 = 0.0
                        #     K4 = 0.0
                        #     K5 = 0.0

                        #     D = np.array([K2, K3, K4, K5])
                        #     if K1 != 0:
                        #         D /= K1

                        #     # undistorted = cv.fisheye.undistortImage(distorted, K, D)
                        #     P = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        #         K, D, (imageWidth, imageHeight), np.eye(3), balance=1.0
                        #     )
                        #     map1, map2 = cv.fisheye.initUndistortRectifyMap(
                        #         K, D, np.eye(3), P, (imageWidth, imageHeight), cv.CV_16SC2
                        #     )
                        #     imgnp = cv.remap(
                        #         imgnp, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT
                        #     )

                        segnp = segnp.reshape((imageHeight, imageWidth, 4))
                        imgs.append(imgnp)

                        color_img = np.zeros_like(segnp).astype(np.uint8)
                        for j, color in enumerate(color_map):
                            for k in range(3):
                                color_img[:, :, k][segnp[:, :, 0] == j] = color[k]

                        segs.append(color_img)
                        segr.append(segnp)
                        dummyByte = dummyByte + imgBytes + imgBytes

                    # print("queue size : ", q.qsize())
                    if q.qsize() > 9:
                        q.get()
                    # print(frame)
                    # print("dic len defor : " , len(packetDict))
                    # print("send : ", key)

                    q.put([worldpointList, imgs, segs, segr])
                    del packetDict[key]
                    # print("dic len after : " , len(packetDict))
                    time.sleep(0.003)

                # else :
                # no full packet
                # continue

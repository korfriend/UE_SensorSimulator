import socket
from multiprocessing import Queue


def ReceiveData(UDPServerSocket: socket.socket, bufferSize: int, packetInit: dict, q: Queue):
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

            print("Num Packets : {}".format(packetInit["packetNum"]))
            print("Bytes of Points : {}".format(packetInit["bytesPoints"]))
            print("Bytes of RGB map : {}".format(packetInit["bytesDepthmap"]))
            print("Bytes of Depth map : {}".format(packetInit["bytesRGBmap"]))
            print("Num Lidars : {}".format(packetInit["numLidars"]))
            print("Lidar Resolution : {}".format(packetInit["lidarRes"]))
            print("Lidar Channels : {}".format(packetInit["lidarChs"]))
            print("Camera Width : {}".format(packetInit["imageWidth"]))
            print("Camera Height : {}".format(packetInit["imageHeight"]))
        else:
            # print("frame : ", frame)
            # print("packet count : ", count)
            if frame not in packetDict:
                packetDict[frame] = {}
            packetDict[frame][count] = packet[8:]
            if len(packetDict[frame]) == packetInit["packetNum"]:
                fullPackets = b"".join([packetDict[frame][i] for i in range(packetInit["packetNum"])])
                q.put(fullPackets)

            # fullPackets = bytearray(b"")
            # packetIndex = 0
            # while packetIndex < packetNum:
            #     bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            #     packet = bytesAddressPair[0]
            #     index = int.from_bytes(packet[0:4], "little")
            #     #if index != packetIndex:
            #         # print(("Error {id}").format(id=index))
            #     packetIndex += 1
            #     fullPackets += packet[4:]

            # offsetPoints = 0

            # offsetPoints += 4
            # pX = struct.unpack("<f", fullPackets[0 + offsetPoints : 4 + offsetPoints])[0]
            # pY = struct.unpack("<f", fullPackets[4 + offsetPoints : 8 + offsetPoints])[0]
            # pZ = struct.unpack("<f", fullPackets[8 + offsetPoints : 12 + offsetPoints])[0]
            # cR = int.from_bytes(fullPackets[12 + offsetPoints : 13 + offsetPoints], "little")
            # cG = int.from_bytes(fullPackets[13 + offsetPoints : 14 + offsetPoints], "little")
            # cB = int.from_bytes(fullPackets[14 + offsetPoints : 15 + offsetPoints], "little")
            # cA = int.from_bytes(fullPackets[15 + offsetPoints : 16 + offsetPoints], "little")
            # # print(pX, pY, pZ)

            # offsetPoints += 16

            # offsetColor = bytesPoints + bytesDepthmap
            # imgBytes = imageWidth * imageHeight * 4
            # imgs = []

            # for i in range(4):
            #     imgnp = np.array(
            #         fullPackets[offsetColor + imgBytes * i : offsetColor + imgBytes * (i + 1)], dtype=np.uint8
            #     )
            #     imgnp = imgnp.reshape((imageWidth, imageHeight, 4))
            #     imgs.append(imgnp)

            # cv.imshow("image_deirvlon 0", imgs[0])
            # cv.imshow("image_deirvlon 1", imgs[1])
            # cv.imshow("image_deirvlon 2", imgs[2])
            # cv.imshow("image_deirvlon 3", imgs[3])

            # cv.waitKey(1)

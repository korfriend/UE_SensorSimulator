import socket

# import time
from multiprocessing import Manager, Process, Queue

import cv2 as cv
import numpy as np
import UDP_Receiver


def func(packetInit: dict, q: Queue):
    while True:
        if q.empty():
            continue

        fullPackets = bytearray(q.get())

        offsetDepth = 0
        depthMapBytes = packetInit["lidarRes"] * packetInit["lidarChs"] * 4
        depthmapnp = np.array(
            fullPackets[offsetDepth : offsetDepth + packetInit["lidarRes"] * packetInit["lidarChs"]], dtype=np.float32
        )
        depthmapnp = depthmapnp.reshape((packetInit["lidarChs"], packetInit["lidarRes"], 1))

        offsetColor = depthMapBytes
        imgBytes = packetInit["imageWidth"] * packetInit["imageHeight"] * 4
        imgs = []

        for i in range(4):
            imgnp = np.array(fullPackets[offsetColor + imgBytes * i : offsetColor + imgBytes * (i + 1)], dtype=np.uint8)
            imgnp = imgnp.reshape((packetInit["imageWidth"], packetInit["imageHeight"], 4))
            imgs.append(imgnp)

        # imgs[3][50:100, 50:100, 0:3] = time.localtime().tm_sec % 255

        cv.imshow("depth_derivlon", depthmapnp)
        cv.imshow("image_deirvlon 0", imgs[0])
        cv.imshow("image_deirvlon 1", imgs[1])
        cv.imshow("image_deirvlon 2", imgs[2])
        cv.imshow("image_deirvlon 3", imgs[3])

        cv.waitKey(1)


if __name__ == "__main__":
    localIP = "127.0.0.1"
    localPort = 12000
    bufferSize = 60000

    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    UDPServerSocket.bind((localIP, localPort))

    manager = Manager()
    packetInit = manager.dict()
    q = Queue(maxsize=0)

    print("UDP server up and listening")
    t1 = Process(target=UDP_Receiver.ReceiveData, args=(UDPServerSocket, bufferSize, packetInit, q))
    t2 = Process(target=func, args=(packetInit, q))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

import socket

import threading
import queue
import cv2 as cv
import numpy as np
import UDP_Receiver
import time 


def func(packetInit: dict, q: queue):

    while True:
        if q.empty():
            print("텅텅")
            time.sleep(0.02)
            #continue

        fullPackets = bytearray(q.get())
        
        index2 = int.from_bytes(fullPackets[0:4], "little")
        print("index : ",index2)
        #print(len(fullPackets))
        offsetDepth = 4
        depthMapBytes = packetInit["lidarRes"] * packetInit["lidarChs"] * 4
        depthmapnp = np.array(
            fullPackets[offsetDepth : offsetDepth + packetInit["lidarRes"] * packetInit["lidarChs"]], dtype=np.float32
        )
        depthmapnp = depthmapnp.reshape((packetInit["lidarChs"], packetInit["lidarRes"], 1))

        offsetColor = depthMapBytes + offsetDepth
        imgBytes = packetInit["imageWidth"] * packetInit["imageHeight"] * 4
        imgs = []
        for i in range(4):
            imgnp = np.array(fullPackets[offsetColor + imgBytes * i : offsetColor + imgBytes * (i + 1)], dtype=np.uint8)
            imgnp = imgnp.reshape((packetInit["imageWidth"], packetInit["imageHeight"], 4))
            imgs.append(imgnp)

        cv.imshow("depth_derivlon", depthmapnp)
        cv.imshow("image_deirvlon 0", imgs[0])
        cv.imshow("image_deirvlon 1", imgs[1])
        cv.imshow("image_deirvlon 2", imgs[2])
        cv.imshow("image_deirvlon 3", imgs[3])
        cv.waitKey(1)

        

if __name__ == "__main__":
 
    packetInit = dict()
    q = queue.Queue(maxsize=10)

    print("UDP server up and listening")
    t1 = threading.Thread(target=UDP_Receiver.ReceiveData, args=(packetInit, q))
    t2 = threading.Thread(target=func, args=(packetInit, q))

    t1.start()
    time.sleep(2)
    print("func2 start")
    t2.start()

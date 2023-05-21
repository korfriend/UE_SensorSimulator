import threading
import queue
import cv2 as cv
import numpy as np
import UDP_Receiver
import time 
import DepthToPoint 
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
from direct.filter.FilterManager import FilterManager

import json
import keyboard

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from direct.showbase.ShowBase import ShowBase
import numpy as np
from panda3d.core import *
from panda3d.core import Point3, Material
from panda3d.core import GeomNode
from panda3d.core import Geom, GeomVertexData, GeomVertexFormat, GeomVertexWriter, GeomPoints

from draw_pointcloud import MyGame
        
game = MyGame()

def func(packetInit: dict, q: queue):

    while True:
        if q.empty():
            # print("텅텅")
            time.sleep(0.02)
            #continue

        fullPackets = bytearray(q.get())
        
        index2 = int.from_bytes(fullPackets[0:4], "little")
        #print("index : ",index2)

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

        if len(depthmapnp) != 0 :
            worldpointList = DepthToPoint.toPoints(packetInit["lidarChs"],packetInit["lidarRes"],
                                30,360 ,depthmapnp)
            game.vertice = worldpointList
        

if __name__ == "__main__":
   
    packetInit = dict()
    q = queue.Queue(maxsize=10)

    print("UDP server up and listening")
    t1 = threading.Thread(target=UDP_Receiver.ReceiveData, args=(packetInit, q))
    t2 = threading.Thread(target=func, args=(packetInit, q))
#   t3 = threading.Thread(target=game.run, args=(packetInit, q))

    t1.start()
    time.sleep(2)
    print("func2 start")
    t2.start()
    game.run()

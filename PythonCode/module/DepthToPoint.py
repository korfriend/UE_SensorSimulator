import numpy as np
import math


def toPoints(channel : int , res : int, vfov : float, hfov : float , depthmap : np.array()):
    point =  []
    print(depthmap.shape)
    for ch in range(channel):
        for r in range(res):
            raydepth = depthmap[ch][r]
            vangle = ch*(vfov / channel)
            hangle = r*(hfov / res)
            rayPointX = raydepth * math.sin(hangle) * math.cos(vangle) 
            rayPointY = raydepth * math.sin(hangle) * math.sin(vangle) 
            rayPointZ = raydepth * math.cos(hangle)
            
            Point = [rayPointX,rayPointY,rayPointZ]
            point.append(Point)






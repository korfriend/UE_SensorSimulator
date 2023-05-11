import numpy as np
import math


# def toPoints(channel : int , res : int, vfov : float, hfov : float , depthmap : np.array()):
#     point =  []
#     print(depthmap.shape)
#     for ch in range(channel):
#         for r in range(res):
#             raydepth = depthmap[ch][r]
#             vangle = ch*(vfov / channel)
#             hangle = r*(hfov / res)
#             rayPointX = raydepth * math.sin(hangle) * math.cos(vangle) 
#             rayPointY = raydepth * math.sin(hangle) * math.sin(vangle) 
#             rayPointZ = raydepth * math.cos(hangle)
            
#             Point = [rayPointX,rayPointY,rayPointZ]
#             point.append(Point)
def toPoints(channel: int, res: int, vfov: float, hfov: float, depthmap: np.array):
    point = []
    #print(depthmap.shape)
    for ch in range(channel):
        for r in range(res):
            raydepth = depthmap[ch][r]
            vangle = math.radians(ch * (vfov / channel) - (vfov/2))  # vfov 범위를 -15 ~ 15도로 조정
            hangle = math.radians(r * (hfov / res))
            
            projectRaydepth = raydepth*math.cos(vangle)

            rayPointX =projectRaydepth*math.cos(hangle)
            rayPointY =projectRaydepth*math.sin(hangle)
            rayPointZ =projectRaydepth*math.sin(vangle)
            '''
            vangle = math.radians(90 - ch * (vfov / channel) - (vfov/2))  # vfov 범위를 -15 ~ 15도로 조정
            hangle = math.radians(r * (hfov / res))
            rayPointX = raydepth * math.sin(vangle) * math.cos(hangle)
            rayPointY = raydepth * math.sin(vangle) * math.sin(hangle)
            rayPointZ = raydepth * math.cos(vangle)
            '''
            

            Point = [(rayPointX, rayPointY, rayPointZ)]
            point.append(Point)
    return point





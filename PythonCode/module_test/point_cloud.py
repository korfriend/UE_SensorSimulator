from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from panda3d.core import Point3, Material
from panda3d.core import GeomNode
from panda3d.core import Geom, GeomVertexData, GeomVertexFormat, GeomVertexWriter, GeomPoints

def draw_pointCloud(self, vertices, colors):
    
    points = [Point3(p[0], p[1], p[2]) for p in vertices]

    # Point cloud 노드 생성
    pointCloudNode = GeomNode("pointCloud")
    # Geom 생성
    vformat = GeomVertexFormat.getV3c4() #아마도 Vector3, Color4 같은 느낌
    vdata = GeomVertexData('point_data', vformat, Geom.UHStatic) #Geom.UHDynamic 으로 바꾸어 주어야 원하는 대로 수정이 가능한것으로
    vertex = GeomVertexWriter(vdata, 'vertex')
    color = GeomVertexWriter(vdata, 'color')
    
    for point, inputColor in zip(points, colors):
        vertex.addData3f(point)
        color.addData4f(inputColor[0], inputColor[1], inputColor[2], 1.0)
        
    prim = GeomPoints(Geom.UHStatic)
    prim.addConsecutiveVertices(0, len(points))
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    pointCloudNode.addGeom(geom)
    
    # Point cloud 노드에 적용할 머티리얼 생성
    material = Material()
    material.setShininess(1000)
    
    # Point cloud 노드를 렌더링하기
    nodePath = self.render.attachNewNode(pointCloudNode)
    nodePath.setMaterial(material)
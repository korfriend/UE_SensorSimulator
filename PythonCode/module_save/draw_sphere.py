from direct.task import Task
from panda3d.core import Geom, GeomVertexData, GeomVertexFormat, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from panda3d.core import Point3
import math
from panda3d.core import Camera
from panda3d.core import LPoint3f
from panda3d.core import NodePath, LPoint3
from panda3d.core import PerspectiveLens, LVector3
from panda3d.core import Shader



def draw_sphere(self, radius,pos,color):
    subdivisions = 32
    
    # 구체의 정점 데이터를 생성합니다.
    format = GeomVertexFormat.getV3()
    vdata = GeomVertexData('sphere', format, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, 'vertex')
    for i in range(subdivisions):
        for j in range(subdivisions):
            theta = (i / subdivisions) * 2 * math.pi
            phi = (j / subdivisions) * math.pi
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vertex.addData3f(x, y, z)

    # 구체의 삼각형 데이터를 생성합니다.
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(subdivisions):
        for j in range(subdivisions):
            a = i * subdivisions + j
            b = (i + 1) % subdivisions * subdivisions + j
            c = i * subdivisions + (j + 1) % subdivisions
            tris.addVertices(a, b, c)
            tris.closePrimitive()

            a = (i + 1) % subdivisions * subdivisions + j
            b = (i + 1) % subdivisions * subdivisions + (j + 1) % subdivisions
            c = i * subdivisions + (j + 1) % subdivisions
            tris.addVertices(a, b, c)
            tris.closePrimitive()

    # 구체의 노드를 생성하고, 삼각형과 정점 데이터를 추가합니다.
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode('sphere')
    node.addGeom(geom)
    
    # 구체를 렌더링합니다.
    #self.sphere = self.render.attachNewNode(node) # this will be handled in main class code
    self.sphere = NodePath(node)
    self.sphere.set_color(color)
    self.sphere.setPos(pos)
    self.sphere.setTwoSided(True)
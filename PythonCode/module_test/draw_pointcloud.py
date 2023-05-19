from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from panda3d.core import Shader
from draw_sphere import draw_sphere
import panda3d.core as p3d
from point_cloud import draw_pointCloud
import numpy as np
from panda3d.core import LMatrix4f, LVector3f



class MyGame(ShowBase):
    configVars = """
    win-size 1280 720
    show-frame-rate-meter 1
    """

    loadPrcFileData("", configVars)
    def __init__(self):
        super().__init__()
        
        # 카메라 각각의 좌표계를 설정
        translation = LVector3f(1, 2, 3)
        translation_matrix = LMatrix4f.translate_mat(translation) 
        camera_front = p3d.NodePath("ChildNode")
        camera_front.reparent_to(self.render)
        # camera_front.setPos(2,2,2)
        # camera_front.set_hpr(0, 0, 45)
        camera_front.set_mat(translation_matrix)
        # self.boat.reparentTo(camera_front)
        
        
        # 정점 정보 생성
        vertices = np.random.randn(30000, 3).astype(np.float32)
        colors = np.random.uniform(0.0, 1.0, size=(300000, 3)).astype(np.float32) # 무작위 색상
#        draw_pointCloud(self, vertices, colors)
        
        ############################### 위쪽 부분은 point cloud 를 생성하기 위한 파트.#########################
        
        # 카메라 위치 조정 
        self.cam.setPos(0, -40, 0)
        
        self.vertices = vertices
        
        #보트 로드
        self.boat = self.loader.loadModel("avikus_boat.glb")
        self.boat.setPos(0,0,1.05)
        # self.boat.reparentTo(self.render)
        self.boat.reparentTo(camera_front)
        self.taskMgr.add(self.update_vertices_task, "update_vertices_task")

    def update_vertices_task(self, task) :
        # Update vertices values
        self.colors = np.random.uniform(0.0, 1.0, size=(300000, 3)).astype(np.float32) 
        draw_pointCloud(self,self.vertices, self.colors)
    
        return task.cont

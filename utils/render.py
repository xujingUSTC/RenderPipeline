import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Iterable

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, RasterizationSettings, OpenGLPerspectiveCameras, HardPhongShader, lighting
from pytorch3d.renderer import PointLights, MeshRenderer, MeshRasterizer

from utils.helper import to_np, to_tensor

def get_quick_render(device):
    default_camera = OpenGLPerspectiveCameras(device=device)
    default_lights = PointLights(device=device)
    raster_setting = RasterizationSettings(image_size=256)
    render = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_setting,
            cameras = default_camera
        ),
        shader = HardPhongShader(
            device=device,
            lights = default_lights,
            cameras = default_camera
        )
        
    )
    return render

def get_mesh(verts, faces, device, textures=None):
    '''
    textures要求是和verts相同形状的numpy数组
    '''
    if len(verts.shape) == 2:
        verts = verts[None, ...]
    if len(faces.shape) == 2:
        faces = faces[None, ...]

    if textures is None:
        textures = torch.ones_like(verts).to(device)
        textures[..., 0]=255
        textures[..., 1]=255
        textures[..., 2]=0
    textures = TexturesVertex(verts_features=textures)
    mesh = Meshes(verts, faces, textures = textures)
    return mesh

class Render(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        #定义SH光照的常数项
        pi = np.pi
        constant_factor = torch.tensor(#前二次的球谐系数，这里选择FLAME框架里面的
            [1 / np.sqrt(4 * pi), 

            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), 
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), 
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),

            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), 
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), 

            (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))])
        self.register_buffer('constant_factor', constant_factor)

    def world2camera(self, 
                    location=((0, 0, 0),), 
                    direction=((0,0,-1),), 
                    up=((0,1,0),)
        ):
        '''
        计算从世界坐标系到camera坐标系的变换矩阵，首先计算平移变换的矩阵，然后旋转变换，这里使用齐次坐标计算
        这里相机看向的方向是-z方向，因此这里需要计算的变换矩阵是direction -> -z; up -> y; tile -> x. tile理应由direction 叉乘得到
        #TODO: check正确与否
        '''
        location = to_tensor(to_np(location)).to(self.device)* -1
        location = torch.cat([location, torch.ones([location.shape[0], 1]).to(location.device)], dim=-1)
        B = location.shape[0]
        T = torch.eye(4).to(self.device).unsqueeze(0).repeat(B, 1, 1)
        T[:, :, -1] = location #(B, 4, 4)

        #相机坐标系的坐标轴
        z_axis = F.normalize(to_tensor(to_np(direction)).to(self.device) * -1)#(B, 3)
        y_axis = F.normalize(to_tensor(to_np(up)).to(self.device))
        x_axis = F.normalize(torch.cross(y_axis, z_axis))
        
        R = torch.cat([x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]], dim=1)#(B, 3, 3)
        R_pad = torch.eye(4).to(self.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
        R_pad[:, :3, :3] = R

        if R_pad.shape[0] != B:
            assert R_pad.shape[0] == 1
            R_pad = R_pad.repeat(B, 1, 1)

        return torch.bmm(R_pad, T)

    def view2cube(self,
        z_near,
        z_far,
        fovY,
        aspect_ratio       
    ):
        '''
        涉及到投影变换的参数设定，主要包括，
            z_near: 近平面距离相机的距离，
            z_far: 远平面距离相机的距离
            fovY: 垂直的视角,用pi的形式表示
            aspect_ratio：纵横比, y/x
        #TODO: 修改成batch的形式？在单目重建的时候，每张图片的设定是不一样的，如果是在之前的虚拟人的场景的建模倒是需要batch的形式
        #TODO：这里x、y的顺序，
        '''
        #首先计算M_persp2ortho
        M_persp2ortho = np.array([[-z_near, 0, 0, 0],
                                  [0, -z_near, 0, 0],
                                  [0, 0, -z_near-z_far, -z_near*z_far],
                                  [0, 0, 1, 0]])

        M_persp2ortho = to_tensor(M_persp2ortho).to(self.device)

        fovY = fovY * np.pi / 180

        y_near = np.tan(fovY/2) * z_near * 2
        x_near = y_near * aspect_ratio

        #因为是看向-z方向
        z_center = - (z_far + z_near) / 2


        #完成frustum到cube的变换之后，计算正交投影， 压缩到标准立方体内部
        ## 将cube平移到以原点为中心，然后对三个维度分别进行缩放
        M_ortho_R = np.array([[2 / x_near, 0, 0, 0],
                                [0, 2 / y_near, 0, 0],
                                [0,0, 2/(z_far-z_near), 0],
                                [0, 0, 0, 1]])

        M_ortho_T = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, -z_center],
                              [0, 0, 0, 1]])

        M_ortho_R = to_tensor(M_ortho_R).to(self.device)
        M_ortho_T = to_tensor(M_ortho_T).to(self.device)

        M_ortho = torch.matmul(M_ortho_R, M_ortho_T)

        M_persp = torch.matmul(M_ortho, M_persp2ortho)
        #TODO: 这里在完成batch的实现之后删去
        M_persp = M_persp.unsqueeze(0)
        return M_persp

    def cube2screeen(
        self,
        image_size,
    ):
        '''
        从标准立方体到像素空间的变换，首先将x、y坐标从-1到1缩放到图片长宽，然后平移到width/2, height/2
        这里的视口变换应该都是一致的，因为batch里面的图片大小必须是一样的
        '''
        if isinstance(image_size, Iterable):
            height, width = image_size[0], image_size[1]
        else:
            height, width = image_size, image_size

        #x, y和height、width的对应关系是什么，来自gemaes101的代码，x、y分别对应width和height
        M_cube2screen = np.array([[width / 2, 0, 0, width/2],
                                  [0, height/2, 0, height/2],
                                  [0, 0, 1, 0],
                                  [0,0,0, 1]])
        M_cube2screen = to_tensor(M_cube2screen).to(self.device).unsqueeze(0)
        return M_cube2screen

    def __barycentric2D(self, 
        pixel, 
        face):
        '''
        face: (3, 3)
        x, y: float
        计算点x, y在face中的重心坐标，利用叉乘计算面积比的方法
        #TODO: 没有考虑batch，实际上应该考虑正负关系的
        '''
        A, B, C = face[0], face[1], face[2]
        P = pixel
        #(AB x AC)
        area = (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
        # AB x AP, 对应的是点C对着的那个三角形的面积
        c3 = ((B[0]-A[0])*(P[1]-A[1]) - (B[1]-A[1])*(P[0]-A[0])) / area
        # AP x AC，对应点B对着的三角形的面积
        c2 = ((P[0]-A[0])*(C[1]-A[1]) - (P[1]-A[1])*(C[0]-A[0])) / area
        #CP x CB 对应点A对着的三角形的面积，如果直接计算的话，可能会出现相加不为1的情况  
        c1 = ((P[0]-C[0])*(B[1]-C[1]) - (P[1]-C[1])*(B[0]-C[0])) / area

        assert c2 >= 0 and c3 >= 0 and c1 >= 0, (c1, c2, c3)

        return torch.stack([c1,c2,c3])    

    def __insideTriangle(self,
        pixel,
        face
    ):
        '''
        计算点是否在三角形内部
        #TODO: 没有抗锯齿，抗锯齿的实现需要计算覆盖的面积
        #TODO：没有考虑batch
        '''
        A, B, C = face[0], face[1], face[2]
        P = pixel
        #PA x PB
        t1 = (P[0]-A[0])*(P[1]-B[1]) - (P[0]-B[0])*(P[1]-A[1])
        #PB x PC
        t2 = (P[0]-B[0])*(P[1]-C[1]) - (P[0]-C[0])*(P[1]-B[1])
        #PC x PA
        t3 = (P[0]-C[0])*(P[1]-A[1]) - (P[0]-A[0])*(P[1]-C[1])

        #判断三个是否同号
        return ((t1 * t2) >= 0) and ((t2 * t3) >=0)
        
        
    def rasterize(self, 
        image_size,
        num_faces_per_pixel,
        transed_verts,#(B, V, 3) 
        faces#(B, F, 3)
    ):
        '''
        光栅化，输出应该是(B, width, height, k, 4)，根据transed_verts的结果，找到最靠近的k个面，以及每个面的重心坐标
        算法是，初始化两张图，一张z_buffer，另一张是color图
        '''
        B, F = faces.shape[0], faces.shape[1]
        faces = faces.view(B, -1, 1).expand(-1, -1, 3)#(B, F*3, 3)
        faces_verts = transed_verts.gather(1, faces.long()).view(B, -1, 3, 3)#(B, F, 3, 3)

        if isinstance(image_size, Iterable):
            height, width = image_size[0], image_size[1]
        else:
            height, width = image_size, image_size
        
        z_buffer = torch.ones([B, height, width, num_faces_per_pixel, 1]).to(self.device) * -1000 #乘以10表示无穷远
        pixel_to_face = torch.zeros([B, height, width, num_faces_per_pixel, 4]).to(self.device)
        
        #光栅化计算
        #TODO: 暂时为了简单，去掉batch的维度，假设batch只会是1

        #这个过程很长
        from tqdm import tqdm
        for i in tqdm(range(F)):
            #首先计算boundary
            cur_face = faces_verts[0][i]#(3, 3)
            x_min, x_max = cur_face[:, 0].min().int(), cur_face[:, 0].max().int()
            y_min, y_max = cur_face[:, 1].min().int(), cur_face[:, 1].max().int()

            x_min = max(x_min, 0)
            x_max = min(width-1, x_max)
            y_min = max(y_min, 0)
            y_max = min(y_max, height-1)



            if x_min>=x_max or y_min>=y_max:
                continue
            
            #注意，如果不加1的话可能会产生孔洞
            for x in range(x_min, x_max+1):
                for y in range(y_min, y_max+1):
                    inside = self.__insideTriangle((x, y), cur_face)
                    if inside:
                        #如果在内部，计算这个点在这个面内部的重心坐标
                        bary_cord = self.__barycentric2D((x, y), cur_face)

                        #计算这个face给予这个点的深度
                        z_value = bary_cord[0] * cur_face[0, -1] + bary_cord[1] * cur_face[1, -1] + bary_cord[2] * cur_face[1, -1]

                        #TODO: 为了简单起见，假设每个像素只对应到一个face上
                        if z_value > z_buffer[0, x, y, 0, 0]:
                            #则更新
                            z_buffer[0, x, y, 0, 0] = z_value
                            pixel_to_face[0, x, y, 0, 0] = i
                            pixel_to_face[0, x, y, 0, 1:] = bary_cord

        pixel_to_face = torch.cat([pixel_to_face, z_buffer], dim=-1)#(B, H, W, k, 5), 5个数分别是face的id，重心坐标和z值, 实际上这里的z值可能不需要保存到下一步
        return pixel_to_face


    def texturing(self,
        verts_uv,#(B, V, 2)
        faces_uv,#(B, F, 3)
        uv_map,#(B, 3, H, W)
        pixel2face,#(B, height, width, k)
        barycoord,#(B, height, width, k, 3)
        visible_mask
    ):
        '''
        shading函数，首先根据重心坐标，计算每个像素的uv坐标, 然后在uv图上sample，得到一张albdo的图片
        '''
        B = faces_uv.shape[0]
        height, width = pixel2face.shape[1], pixel2face.shape[2]

        #uv map的坐标在0,1之间，但是grid_sample的grid取值在-1,1之间，因此缩放一下，可能会导致一些问题
        verts_uv = (verts_uv * 2) - 1

        faces_uv = faces_uv.view(B, -1, 1).expand(-1, -1, 2)
        faces_uv_cord = verts_uv.gather(1, faces_uv.long()).view(B, -1, 3, 2)#(B, F, 3, 2)

        pixel_uv = pixel2face.view(B, -1, 1, 1).expand(-1, -1, 3, 2)
        pixel_uv = faces_uv_cord.gather(1, pixel_uv.long()).view(B, height, width, 3, 2)#(B, height, width, 3, 2)

        pixel_uv = (pixel_uv * (barycoord.squeeze(-2).unsqueeze(-1))).sum(-2)#(B, height, width, 2)

        #从uv map上sample
        albedo = F.grid_sample(uv_map, grid=pixel_uv)#(B, 3, H, W)
        visible_mask = visible_mask.squeeze(-1)[:, None, :, :] #(B, H, W)
        albedo = albedo * visible_mask
        return albedo, pixel_uv, faces_uv_cord

    def __verts_normal(self,
        verts,#（B, V, 3）
        faces # (B, F, 3)
    ):
        '''
        normal是空间属性，使用重心坐标计算的话是否需要将像素的位置逆向变换到空间上呢？
        '''
        #首先找到每个face对应的顶点的坐标(B, F, 3, 3)
        B, V = verts.shape[0], verts.shape[1]
        verts_normals = torch.zeros([B, V, 3]).to(self.device)

        faces = faces + (torch.arange(B, dtype=torch.int32).to(self.device) * V).view(-1, 1, 1)
        verts = verts.view(-1, 3)#(B*V, 3)
        faces_verts = verts[faces.long()]#(B, F, 3, 3)

        #对每个顶点，其normal等于所在面的normal根据面积的加权
        faces = faces.reshape(-1, 3).long()
        faces_verts = faces_verts.reshape(-1, 3, 3)
        verts_normals = verts_normals.reshape(-1, 3)
        verts_normals.index_add_(0, faces[:, 0],
                                    torch.cross(faces_verts[:, 1]-faces_verts[:, 0], faces_verts[:, 2]-faces_verts[:, 0]))
        verts_normals.index_add_(0, faces[:, 1],
                                    torch.cross(faces_verts[:, 2]-faces_verts[:, 1], faces_verts[:, 0]-faces_verts[:, 1]))
        verts_normals.index_add_(0, faces[:, 2], 
                                    torch.cross(faces_verts[:, 0]-faces_verts[:, 2], faces_verts[:, 1]-faces_verts[:, 2]))

        verts_normals = F.normalize(verts_normals, dim=1, eps=1e-6)

        verts_normals = verts_normals.reshape(-1, V, 3)

        verts_normals = verts_normals.reshape(B*V, 3)
        faces = faces.reshape(B, -1, 3)
        faces_normals = verts_normals[faces.long()]#(B, F, 3, 3)

        verts_normals = verts_normals.reshape(B, V, 3)
        return verts_normals, faces_normals


    def lighting(self,
        pixel_normal,#(B, 3, H, W)
        lighting_params,#(B, 9, 3)
        type
    ):
        '''
        type: SH, point, directional
        lighting_params: 对于SH，应该是(B, 9, 3); 
                        对于point light，则是三个通道的intensity，和光源的location，这里需要知道像素对应的空间位置
                        对于directional，则是三个通道的intensity，和光原的direction
        观测角度是-z方向，从shading point出发就是+z方向
        '''
        if type == 'SH':
            N = pixel_normal#(N, 3, H, W)
            sh = torch.stack([
                N[:, 0] * 0. + 1., 
                N[:, 0], 
                N[:, 1], 
                N[:, 2], 
                N[:, 0] * N[:, 1], 
                N[:, 0] * N[:, 2],
                N[:, 1] * N[:, 2], 
                N[:, 0] ** 2 - N[:, 1] ** 2, 
                3 * (N[:, 2] ** 2) - 1
            ],
                1)  # [bz, 9, h, w]
            # print(N.shape, sh.shape)
            sh = sh * self.constant_factor[None, :, None, None]#函数值乘以系数
            # import ipdb; ipdb.set_trace()
            # print(sh.shape, lighting_params.shape)
            shading = torch.sum(lighting_params[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
            return shading #(B, 3, H, W)
        elif type == 'point':
            raise NotImplementedError("I dont know how to implement")
        elif type == 'directional':
            #仅仅考虑漫反射和环境光，高光项涉及到材质的问题
            I_d = lighting_params[:, :3]#(B, 3)，漫反射系数
            I_a = lighting_params[:, 3:6]#(B, 3), 环境光系数
            direction = lighting_params[:, 6:]#(B, 3)

            direction = F.normalize(direction, dim=1).view(-1, 3, 1, 1)#(B, 3)
            #Bling-Phong lighting
            pixel_normal = F.normalize(pixel_normal, dim=1)#(B, 3, H, W)
            dot = (direction * pixel_normal).sum(1).unsqueeze(1)#(B, 3, H, W)
            diffuse_color = (I_d.view(-1, 3, 1, 1)) * dot #(B, 3, H, W)
            ambient_color = I_a.view(-1, 3, 1, 1)
            shading = diffuse_color + ambient_color
            return shading


    def forward(self, 
        verts,#(B, V, 3) 
        faces,#(B, F, 3)
        verts_uv,#(B, V, 2)
        faces_uv,#(B, F, 3)
        image_size,
        num_faces_per_pixel,
        uv_map, #(B, 3, H, W)
        lighting_params,
        lighting_type,
        #camera的参数也太多了，这部分要参考其他的fitting代码
        z_near,
        z_far,
        fovY,
        aspect_ratio,
        lighting_direction=None,
        camera_location=((0, 0, 0),),
        camera_direction=((0,0,-1),),
        camera_up=((0,1,0),),

     ):
        assert uv_map.shape[1] == 3, "channel first"
        #首先从世界坐标向相机坐标变换
        B = verts.shape[0]

        world2view_m = self.world2camera(
            location=camera_location,
            direction=camera_direction,
            up=camera_up
        )#(B, 4, 4)
        if world2view_m.shape[0] != B:
            assert world2view_m.shape[0] == 1
            world2view_m = world2view_m.repeat(B, 1, 1)
        
        #转换成齐次坐标
        verts = torch.cat([verts, torch.ones([B, verts.shape[1], 1]).to(self.device)], dim=-1)#(B, V, 4)
        verts = verts.permute(0, 2, 1)#(B, 4, V)

        #计算变换后的矩阵
        transed_verts = torch.bmm(world2view_m, verts)#(B, 4, 4) * (B, 4, V)-> (B, 4, V)
        # transed_verts = transed_verts.permute(0, 2, 1)

        view_verts = transed_verts.clone()

        #下一步是投影变换，从相机坐标系投影到标准立方体上
        M_persp = self.view2cube(z_near, z_far, fovY, aspect_ratio)#(1, 4, 4)
        if M_persp.shape[0] != B:
            assert M_persp.shape[0] == 1
            M_persp = M_persp.repeat(B, 1, 1)

        transed_verts = torch.bmm(M_persp, transed_verts)#(B, 4, V)

        transed_verts = transed_verts / (transed_verts[:, -1, :].unsqueeze(1))

        persp_verts = transed_verts.clone()

        #从标准立方体到像素空间
        M_cube2screen = self.cube2screeen(image_size)
        if M_cube2screen.shape[0] != B:
            assert M_cube2screen.shape[0] == 1
            M_cube2screen = M_cube2screen.repeat(B, 1, 1)
        transed_verts = torch.bmm(M_cube2screen, transed_verts) #(B, 4, V)

        transed_verts = transed_verts.permute(0, 2, 1)[..., :3]#(B, V, 3)

        screen_verts = transed_verts.clone()

        #光栅化，为每一个像素找到对应的face
        pixel_to_face = self.rasterize(image_size, 
                                       num_faces_per_pixel,
                                       transed_verts,
                                       faces)
        pixel_face_idx = pixel_to_face[..., 0]#(B, H, W, k)
        barycentric_coord = pixel_to_face[..., 1: 4]#(B, H, W, k, 3)
        visible_mask = (pixel_to_face[..., -1] > -10)#(B, H, W, k)

        #texture shading
        albedo, pixel_uv, faces_uv_coord = self.texturing(
            verts_uv=verts_uv,
            faces_uv=faces_uv,
            uv_map=uv_map,
            pixel2face=pixel_face_idx,
            barycoord = barycentric_coord,
            visible_mask=visible_mask
        )#(B, 3, H, W)

        #lighting
        ## 首先计算normal, 按照FLAME代码中的，这里使用的是原始空间的verts
        #TODO：使用原始的verts还是变换之后的，使用原始的话，这里的重心坐标是在像素空间计算的，是否需要反变换
        verts_normals, faces_normals = self.__verts_normal(
            verts = verts.permute(0, 2, 1)[:, :, :3],
            faces = faces
        )#(B, V, 3), (B, F, 3, 3)
        
        height, width = pixel_face_idx.shape[1], pixel_face_idx.shape[2]
        pixel_face_idx = pixel_face_idx.view(B, -1, 1, 1).expand(-1, -1, 3, 3)#(B, H*W*k, 3, 3)
        pixel_normals = faces_normals.gather(1, pixel_face_idx.long())#(B, H*W*k, 3, 3)
        pixel_normals = pixel_normals.view(B, height, width, -1, 3, 3)
        pixel_normals = (pixel_normals * barycentric_coord.unsqueeze(-1)).sum(-2)#(B, H, W, k, 3)
        print(pixel_normals.shape)
        #TODO: 暂时默认只能有一个面，FLAME的代码里面也是这么写的
        pixel_normals = pixel_normals.squeeze(-2).permute(0, 3, 1, 2)#(B, 3, H, W)

        #lighting
        shading = self.lighting(
            pixel_normal=pixel_normals,
            lighting_params=lighting_params,
            type = lighting_type
        )#(B, 3, H, W)
        
        #最后的输出是shading和lighting的乘积
        output = None

        return view_verts, persp_verts, screen_verts, albedo, pixel_uv, faces_uv_coord, barycentric_coord, shading, visible_mask, faces_normals, pixel_normals


if __name__ == '__main__':
    device = torch.device('cpu')
    test_model = Render(device)
    location = ((3, 4, 5),(6, 7, 8), (9, 10, 11))
    output = test_model.world2camera(location)
    print(output.shape)
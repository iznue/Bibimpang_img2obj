from vedo import *
import aspose.threed as a3d

def obj_to_fbx(obj_dir, fbx_dir):
    scene = a3d.Scene.from_file(obj_dir)
    scene.save(fbx_dir)

def fbx_to_obj(fbx_dir, obj_dir):
    scene = a3d.Scene.from_file(fbx_dir)
    scene.save(obj_dir)

def make_thumbnail(obj_dir, texture_dir):
    mesh = Mesh(obj_dir) #obj 경로
    mesh.texture(texture_dir) #texture 경로

    # # Plotter 객체 생성
    plotter = Plotter(offscreen=True)
    
    # # 메시 추가
    plotter.add(mesh)
    
    # # 카메라 재설정
    plotter.reset_camera()
    # # plotter.roll(90)
    # # plotter.fov(30)
    # plotter.azimuth(90)
    # plotter.roll(-90)
    
    # plotter.zoom(1.4)
    plotter.zoom(1.4).show().screenshot(obj_dir+"thumb.png")

if __name__ == "__main__":
    obj_dir = 'logs/corgi_nurse_mesh.obj'
    fbx_dir = 'logs/corgi_nurse_mesh.fbx'
    t_dir = 'logs/corgi_nurse_mesh_albedo.png'

    obj_to_fbx(obj_dir, fbx_dir)
    fbx_to_obj(fbx_dir, obj_dir)
    make_thumbnail(obj_dir, t_dir)
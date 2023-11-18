from flask import Blueprint, request
from omegaconf import OmegaConf
from utils import Step1
from utils2 import Step2
from thumbnail import obj_to_fbx, obj_to_thumbnail, remove_bg
import os
from PIL import Image
import shutil

bp = Blueprint('main', __name__, url_prefix='/')

###################################### route
@bp.route('/obj')
def hello_pybo():
    return 'make_3d_obj'


@bp.route('/img2obj', methods=['GET', 'POST'])
def image_to_obj():
    ################################## 사용자로부터 파일 받아오기
    # print(request.files)
    f = request.files['user-image']
    print(f.filename)
    f_name = f.filename
    dir_name = f_name[:-4]
    img_path = f'static/image3d/{dir_name}/{f_name}'
    # img_path = 'data/img_to_obj/' + f_name[:-4] + '/' + f_name
    os.makedirs(f'static/image3d/{dir_name}', exist_ok=True)
    f.save(img_path)
    
    config_path = "/workspace/configs/image.yaml"
    
    opt = OmegaConf.load(config_path)
    
    remove_bg(img_path)
    # img_path = 'data/img_to_obj/' + f_name[:-4] + '_rm.png'
    img_path = img_path[:-4] + '_rm.png'
    opt.input = img_path
    opt.save_path = f_name[:-4]
    opt.outdir = 'static/image3d/' + f_name[:-4]
    
    print('opt.input : ' + opt.input, 'opt.save_path : ' + opt.save_path, 'opt.outdir : ' + opt.outdir)
    
    # ################################## train_1
    step1 = Step1(opt)

    step1.train(opt.iters)   
    
    # ################################## train_2
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")
        
    step2 = Step2(opt)
    
    step2.train(opt.iters_refine)

    # ################################## make_thumbnail
    # opt.outdir : folder_dir ex) data/prompt  
    # opt.mesh : obj_dir ex) data/prompt/prompt.obj
    obj_dir = opt.mesh[:-9] + '.obj'
    fbx_dir = obj_dir[:-3] + 'fbx'
    # texture_dir = obj_dir[:-4] + '_albedo.png'
    # thumbnail_dir = 'data/4-view_images/' + opt.prompt + '_generated_one.png'
    # thumbnail_dir = 'data/img_to_obj/' + f_name[:-4]

    obj_to_fbx(obj_dir, fbx_dir)
    # obj_to_thumbnail(obj_dir, texture_dir)
    # remove_bg(thumbnail_dir)
    
    # ################################## docker 파일 잠금 문제 해결하기
    # # parser : chmod 777 -R ./data
    # obj_dir = opt.mesh[:-9] + '.obj'
    # texture_dir = obj_dir[:-4] + '_albedo.png'
    os.chmod(opt.outdir+'/', 0o755)
    # os.chmod(texture_dir, 0o755)
    # os.chmod(img_path, 0o755)
    
    shutil.move(opt.outdir + '/' + opt.save_path + '.fbx', 'static/image3d/fbx')
    shutil.move(opt.outdir + '/' + opt.save_path + '_rm.png', 'static/image3d/thumb')
    shutil.move(opt.outdir + '/' + opt.save_path + '_albedo.png', 'static/image3d/texture')    
    
    return 'finish_create_img_to_obj'
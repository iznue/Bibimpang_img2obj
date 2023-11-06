from flask import Blueprint, request
from omegaconf import OmegaConf
from utils import Step1
from utils2 import Step2
from thumbnail import obj_to_fbx, obj_to_thumbnail, remove_bg
import os

bp = Blueprint('main', __name__, url_prefix='/')

###################################### route
@bp.route('/obj')
def hello_pybo():
    return 'make_3d_obj'


@bp.route('/text2obj', methods=['GET', 'POST'])
def text_to_obj():
    prompt_txt = request.get_json()
    # print(prompt)
    prompt = prompt_txt['prompt']
    print(prompt)
    
    config_path = "/workspace/configs/text_mv.yaml"
    
    opt = OmegaConf.load(config_path)
    opt.prompt = 'a cute DSLR photo of ' + prompt
    opt.save_path = prompt
    opt.outdir = 'data/'+prompt
    
    ################################## train_1
    step1 = Step1(opt)

    step1.train(opt.iters)   
    
    ################################## train_2
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")
        
    step2 = Step2(opt)
    
    step2.train(opt.iters_refine)

    ################################## make_thumbnail
    # opt.outdir : folder_dir ex) data/prompt  
    # opt.mesh : obj_dir ex) data/prompt/prompt.obj
    obj_dir = opt.mesh[:-9] + '.obj'
    fbx_dir = obj_dir[:-3] + 'fbx'
    texture_dir = obj_dir[:-4] + '_albedo.png'
    thumbnail_dir = 'data/4-view_images/' + opt.prompt + '_generated_one.png'

    obj_to_fbx(obj_dir, fbx_dir)
    # obj_to_thumbnail(obj_dir, texture_dir)
    remove_bg(thumbnail_dir)
    
    return 'finish_create_obj'
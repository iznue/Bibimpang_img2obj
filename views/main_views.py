from flask import Blueprint, request

import os
import cv2
import time
import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

bp = Blueprint('main', __name__, url_prefix='/')

###################################### route
@bp.route('/obj')
def hello_pybo():
    return 'make_3d_obj'


@bp.route('/text2obj', methods=['GET', 'POST'])
def text_to_obj():
    import argparse
    from omegaconf import OmegaConf
    from utils import Step1
    from utils2 import Step2
    
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
    
    return 'finish_create_obj'
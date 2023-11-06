from vedo import *
import aspose.threed as a3d

import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rembg

def obj_to_fbx(obj_dir, fbx_dir):
    scene = a3d.Scene.from_file(obj_dir)
    scene.save(fbx_dir)

def fbx_to_obj(fbx_dir, obj_dir):
    scene = a3d.Scene.from_file(fbx_dir)
    scene.save(obj_dir)

def obj_to_thumbnail(obj_dir, texture_dir):
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

def remove_bg(png_path):
    model = 'u2net'
    session = rembg.new_session(model)
    recenter = True

    # load image
    print(f'[INFO] loading image {png_path}...')
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    
    # carve background
    print(f'[INFO] background removal...')
    carved_image = rembg.remove(image, session=session) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    if recenter:
        print(f'[INFO] recenter...')
        final_rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = 200
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (200 - h2) // 2
        x2_max = x2_min + h2
        y2_min = (200 - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)  
    else:
        final_rgba = carved_image
    
    print('######################### path')
    out_base = os.path.splitext(os.path.basename(png_path))[0]
    out_dir = os.path.dirname(png_path)
    out_rgba = os.path.join(out_dir, out_base + '_rm.png')
    print(out_base)
    print(out_dir)
    print(out_rgba)
    print('#########################')
    # write image
    cv2.imwrite(out_rgba, final_rgba)
    print(f'[INFO] Processed image saved to {out_rgba}')


if __name__ == "__main__":
    obj_dir = 'logs/corgi_nurse_mesh.obj'
    fbx_dir = 'logs/corgi_nurse_mesh.fbx'
    t_dir = 'logs/corgi_nurse_mesh_albedo.png'

    obj_to_fbx(obj_dir, fbx_dir)
    fbx_to_obj(fbx_dir, obj_dir)
    remove_bg(obj_dir, t_dir)
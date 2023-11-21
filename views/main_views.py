from flask import Blueprint, request, session, current_app
from omegaconf import OmegaConf
from utils import Step1
from utils2 import Step2
from thumbnail import obj_to_fbx, obj_to_thumbnail, remove_bg
import os
from PIL import Image
import shutil
from config import db
from sqlalchemy import create_engine, text
import deepl

translator = deepl.Translator("a3e08092-e802-0017-285a-dc6070362a23:fx")

bp = Blueprint('main', __name__, url_prefix='/')

###################################### route
@bp.route('/obj')
def hello_pybo():
    return 'make_3d_obj'

# @bp.route('/db', methods=['GET', 'POST'])
# def dbtest():
#     data = request.get_json()

#     new_data = generate_ai_path(
#         objName=data.get('objName'),
#         fileName=data.get('fileName'),
#         fbxName=data.get('fbxName'),
#         textureName=data.get('textureName'),
#         objType=data.get('objType'),
#         MakeTimeStamp=data.get('MakeTimeStamp')
#     )
#     db.session.add(new_data)
#     db.session.commit()

#     return 'make_3d_obj'


@bp.route('/img2obj', methods=['GET', 'POST'])
def image_to_obj():
    ################################## 사용자로부터 파일 받아오기
    # print(request.files)
    f = request.files['user-image']
    # print(f.filename)
    f_name = f.filename
    # decodingFileName = base64.b64decode(file_name)
    # decodeStringFileName = decodingFileName.decode('utf-8')
    print(f_name)
    name = f_name.split('_')
    timestamp = name[0]
    obj_name = name[1][:-4]
    f_format = name[1][-4:]
    print(timestamp, obj_name, f_format)
    obj_name = translator.translate_text(obj_name, target_lang="en-us")
    # print(type(obj_name), type(f_format))
    obj_name = str(obj_name) + f_format
    print(timestamp, obj_name)
    
    dir_name = obj_name[:-4]
    print(dir_name)
    
    img_path = f'static/image3d/{dir_name}/{obj_name}'
    # img_path = 'data/img_to_obj/' + f_name[:-4] + '/' + f_name
    os.makedirs(f'static/image3d/{dir_name}', exist_ok=True)
    f.save(img_path)
    
    config_path = "configs/image.yaml"
    
    opt = OmegaConf.load(config_path)
    
    remove_bg(img_path)
    # img_path = 'data/img_to_obj/' + f_name[:-4] + '_rm.png'
    img_path = img_path[:-4] + '_rm.png'
    opt.input = img_path
    opt.save_path = obj_name[:-4]
    opt.outdir = 'static/image3d/' + obj_name[:-4]
    
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

    print(opt.mesh)
    # ################################## make_thumbnail
    # opt.outdir : folder_dir ex) data/prompt  
    # opt.mesh : obj_dir ex) data/prompt/prompt.obj
    obj_dir = opt.mesh[:-9] + '.obj'
    fbx_dir = obj_dir[:-3] + 'fbx'

    # # texture_dir = obj_dir[:-4] + '_albedo.png'
    # # thumbnail_dir = 'data/4-view_images/' + opt.prompt + '_generated_one.png'
    # # thumbnail_dir = 'data/img_to_obj/' + f_name[:-4]

    obj_to_fbx(obj_dir, fbx_dir)
    # # obj_to_thumbnail(obj_dir, texture_dir)
    # # remove_bg(thumbnail_dir)
    
    # # ################################## docker 파일 잠금 문제 해결하기
    # # # parser : chmod 777 -R ./data
    # # obj_dir = opt.mesh[:-9] + '.obj'
    # # texture_dir = obj_dir[:-4] + '_albedo.png'
    # # os.chmod(opt.outdir+'/', 0o755)
    # # os.chmod(texture_dir, 0o755)
    # # os.chmod(img_path, 0o755)
    
    print(opt.outdir, opt.save_path)

    # shutil.move(opt.outdir + '/' + opt.save_path + '.fbx', 'static/image3d/fbx')
    shutil.move(opt.outdir + '/' + opt.save_path + '_rm.png', 'static/image3d/thumb')
    shutil.move(opt.outdir + '/' + opt.save_path + '_albedo.png', 'static/image3d/texture')    
    shutil.move(opt.outdir + '/' + opt.save_path + '.fbx', 'static/image3d/fbx')

    # MySQL 데이터베이스에 정보 저장
    
    connection = current_app.database.connect()

    query = text("""
    INSERT INTO generate_ai_path (
        objName,
        fileName,
        fbxName,
        textureName,
        objType,
        MakeTimeStamp
        ) VALUES (
        :objName,
        :fileName,
        :fbxName,
        :textureName,
        :objType,
        :MakeTimeStamp
        )
    """)
    new_data = connection.execute(query, {
        'objName': obj_name,
        'fileName': '/home/meta-ai2/bibimpang_serve/static/image3d/thumb/'+opt.save_path+'_rm.png', # thumbnail
        'fbxName': '/home/meta-ai2/bibimpang_serve/static/image3d/fbx/'+opt.save_path+'.fbx',
        'textureName': '/home/meta-ai2/bibimpang_serve/static/image3d/texture/'+opt.save_path+'_albedo.png',
        'objType': '2S3',
        'MakeTimeStamp': timestamp
    })

    # Commit the transaction
    connection.commit()

    # Close the connection
    connection.close()

    return 'finish_create_img_to_obj'
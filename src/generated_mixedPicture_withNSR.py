import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import argparse
from PIL import Image
import os
from Image_Segmentation.network import U_Net
import torch.nn as nn
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel


import neural_renderer
from neural_renderer.save_obj import create_texture_image
import utils.nmr_test as nmr
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ['TORCH_USE_CUDA_DSA'] = '1'
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--obj", type=str, default='car_assets/audi_et_te.obj')
parser.add_argument("--faces", type=str, default='car_assets/exterior_face.txt') # exterior_face   all_faces
#parser.add_argument("--textures", type=str, default='/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/logs/epoch-10+withNSR-True+weights-yolov3_9_5.pt+dataset-phy_attack_multi_weather+smooth-tensor3+patchInitialWay-random+batch_size-1+lr-0.01+model-resnet50+loss_func-loss_midu+loss_content+loss_smooth+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+/texture_10.npy')
parser.add_argument("--textures", type=str, default='textures/texture.npy')
parser.add_argument('--devicenumber', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument("--dirname", type=str, default="test")
parser.add_argument("--texturesize", type=int, default=6)
#parser.add_argument("--datapath", type=str, default="carla_dataset/")
args = parser.parse_args()

devicenumber=select_device(args.devicenumber,batch_size=1)

BATCH_SIZE = args.batchsize

obj_file =args.obj
texture_size = args.texturesize

vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)

# Camouflage Textures
texture_content_adv = torch.from_numpy(np.load(args.textures)).cuda(device=devicenumber)
texture_origin =textures[None, :, :, :, :, :].cuda(device=devicenumber)
texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
with open(args.faces, 'r') as f:
    face_ids = f.readlines()
    for face_id in face_ids:
        if face_id != '\n':
            texture_mask[int(face_id)-1, :, :, :, :] = 1
texture_mask = torch.from_numpy(texture_mask).cuda(device=devicenumber).unsqueeze(0)



def cal_texture(texture_content, CONTENT=False):
    textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    print(f"texture_origin.shape{texture_origin.shape},texture_mask.shape{texture_mask.shape},texures.shape{textures.shape}")
    #print(textures)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


@torch.no_grad()
def run_cam(batch_size=BATCH_SIZE):

    textures_adv = cal_texture(texture_content_adv, CONTENT=True)
    print(textures_adv.shape)
    save_path = 'texture_image/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path=os.path.join(save_path,args.dirname)
    texture_save_texture_path=os.path.join(save_path,"texture")
    if not os.path.exists(texture_save_texture_path):
        os.makedirs(texture_save_texture_path)

    texure_save_path=os.path.join(texture_save_texture_path,"model_save.obj")
    neural_renderer.save_obj(texure_save_path, vertices, faces, textures_adv.squeeze(0),texture_size_out=16)
    #
    mask_renderer = nmr.NeuralRenderer(img_size=640).to(devicenumber)
    mask_renderer.renderer.renderer.camera_mode = "look_at"
    mask_renderer.renderer.renderer.light_direction = [0, 0, 1]
    mask_renderer.renderer.renderer.camera_up = [0, 0, 1]
    mask_renderer.renderer.renderer.background_color = [1, 1, 1]

    mask_renderer.renderer.renderer.eye = [2, 2, 2]
    img=mask_renderer.forward(vertices[None, :, :], faces[None, :, :], textures_adv)
    img = img.cpu().numpy().squeeze().transpose((1, 2, 0))
    print(f"img.shape{img.shape}")
    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(os.path.join(texture_save_texture_path, "textur_render.png"))


if __name__ == "__main__":

    batch_size = 1
    input_size = 800

    run_cam()

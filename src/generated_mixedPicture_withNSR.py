import torch
from torch.utils.data import DataLoader
from data_loader import MyDatasetTestAdv
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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ['TORCH_USE_CUDA_DSA'] = '1'
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--obj", type=str, default='car_assets/audi_et_te.obj')
parser.add_argument("--faces", type=str, default='car_assets/exterior_face.txt') # exterior_face   all_faces
#parser.add_argument("--textures", type=str, default='/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/logs/epoch-10+withNSR-True+weights-yolov3_9_5.pt+dataset-phy_attack_multi_weather+smooth-tensor3+patchInitialWay-random+batch_size-1+lr-0.01+model-resnet50+loss_func-loss_midu+loss_content+loss_smooth+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+/texture_10.npy')
parser.add_argument("--textures", type=str, default='/data/zhoujw/2024.1.2_last_logs/epoch-10+withNewNSR-True+fog-new+loss-active_new_car+texturesize-6+weights-yolov3_9_5.pt+dataset-phy_multi_weather_new_day_right+smooth-tensor3+patchInitialWay-random_right+batch_size-1+lr-0.01+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+/texture_4.npy')
parser.add_argument('--devicenumber', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument("--datapath", type=str, default="/data/zhoujw/phy_multi_weather_new_day_right/")
parser.add_argument("--dirname", type=str, default="test2_25")
parser.add_argument("--texturesize", type=int, default=6)
#parser.add_argument("--datapath", type=str, default="carla_dataset/")
args = parser.parse_args()

devicenumber=select_device(args.devicenumber,batch_size=1)

BATCH_SIZE = args.batchsize
mask_dir = os.path.join(args.datapath, 'masks/')

obj_file =args.obj
texture_size = args.texturesize

vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)


# Camouflage Textures
texture_content_adv = torch.from_numpy(np.load(args.textures)).cuda(device=devicenumber)


#texture_content_adv = torch.from_numpy(np.ones((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')).cuda(device=devicenumber)*100
#texture_content_adv = (torch.from_numpy(np.random.random((faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32')).cuda(device=devicenumber)*2-1)*100
#print(texture_content_adv)

# texture_min=np.min(np.random.random((faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32'))
# texture_max=np.max(np.random.random((faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32'))
# print(f"texture_max:{texture_max}")
    
# print(f"texture_min:{texture_min}")
#texture_content_adv=textures.clone().cuda(device=devicenumber)
texture_origin =textures[None, :, :, :, :, :].cuda(device=devicenumber)
texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
with open(args.faces, 'r') as f:
    face_ids = f.readlines()
    for face_id in face_ids:
        if face_id != '\n':
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
texture_mask = torch.from_numpy(texture_mask).cuda(device=devicenumber).unsqueeze(0)



def cal_texture(texture_content, CONTENT=False):
    textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    print(f"texture_origin.shape{texture_origin.shape},texture_mask.shape{texture_mask.shape},texures.shape{textures.shape}")
    #print(textures)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


@torch.no_grad()
def run_cam(data_dir, batch_size=BATCH_SIZE):
    print(data_dir)

    dataset = MyDatasetTestAdv(data_dir, input_size, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True,device_number=devicenumber)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=2,
    )

    print(len(dataset))
    tqdm_loader = tqdm(loader)
    
    textures_adv = cal_texture(texture_content_adv, CONTENT=True)
    
    # image_adv_array, _ =create_texture_image(textures_adv.squeeze(0),6)
    # image_adv_array= (image_adv_array * 255).astype(np.uint8)
    # image_adv = Image.fromarray(image_adv_array)
    # image_adv.save("image_adv.png")
    print(textures_adv.shape)
    # vertices_squeeze=vertices.unsqueeze(0)
    # faces_squeeze=faces.unsqueeze(0)
    save_path = '/data/zhoujw/last_save_path/multi-weather'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path=os.path.join(save_path,args.dirname)
    texture_save_texture_path=os.path.join(save_path,"texture")
    if not os.path.exists(texture_save_texture_path):
        os.makedirs(texture_save_texture_path)

    texure_save_path=os.path.join(texture_save_texture_path,"model_save.obj")
    neural_renderer.save_obj(texure_save_path, vertices, faces, textures_adv.squeeze(0),texture_size_out=16)

    dataset.set_textures(textures_adv)
    texure_save_image_path=os.path.join(save_path,"images")
    model_nsr=U_Net()
    
    saved_state_dict = torch.load('/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/logs/epoch-9+dataset-DTN+ratio-false+night-4+day-0+sync-true+patchInitialWay-random+batch_size-16+lr-0.01+model-resnet50+loss_func-loss_midu+loss_content+loss_smooth+lamb-0.0001+D1-0.9+D2-0.1+T-0.0001+/model_nsr_s9_l19.pth')  # 原始的参数字典

# 假设参数是使用DistributedModel保存的
# 如果原始模型是使用DataParallel进行分布式训练，可以使用以下代码来修复参数字典的键名
    
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        name = k[7:]  # 去掉 'module.' 前缀
        new_state_dict[name] = v
    saved_state_dict = new_state_dict
    model_nsr.load_state_dict(saved_state_dict)
    model_nsr.to(devicenumber)
    model_nsr.eval()
    if not os.path.exists(texure_save_image_path):
        os.makedirs(texure_save_image_path)
    for i, (index, total_img, texture_img, mask,img_cut, filename) in enumerate(tqdm_loader):
            filename = filename[0].split('.')[0]
            #if not os.path.exists(fr'{texure_save_image_path}/{filename}.png'):
            #print("newone")
            img_cut = img_cut.to(devicenumber, non_blocking=True).float() / 255.0
            out_tensor = model_nsr(img_cut)
            sig = nn.Sigmoid()
            relu= nn.ReLU()
            out_tensor=sig(out_tensor)  # forward
            tensor1 = out_tensor[:,0:3, :, :]
            tensor2 = out_tensor[:,3:6, :, :]
            # tensor3_ = out_tensor[:,6:9, :, :]
            # tensor4 = out_tensor[:,9:12, :, :]
            # print(tensor1.shape)
            # print(tensor2.shape)
            tensor3=torch.clamp(texture_img*tensor1+tensor2,max=1)
            #tensor3=torch.clamp((relu(texture_img-tensor1)+tensor2)*tensor3_+tensor4,max=1)

            imgs=(1 - mask) * total_img +(255 * tensor3) * mask
            
            
            
            texture_img_np = imgs.data.cpu().numpy()[0]
            texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
            texture_img_np.save(fr'{texure_save_image_path}/{filename}.png')
            texture_img_np = 255*tensor1.data.cpu().numpy()[0]
            texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
            texture_img_np.save(fr'{texure_save_image_path}/tensor1.png')
            texture_img_np = 255*tensor2.data.cpu().numpy()[0]
            texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
            texture_img_np.save(fr'{texure_save_image_path}/tensor2.png')
            texture_img_np = 255*texture_img.data.cpu().numpy()[0]
            texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
            texture_img_np.save(fr'{texure_save_image_path}/tensor.png')
            # # Yolo-v5 detection
            # results = net(texture_img_np)
            # # print(results)
            # print(results)
            # results.save(fr'{save_path}/{filename}_pred.png')
            
            #results.show()



if __name__ == "__main__":
    data_dir = f"{args.datapath}/test/"
    batch_size = 1
    input_size = 800
    # texure_save_path=os.path.join('./',"texture_origin.obj")
    # print(texure_save_path)
    # neural_renderer.save_obj(texure_save_path, vertices, faces, textures)

    run_cam(data_dir)
# （News！！）Improved version: RAUCA-E2E
Based on RAUCA, we propose a improved version of robust and accurate camouflage generation method RAUCA-E2E, the new work can be found here [https://github.com/SeRAlab/RAUCA-E2E](https://github.com/SeRAlab/RAUCA-E2E).

# [ICML 2024]RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation

This is the official implementation of the Robust-and-Accurate-UV-map-based-Camouflage-Attack(RAUCA) method proposed in our ICML 2024 paper [RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation](https://arxiv.org/abs/2402.15853)

[Code](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack/tree/main/src); [Poster](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack/tree/main/assets/RAUCA_Poster.pdf)



## Abstract
Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle, resulting in suboptimal attack performance. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, Neural Renderer Plus (NRP), which can accurately project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA consistently outperforms existing methods in both simulation and real-world settings.

## Framework
![pipeline](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack/blob/main/assets/pipeline.png)
The overview of RAUCA. First, a multi-weather dataset is created using CARLA, which includes car images, corresponding mask images, and camera angles. Then the car images are segmented using the mask images to obtain the foreground car and background images. The foreground car, together with the 3D model and the camera angle is passed through the NRP rendering component for rendering. The rendered image is then seamlessly integrated with the background. Finally, we optimize the adversarial camouflage through back-propagation with our devised loss function computed from the output of the object detector.

## Requirements:

#### Setup

```
conda create -n RAUCA python=3.9
conda activate RAUCA
cd src
git clone https://github.com/zhoujiawei3/Neural_Renderer_RAUCA.git
conda create -n RAUCA python=3.9
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
cd neural_renderer_RAUCA
#install neural_renderer, if meet bugs can search guide in [here](https://winterwindwang.github.io/2021/07/22/nerual_rendered_build.html)
sudo apt install ninja-build
python setup.py install
ninja -f  build/temp.linux-x86_64-cpython-39/build.ninja
python setup.py install
ninja -f  build/temp.linux-x86_64-cpython-39/build.ninja
python setup.py install
ninja -f  build/temp.linux-x86_64-cpython-39/build.ninja
python setup.py install
cd ..
conda env update --file environment.yml
```

Dowdload the YOLO-V3 weight from [here](https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt). Put it into `src` folder and rename it to `yolov3_9_5.pt`

After train the adversarial camouflage, you can see how camouflage like with the code in `src` folder.
## Dataset:
The multi-weather dataset for adversarial camouflage generation can get [here](https://pan.baidu.com/s/17LdfDcGt3aZygN84JCP46Q?pwd=ir65).
Update path in `src/data/carla.yaml`.

## NRP-weight:
We offer the NRP-weight that can be used directly. It can get [here](https://pan.baidu.com/s/1iKtlv44Uq_1YcQyLH0SSlQ?pwd=e17m). Put it into `src/NRP_weights/` folder.


## Run:
To train NRP:
```bash
python NRP_training.py
```
To get camouflage:
```bash
python train_camouflage.py --datapath {datapath}
```

## Post-processing to get the deployable UV map:
The output of `generate_camouflage_E2E.py` is in the form of `.npy` (Our generated texture is [here](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack/tree/main/src/textures/texture.npy)). To get the image of the texture, you can use the following script,
```bash
python generated_mixedPicture_withNSR.py --textures=texture/texture.npy
```
The image of the UV map generated with this script is show in [src/texture_image/test/texture/model_save.png](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack/tree/main/texture_image/test/texture/model_save.png).

Although the script can convert the npy file into an image texture format, it is difficult to print directly due to its fragmented root structure. Therefore, we further re-bake it in Blender to make it printable and deployable in the real world. The re-bake texture image can be found [Here](https://github.com/SeRAlab/Robust-and-Accurate-UV-map-based-Camouflage-Attack/blob/main/src/texture_image/test/texture/deployable_UV_map.png). The conversion tutorial can be found [Here](https://www.bilibili.com/video/BV1abD3YnEQW/?spm_id_from=333.999.0.0). 

## Attack Performance Demonstration

https://github.com/user-attachments/assets/3abeaa66-e488-4e13-a331-53d7fbb30666

https://github.com/user-attachments/assets/72490129-7dcc-4ee7-858e-caea9216e629










## Citation
```bibtex
@inproceedings{
zhou2024rauca,
title={{RAUCA}: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation},
author={Jiawei Zhou and Linye Lyu and Daojing He and YU LI},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=pBTLGM9uWx}
}


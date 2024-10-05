<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
    <h1>法向-深度扩散模型</h1>
<p>

法向-深度扩散模型: 从文本生成法向图和深度图的扩散模型。

## 文本到ND

![teaser-nd](assets/text-to-nd-laion.png)

## 文本到ND-MV

![teaser-nd-mv](assets/nd-mv.jpg)

## [项目主页](https://aigc3d.gitee.io/richdreamer/)| [论文](https://arxiv.org/abs/2311.16918) | [bilibili](https://www.bilibili.com/video/BV1Qb4y1K7Sb/?spm_id_from=888.80997.embed_other.whitelist)
- [x] 推理代码
- [x] 训练代码  
- [x] 预训练模型: ND, ND-MV, Albedo-MV
- [ ] 预训练模型: ND-MV-VAE
- [x] Objaverse数据集渲染的多视图图像

## 新闻
- 2023-12-25: mvs_objavse通过Alibaba OSS向外发布，我们提供了一个方便的多线程下载链接。
- 2023-12-11: 推理代码和预训练模型发布。

## 3D 生成
- 该代码仓库仅包含RichDreamer论文的扩散模型和2D图像生成代码
- 对于 3D 生成，请查看 [RichDreamer](https://github.com/modelscope/RichDreamer).


## 推理准备
1. 使用以下脚本安装环境依赖
```bash 
conda create -n nd
conda activate nd
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/CompVis/taming-transformers.git 
pip install webdataset
pip install img2dataset
```
我们还提供了dockerfile来构建docker镜像。
```bash
sudo docker build -t mv3dengine_22.04:cu118 -f docker/Dockerfile . 
```

2. 下载预训练权重。 
- [ND](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd-laion_ema.ckpt): 在Laion-2B上训练的法向-深度扩散模型
- [ND-MV](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd_mv_ema.ckpt): 多视角法向-深度扩散模型
- [Alebdo-MV](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/albedo_mv_ema.ckpt): 多视角深度图条件控制的反照率扩散模型

**或者**使用我们提供的下载脚本：
```bash
python tools/download_models/download_nd_models.py
```

## 推理(采样)
我们提供了采样脚本
```bash 
python demo_inference.sh
```
**或**使用以下详细说明:

### 文本到ND采样
```
# dmp求解器
python ./scripts/t2i.py --ckpt $ckpt_path --prompt $prompt --dpm_solver --n_samples 2 --save_dir $save_dir
# plms求解器  
python ./scripts/t2i.py --ckpt $ckpt_path --prompt $prompt --plms --n_samples 2  --save_dir $save_dir
# ddim求解器
python ./scripts/t2i.py --ckpt $ckpt_path --prompt $prompt --n_samples 2  --save_dir $save_dir
```

### 文本到ND-MV采样
```
# nd-mv  
python ./scripts/t2i_mv.py --ckpt_path $ckpt_path --prompt $prompt  --num_frames 4  --model_name nd-mv --save_dir $save_dir   

# nd-mv-vae(即将推出)
python ./scripts/t2i_mv.py --ckpt_path $ckpt_path --prompt $prompt  --num_frames 4  --model_name nd-mv-vae --save_dir $save_dir  

```

### 文本到Albedo-MV采样
```
python ./scripts/td2i_mv.py --ckpt_path $ckpt_path --prompt $prompt --depth_file $ depth_file --num_frames 4  --model_name albedo-mv --save_dir $save_dir  

```


## 训练准备  

1. 下载Laion-2B-en-5-AES(**训练ND模型所需**)  

从[parquet](https://huggingface.co/datasets/laion/laion2B-en)下载laion-2b数据集，然后将下载的文件放入```./laion2b-dataset-5-aes```
```bash  
cd ./tools/download_dataset  
bash ./download_2b-5_aes.sh  
cd -
```  

2. 下载单目法向和深度图估计的网络权重(**训练ND模型所需**) 
- NormalBae [scannet.pt](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/scannet.pt)  
- Midas3.1 [dpt_beit_large512.pt](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/dpt_beit_large_512.pt)  

```bash   
# 将scannet.pt移至normalbae Prior Model   
mv scannet.pt ./libs/ControlNet-v1-1-nightly/annotator/normalbae/scannet.pt   
# 将dpt_beit_large512.pt移至./libs/omnidata_torch/pretrained_models/dpt_beit_large_512.pt  
mv dpt_beit_large512.pt ./libs/omnidata_torch/pretrained_models/dpt_beit_large_512.pt  
```  

3. 下载Objaverse数据集渲染的多视图图像(*训练ND-MV和Albedo-MV模型所需*)  
- 使用我们提供的多线程脚本下载我们渲染的数据集

```bash
# 下载文件位置信息的储存表
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/valid_paths_v4_cap_filter_thres_28.json
# 例子: python ./scripts/data/download_objaverse.py ./mvs_objaverse ./valid_paths_v4_cap_filter_thres_28.json 50
python ./scripts/data/download_objaverse.py /path/to/savedata /path/to/valid_paths_v4_cap_filter_thres_28.json nthreads(eg. 10)
# 设置软连接到数据集
ln -s /path/to/savedata mvs_objaverse
# 下载文本标注文件
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/text_captions_cap3d.json
```

## 训练
### 训练Normal-Depth-VAE模型  
1. 下载在ImageNet上预训练的[预训练-VAE权重](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd-vae-imgnet.ckpt): ${pretained-VAE weights}
2. 修改 `configs/autoencoder_normal_depth/autoencoder_normal_depth.yaml` 中的配置文件, 设置 `model.ckpt_path=/path/to/${pretained-VAE weights}`   

```bash    
# 训练 VAE  
bash ./scripts/train_vae/train_nd_vae/train_rgbd_vae_webdatasets.sh \ model.ckpt_path=${pretained-VAE weights} \  
data.params.train.params.curls='path_laion/{00000..${:5 end_id}}.tar' \  
--gpus 0,1,2,3,4,5,6,7  
```  

### 训练 Normal-Depth-Diffusion 模型  
经过 Normal-Depth-VAE 的训练，获得 `Normal-Depth-VAE` 模型或您可以从 [ND-VAE](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd-vae-laion.ckpt) 下载它  

```bash    
# 步骤1    
export SD-MODEL-PATH=/path/to/sd-1.5  
bash scripts/train_normald_sd/txt_cond/web_datasets/train_normald_webdatasets.sh --gpus 0,1,2,3,4,5,6,7 \  
    model.params.first_stage_ckpts=${Normal-Depth-VAE} model.params.ckpt_path=${SD-MODEL-PATH} \  
    data.params.train.params.curls='path_laion/{00000..${:5 end_id}}.tar'  

# 步骤2 修改您的权重路径。configs/stable-diffusion/normald/sd_1_5/txt_cond/web_datasets/laion_2b_step2.yaml  
bash scripts/train_normald_sd/txt_cond/web_datasets/train_normald_webdatasets_step2.sh --gpus 0,1,2,3,4,5,6,7 \  
    model.params.first_stage_ckpts=${Normal-Depth-VAE} \  
    model.params.ckpt_path=${pretrained-step-weights} \  
    data.params.train.params.curls='path_laion/{00000..${:5 end_id}}.tar'  
```  

### 训练 MultiView-Normal-Depth-Diffusion 模型  
经过 Normal-Depth-Diffusion 的训练，获取 `Normal-Depth-Diffusion` 模型或您可以从 [ND](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd-laion.ckpt) 下载它,  

我们提供了两种版本的 MultiView-Normal-Depth Diffusion 模型  

a. 无VAE去噪
b. 带VAE去噪

在当前版本中,我们提供无VAE去噪  

```bash   
# a. 无 VAE 版本训练   
bash ./scripts/train_normald_sd/txt_cond/objaverse/objaverse_finetune_wovae_mvsd-4.sh --gpus 0,1,2,3,4,5,6,7,  \  
    model.params.ckpt_path=${Normal-Depth-Diffusion}  
# b. VAE 版本训练
bash ./scripts/train_normald_sd/txt_cond/objaverse/objaverse_finetune_mvsd-4.sh --gpus 0,1,2,3,4,5,6,7, \  
    model.params.ckpt_path=${Normal-Depth-Diffusion}   

```  

### MultiView-Depth-Conditioned-Albedo-Diffusion 模型训练  
经过 Normal-Depth-Diffusion 的训练，获得 `Normal-Depth-Diffusion` 模型或您可以从 [ND](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd-laion.ckpt) 下载,

```bash   
bash scripts/train_abledo/objaverse/objaverse_finetune_mvsd-4.sh --gpus 0,1,2,3,4,5,6,7, model.params.ckpt_path=${Normal-Depth-Diffusion}   
```  

## 致谢  
我们大量参考了以下仓库的代码。感谢作者分享他们的代码。  
- [stable diffusion](https://github.com/CompVis/stable-diffusion)  
- [mvdream](https://github.com/bytedance/MVDream)  

## 引用     

```  
@article{qiu2023richdreamer,  
    title={RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D},   
    author={Lingteng Qiu and Guanying Chen and Xiaodong Gu and Qi zuo and Mutian Xu and Yushuang Wu and Weihao Yuan and Zilong Dong and Liefeng Bo and Xiaoguang Han},  
    year={2023},  
    journal = {arXiv preprint arXiv:2311.16918}  
}  
```

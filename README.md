# Tetrahedron Splatting for 3D Generation
### [[Project page]](https://fudan-zvg.github.io/tet-splatting) | [[Paper]](https://arxiv.org/abs/2406.01579)

> [**Tetrahedron Splatting for 3D Generation**](https://arxiv.org/abs/2406.01579),            
> [Chun Gu](https://sulvxiangxin.github.io/), Zeyu Yang, Zijie Pan, [Xiatian Zhu](https://surrey-uplab.github.io/), [Li Zhang](https://lzrobots.github.io)  
> **Arxiv preprint**

**Official implementation of "Tetrahedron Splatting for 3D Generation".** 


## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## ⚙️ Installation

```bash
git clone https://github.com/fudan-zvg/tet-splatting.git --recursive
conda create -n tetsplatting python=3.9
conda activate tetsplatting

# install pytorch (e.g. cuda 11.7)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# install other denpendencies
pip install -r requirements.txt
```

You can also refer to [threestudio](https://github.com/threestudio-project/threestudio#installation) or [RichDreamer](https://github.com/modelscope/RichDreamer#install) for preparing the environment.

Download pretrained weights:

```bash
python tools/download_nd_models.py
# copy 256_tets file for dmtet.
cp ./pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/256_tets.npz ./load/tets/
# link your huggingface models to ./pretrained_models/huggingface
cd pretrained_models && ln -s ~/.cache/huggingface ./
```

## 🔄 Generation

```bash
# Run a single prompt
python3 ./run_tetsplatting.py -t $prompt -o $output --gpus $GPU

# Run from prompt list
# e.g. bash ./scripts/tetsplatting/run_batch.sh 0 1 ./prompts_dmtet.txt 0
bash ./scripts/tetsplatting/run_batch.sh $start_id $end_id ${prompts_dmtet.txt} ${GPU}
```

## Acknowledgement

This work is built on many amazing research works:

- [threestudio](https://github.com/threestudio-project/threestudio)
- [RichDreamer](https://github.com/modelscope/RichDreamer)
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec)
- [StopThePop](https://github.com/r4dl/StopThePop-Rasterization)

## 📜 BibTeX
```bibtex
@inproceedings{gu2024tetrahedron,
  title={Tetrahedron Splatting for 3D Generation},
  author={Gu, Chun and Yang, Zeyu and Pan, Zijie and Zhu, Xiatian and Zhang, Li},
  booktitle={NeurIPS},
  year={2024}
}
```
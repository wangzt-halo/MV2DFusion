<div align="center">
<h1>MV2DFusion</h1>
</div>

---

This repo is the official PyTorch implementation for paper: 
[MV2DFusion](https://arxiv.org/abs/2408.05945)

The rise of autonomous vehicles has significantly increased the demand for robust 3D object detection systems. While cameras and LiDAR sensors each offer unique advantages--cameras provide rich texture information and LiDAR offers precise 3D spatial data--relying on a single modality often leads to performance limitations. This paper introduces MV2DFusion, a multi-modal detection framework that integrates the strengths of both worlds through an advanced query-based fusion mechanism. By introducing an image query generator to align with image-specific attributes and a point cloud query generator, MV2DFusion effectively combines modality-specific object semantics without biasing toward one single modality. Then the sparse fusion process can be accomplished based on the valuable object semantics, ensuring efficient and accurate object detection across various scenarios. Our framework's flexibility allows it to integrate with any image and point cloud-based detectors, showcasing its adaptability and potential for future advancements. Extensive evaluations on the nuScenes and Argoverse2 datasets demonstrate that MV2DFusion achieves state-of-the-art performance, particularly excelling in long-range detection scenarios.

## Preparation

---
### Environment
- Linux
- Python == 3.8.10
- CUDA == 11.3
- PyTorch == 1.11.0

### Installation
Follow the instructions below to install required packages.
```shell
git clone 
cd MV2DFusion

pip install mmcls                     # 0.23.2
pip install mmcv-full                 # 1.6.1
pip install mmdet                     # 2.25.1
pip install mmdet3d                   # 1.0.0rc4
pip install mmsegmentation            # 0.28.0
pip install nuscenes-devkit
pip install av2
pip install refile                    # 0.4.1
pip install spconv-cu113              # 2.3.6
pip install flash-attn                # 1.0.2
pip install torch-scatter             # 2.1.2
git clone https://github.com/Abyssaledge/TorchEx.git
cd TorchEx/
pip install -e . && cd ..
```

### Data Processing   
Follow [mmdet3d](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md) and [streampetr](https://github.com/exiawsh/StreamPETR) to process datasets.
You can also process the datasets using the scripts under `tools/`.

### Pretrained Weights   
We use nuImages pretrained weights ([link](https://github.com/open-mmlab/mmdetection3d)) for image-based detector, 
FSDv2 pretrained weights ([link](https://github.com/tusen-ai/SST)) for point-basede detector. 
You can download the pretrained weights and put them into `weights/` directory.

We also provide the pretrained weights that can be downloaded from [[google drive](https://drive.google.com/drive/folders/1gnBZdzRJbvR5-wWpNXQWo5Qhl1No6dqS?usp=drive_link)].

### Directory Structure
After preparation, you will be able to see the following directory structure:  
  ```
  MV2D
  ├── configs/
  ├── projects/
  ├── tools/
  ├── data/
  │   ├── nuscenes/
  │     ├── samples/
  │     ├── sweeps/
  │     ├── maps/
  │     ├── v1.0-trainval/
  │     ├── v1.0-test/
  │     ├── nuscenes2d_temporal_infos_train.pkl
  │     ├── nuscenes2d_temporal_infos_val.pkl
  │     ├── nuscenes2d_temporal_infos_test.pkl
  │     ├── ...
  │   ├── argo/
  │     ├── converted/
  |         ├── av2_train_infos_mini.pkl
  |         ├── av2_val_infos_mini.pkl
  |         ├── val_anno.feather
  |         ├── train -> ../sensor/train
  |         ├── val -> ../sensor/val
  |         ├── ...
  │     ├── sensor/
  |         ├── train/
  |         ├── val/
  |         ├── test/
  |         ├── ...
  ├── weights/
  ├── README.md
  ```

## Train & Inference

---
You can train the model following:
```bash
bash tools/dist_train.sh projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py 8 
```
You can evaluate the model following:
```bash
bash tools/dist_test.sh projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py work_dirs/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/latest.pth 8 --eval bbox
```

**Note:** *Sometimes the training process may crash for unknown reasons. A simple solution is to resume training from a recent stable checkpoint.*

## Main Results

---
### nuScenes validation set
|                                                  config                                                  |  NDS  |  mAP  |  checkpoint  |
|:--------------------------------------------------------------------------------------------------------:|:-----:|:-----:|:------------:|
|      [MV2DFusion-R50](./projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py)      | 0.748 | 0.730 | [download](https://drive.google.com/drive/folders/1UEB-ynVrprZrUzY6UWzEZFPjaF-FEzpx?usp=drive_link) |  
|[MV2DFusion-ConvNext](./projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep24_nusc.py) | 0.753 | 0.743 | [download](https://drive.google.com/drive/folders/1sQFDBd_ujRJ5voRDIIuD009GQXfTylsj?usp=drive_link) |  

### nuScenes test set
|                                                       config                                                        |  NDS  |  mAP  |  checkpoint  |
|:-------------------------------------------------------------------------------------------------------------------:|:-----:|:-----:|:------------:|
| [MV2DFusion-ConvNext](./projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48_trainval_nusc.py)  | 0.767 | 0.745 | [download](https://drive.google.com/drive/folders/1e_fbfJSoDxASx6l0rBW5ZkdxbkmVeVRi?usp=sharing) |  

### Argoverse 2 validation set
|                                         config                                         |  CDS  |  mAP  |  checkpoint  |
|:--------------------------------------------------------------------------------------:|:-----:|:-----:|:------------:|
| [MV2DFusion-R50](./projects/configs/argo/mv2dfusion-fsd_freeze-r50_1536-ep6_argov2.py) | 0.395 | 0.486 | [download](https://drive.google.com/drive/folders/1xq5KBAG7TBKAZXqrndlG7Nl1vqEo-XCC?usp=drive_link) |  


## Acknowledgement

---
Many thanks to the awesome open-sourced projects:
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [SST](https://github.com/tusen-ai/SST/tree/main)
- [Far3D](https://github.com/megvii-research/Far3D)

## Citation

---
If you find this repo useful for your research, please consider citing:
```
@article{wang2024mv2dfusion,
  title={MV2DFusion: Leveraging Modality-Specific Object Semantics for Multi-Modal 3D Detection},
  author={Wang, Zitian and Huang, Zehao and Gao, Yulu and Wang, Naiyan and Liu, Si},
  journal={arXiv preprint arXiv:2408.05945},
  year={2024}
}

@article{wang2023object,
  title={Object as query: Equipping any 2d object detector with 3d detection ability},
  author={Wang, Zitian and Huang, Zehao and Fu, Jiahui and Wang, Naiyan and Liu, Si},
  journal={arXiv preprint arXiv:2301.02364},
  year={2023}
}
```

## Contact

---
For questions about our paper or code, please contact **Zitian Wang** (wangzt.kghl@gmail.com).
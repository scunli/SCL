# LSNet: Laplacian Pyramid Monocular Depth Estimation

Official PyTorch implementation of **Laplacian Pyramid Monocular Depth Estimation with Global‑Local Collaborative Decoding** (LSNet).  
The model introduces a **Large‑kernel Perception and Small‑kernel Aggregation (LKP‑SKA)** module that mimics the human visual system – global context captured by large kernels guides adaptive local detail refinement.

# Important Notice
The code and data provided in this repository are directly associated with our manuscript currently under review at **The Visual Computer** journal. During the review process, the repository will remain in this state. If you use these resources in your work, please cite our related paper.
---

## Key Features

- **LKP‑SKA decoder** – dynamic small‑kernel aggregation steered by large‑kernel global priors  
- **Serial ASPP** – cascaded atrous convolutions for coherent multi‑scale context  
- **Composite loss** – scale‑invariant loss + gradient loss (staged activation)  
- **Memory efficient** – gradient accumulation, mixed precision, custom SKA implementation  
- **Edge‑aware evaluation** – boundary F1 and edge RMSE metrics

---

##  Dependencies & Environment

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+ (recommended)
- Other packages: `numpy`, `opencv-python`, `Pillow`, `matplotlib`, `tqdm`, `tensorboardX`, `scikit-image`, `path.py`, `imageio`

Install with:

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib tqdm tensorboardX scikit-image path.py imageio
```

##  Dataset Preparation

- **KITTI**
  place data as:

```bash
  datasets/KITTI/rgb/depth/eigen_train_files_with_gt_dense.txt
  datasets/KITTI/rgb/depth/eigen_train_files_with_gt_dense.txt
```

##  Dataset Preparation

- **NYU**
  place data as:

```bash
  datasets/NYU/rgb/depth/nyudepthv2_train_files_with_gt_dense.txt
  datasets/NYU/rgb/depth/nyudepthv2_test_files_with_gt_dense.txt
```

##  Training

```bash
# Single GPU
python train.py --dataset KITTI --batch_size 16 --epochs 25 --lr 0.0001 --gpu_num 0

# Multi‑GPU distributed
python train.py --dataset KITTI --distributed --gpu_num 0,1,2,3

# Training on NYU
python train.py --dataset NYU --batch_size 16 --epochs 30 --max_depth 10.0
```

##  Evaluation

```bash
# Evaluate a single model
python eval.py --evaluate --model_dir checkpoints/your_model.pkl --dataset KITTI

# Evaluate all models in a directory
python eval.py --evaluate --multi_test --models_list_dir checkpoints/ --dataset KITTI

# Save depth predictions as images
python eval.py --evaluate --model_dir checkpoints/your_model.pkl --img_save --result_dir ./results
```

##  Inference on a Single Image

```bash
# Single image
python demo.py --model_dir checkpoints/your_model.pkl --img_dir image.jpg --cuda

# Whole folder
python demo.py --model_dir checkpoints/your_model.pkl --img_folder_dir ./images --cuda
```

## Key Algorithm Description

### 1. LKP‑SKA Module

**Large‑kernel Perception (LKP)**  
`1×1` pointwise conv → `7×7` depthwise conv → generates dynamic weight tensor `W` that encodes global scene layout.

**Small‑kernel Aggregation (SKA)**  
Grouped dynamic convolution using `W` as kernel weights – adaptively fuses local features (kernel size `3`).  
The implementation (`ska.py`) uses a memory‑efficient custom `torch.autograd.Function`.

### 2. Laplacian Pyramid Decoder

- 5 levels (coarse to fine).  
- Levels 5,4,3: LKP‑SKA module.  
- Levels 2,1: standard convolutions.  
- RGB Laplacian residuals explicitly guide depth residual prediction (with weight coefficient `α`).

### 3. Serial ASPP

Cascaded dilated convolutions (dilations `6 → 12 → 18`) instead of parallel branches, providing coherent multi‑scale context (`Dilated_bottleNeck` in `model.py`).

##  Citation

```bash
@article{song2026lsnet,
  title={Laplacian Pyramid Monocular Depth Estimation with Global-Local Collaborative Decoding},
  author={Song, Cunli and Song, Qiukun and Zhang, Xuesong},
  year={2026}
}
```


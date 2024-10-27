# SFNet


Project of manuscripts *SFNet: Fusion of Spatial and Frequency-Domain Features for Remote Sensing Image Forgery Detection*.

The entire code is based on the excellent mmpretrain framework implementation.

What this project has to offer.
- [x] SFNet model implementation
- [x] Training and testing scripts.

## Installation

### Prerequisites
In this section we demonstrate how to prepare an environment with PyTorch.

MMPretrain works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 1.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 2.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 3.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

### Install from source

Download the source code from our [repository](https://github.com/GeoX-Lab/RSTI/tree/main/SFNet).

```shell
pip install -U openmim && mim install -e .
```

## Usage

### Train

```shell
python tools/train.py
    --config configs/0_RSI_Authentication/sfnet_full.py
    --work-dir work_dirs/0_RSI_Authentication/sfnet_full
```


### Test

```shell
python tools/test.py
    --config configs/0_RSI_Authentication/sfnet_full.py
    --checkpoint XXXX.pth
    --work-dir XXXX

```
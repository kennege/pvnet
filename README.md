**Good news! We release a clean version of PVNet: [clean-pvnet](https://github.com/zju3dv/clean-pvnet), including**

1. [how to train the PVNet on the custom dataset](https://github.com/zju3dv/clean-pvnet#training-on-the-custom-dataset).
2. The training and testing on the tless dataset, where we detect multiple instances in an image.

# PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation

![introduction](./assets/introduction.png)

> [PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation](https://arxiv.org/pdf/1812.11788.pdf)  
> Sida Peng, Yuan Liu, Qixing Huang, Xiaowei Zhou, Hujun Bao   
> CVPR 2019 oral  
> [Project Page](https://zju3dv.github.io/pvnet)

Any questions or discussions are welcomed!

## Truncation LINEMOD Dataset

Check [TRUNCATION_LINEMOD.md](TRUNCATION_LINEMOD.md) for information about the Truncation LINEMOD dataset.

## Installation

One way is to set up the environment with docker: [How to install pvnet with docker](docker/how-to-docker.md).

Thanks **Joe Dinius** for providing the docker implementation.

Another way is to use the following commands.

1. Set up python 3.6.7 environment

```
pip install -r requirements.txt
```

We need compile several files, which works fine with pytorch v0.4.1/v1.1 and gcc 5.4.0.

For users with a RTX GPU, you must use CUDA10 and pytorch v1.1 built from CUDA10.

2. Compile the Ransac Voting Layer

```
ROOT=/path/to/pvnet
cd $ROOT/lib/ransac_voting_gpu_layer
python setup.py build_ext --inplace
```

3. Compile some extension utils

```
cd $ROOT/lib/utils/extend_utils
```

Revise the `cuda_include` and `dart` in `build_extend_utils_cffi.py` to be compatible with the CUDA in your computer.

```
sudo apt-get install libgoogle-glog-dev=0.3.4-0.1
sudo apt-get install libsuitesparse-dev=1:4.4.6-1
sudo apt-get install libatlas-base-dev=3.10.2-9
python build_extend_utils_cffi.py
```

If you cannot install `libsuitesparse-dev=1:4.4.6-1`, please install `libsuitesparse`, run `build_ceres.sh` and move `ceres/ceres-solver/build/lib/libceres.so*` to `lib/utils/extend_utils/lib`.

Add the `lib` under `extend_utils` to the LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/pvnet/lib/utils/extend_utils/lib
```

## Dataset Configuration

### Prepare the dataset

Download the LINEMOD, which can be found at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EXK2K0B-QrNPi8MYLDFHdB8BQm9cWTxRGV9dQgauczkVYQ?e=beftUz).

Download the LINEMOD_ORIG, which can be found at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EaoGIPguY3FAgrFKKhi32fcB_nrMcNRm8jVCZQd7G_-Wbg?e=ig4aHk).

Download the OCCLUSION_LINEMOD, which can be found at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/ESXrP0zskd5IvvuvG3TXD-4BMgbDrHZ_bevurBrAcKE5Dg?e=r0EgoA).

### Create the soft link

```
mkdir $ROOT/data
ln -s path/to/LINEMOD $ROOT/data/LINEMOD
ln -s path/to/LINEMOD_ORIG $ROOT/data/LINEMOD_ORIG
ln -s path/to/OCCLUSION_LINEMOD $ROOT/data/OCCLUSION_LINEMOD
```

### Compute FPS keypoints

```
python lib/utils/data_utils.py
```

### Synthesize images for each object

See [pvnet-rendering](https://github.com/zju-3dv/pvnet-rendering) for information about the image synthesis.

## Demo

Download the pretrained model of cat from [here](https://1drv.ms/u/s!AtZjYZ01QjphgQkDZa7fyvvaD7P6) and put it to `$ROOT/data/model/cat_demo/199.pth`. 

Run the demo

```
python tools/demo.py
```

If setup correctly, the output will look like

![cat](./assets/cat.png)

### Visualization of the voting procedure

We add a jupyter notebook [visualization.ipynb](./visualization.ipynb) for the keypoint detection pipeline of PVNet, aiming to make it easier for readers to understand our paper. Thanks for Kudlur, M 's suggestion. 

## Training and testing

### Training on the LINEMOD

Before training, remember to add the `lib` under `extend_utils` to the LD_LIDBRARY_PATH

```
export LD_LIDBRARY_PATH=$LD_LIDBRARY_PATH:/path/to/pvnet/lib/utils/extend_utils/lib
```

Training

```
python tools/train_linemod.py --cfg_file configs/linemod_train.json --linemod_cls cat
```

### Testing

We provide the pretrained models of each object, which can be found at [here](https://1drv.ms/f/s!AtZjYZ01QjphgQBQDQghxjbkik5f).

Download the pretrained model and move it to `$ROOT/data/model/{cls}_linemod_train/199.pth`. For instance

```
mkdir $ROOT/data/model
mv ape_199.pth $ROOT/data/model/ape_linemod_train/199.pth
```

Testing

```
python tools/train_linemod.py --cfg_file configs/linemod_train.json --linemod_cls cat --test_model
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{peng2019pvnet,
  title={PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation},
  author={Peng, Sida and Liu, Yuan and Huang, Qixing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2019}
}
```


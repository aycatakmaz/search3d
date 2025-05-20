<p align="center">

  <h1 align="center">Search3D 🔎: Hierarchical Open-Vocabulary 3D Segmentation</h1>

<p align="center">
    <a href="https://aycatakmaz.github.io/">Ayca Takmaz</a><sup>1,2,†</sup>,</span>
    <a href="https://alexdelitzas.github.io/">Alexandros Delitzas</a><sup>1</sup>,
    <a href="https://studios.disneyresearch.com/people/bob-sumner/">Robert W. Sumner</a><sup>1</sup>,
    <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>1,2,3,*</sup>,
    <a href="https://scholar.google.de/citations?user=dfjN3YAAAAAJ&hl=en">Johanna Wald</a><sup>2,*</sup>,
    <a href="https://federicotombari.github.io/">Federico Tombari</a><sup>2</sup>
    <br>
    <sup>1</sup>ETH Zurich, 
    <sup>2</sup>Google, 
    <sup>3</sup>Stanford <br>
    <sup>†</sup>work done as an intern at Google Zurich
    <sup>*</sup>equal supervision
  </p>


  <h2 align="center">IEEE RA-L 2025</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2409.18431">Paper</a> | <a href="https://search3d-segmentation.github.io">Project Page</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://search3d-segmentation.github.io/static/images/teaser.jpg" alt="Logo" width="80%">
  </a>
</p>
<p align="center">
<strong>Search3D</strong> is a an approach that builds a hierarchical open-vocabulary 3D scene representation, enabling the search for entities at varying levels of granularity: fine-grained object parts, entire objects, or regions described by attributes like materials.
</p>
<br>


---
## Environment setup and installation
Below, we outline the steps for setting up the environment and installing the necessary packages.

### Step 1. Creating the environment
```bash
conda create -n search3d python=3.10
conda activate search3d
pip install -e .  # install current repository in editable mode
```

### Step 2.  Installing the packages required for SigLIP
```bash
pip install numpy==1.26 
pip install --upgrade "jax[cuda12_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r search3d/mask_feature_computation/big_vision_siglip/big_vision/requirements.txt
```

```python
# you can verify that the installed jax and tensorflow can indeed access the GPUs in Python with the following test:
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

### Step 3.  Installing the packages required for general Search3D ops
```bash
pip install numpy==1.26 torch==1.12.1 torchvision==0.13.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install trimesh open3d imageio open-clip-torch
```

### Step 4.  Installing the packages required for Semantic-SAM ops
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install opencv-python transformers hydra-core omegaconf kornia

cd search3d/dense_feature_computation/semantic_sam/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Step 5.  Installing the packages required for Segmentator (geometric oversegmentation with graph cut)
```bash
pip install numba

cd search3d/object_and_part_computation/segmentator/csrc
mkdir build && cd build

export CUDA_BIN_PATH=/usr/local/cuda-11.7
cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

make && make install
```

### Step 6. Installing the packages required for instance-level object mask prediction (Mask3D)
```bash
TBD
```

## Downloading resources and checkpoints
You can download all necessary checkpoints for the underlying submodules (SigLIP, SemanticSAM etc.) from [this Google Drive folder](https://drive.google.com/drive/folders/1UddvGpz2mJELq0dQBU-Y7CVaB6TCXG7r?usp=sharing). Once you download the checkpoints into a folder and unpack it, you can link that directory to the `resources` folder in this repository. You can do this in the following way:

```bash
mkdir resources
ln -s /path/to/folder/with/the/downloaded/checkpoints resources
```

## Downloading the datasets
```bash
TBD
```

## Running Search3D: Computing the masks and features using Search3D
There are a couple of components of Search3D that we run in order to compute object masks, object features, segments and segment features. At the moment in the current form of this codebase, we perform the merging of the segments and hierarchical search directly in our evaluation scripts. We plan to integrate that directly in this codebase too in the near future. Here, we are outlining how to compute the masks and features for the MultiScan dataset.

### Step 1. Compute object masks
```bash
# first set-up the environment (see previous section)
# run the following script that computes and extracts all object masks for all scenes in MultiScan
# please don't forget to set the dataset directory and output directory in this script!
bash run_search3d_multiscan_obj_masks.sh
```

### Step 2. Compute object features
```bash
# run the following script that reads all object masks extracted in the previous step and computes object features for all scenes in MultiScan
# please don't forget to set the dataset directory and output directory in this script!
bash run_search3d_multiscan_obj_features.sh
```

### Step 3. Compute segments and segment features
```bash
# run the following script that reads all object masks extracted in the first step, computes segments constrained to these object instances and exports the hierarchical scene representation.
# then, it computes segment features for all scenes in MultiScan (see the second section in the following script)
# please don't forget to set the dataset directory and output directory in this script!
bash run_search3d_multiscan_obj_features.sh
```



---
## Citation :pray:
```
@article{takmaz2025search3d,
  title={{Search3D: Hierarchical Open-Vocabulary 3D Segmentation}},
  author={Takmaz, Ayca and Delitzas, Alexandros and Sumner, Robert W. and Engelmann, Francis and Wald, Johanna and Tombari, Federico},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025}
}
```
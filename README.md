# Evaluation of MatchAnything Foundation Models on AmalgaMatch Dataset


## Description:
This repository contains codes and material accompanying the article [Foundation Models for Multimodal Image Data Fusion in Materials Science](). The codes run an evaluation of MatchAnything models pre-trained by He et al. ("MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training") on the whole AmalgaMatch Dataset. The AmalgaMatch dataset contains image pairs annotated with point correspondences, where images are micrographs captured using various common materials microscopy techniques. The dataset covers multiple use-cases in correlative materials microscopy and a wide range of materials. Detailed information about the dataset can be found in the accompanying [data descriptor article]().

## Installation
Create the python environment by:
```
conda env create -f environment.yaml
conda activate env
```
The MatchAnything model evaluation on the AmalgaMatch dataset was run with CUDA 12.7.


## Model weights
Download the MatchAnything pre-trained weights from the [Original Source](https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view?usp=sharing), place them under the weights directory, and unzip the file 


## Download of Evaluation Data
Download the contents of the `test_data` directory from the [AmalgaMatch Repository](http://dx.doi.org/10.24406/fordatis/436), place the zip file under `repo_directory/data/test_data`, and unzip the file


The data structure should looks like:
```
repo_directory/data/test_data
    - 5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale
    - AF9628-Martensitic_SEM-SE-Stitch_EBSD_SameSlice
    - ...
```

## Evaluation
For evaluation/inference run the inference_datasets.py code in the tools subfolder and change the CONFIG dict therein to specify a subset and MatchAnything variant.



# Acknowledgement
We thank the authors of for sharing their codes and model weights
- [MatchAnything](https://github.com/zju3dv/MatchAnything),
- [ELoFTR](https://github.com/zju3dv/EfficientLoFTR),
- [ROMA](https://github.com/Parskatt/RoMa)

The codes in this repository are directly derived from the [MatchAnything HuggingFace Space](https://huggingface.co/spaces/LittleFrog/MatchAnything/tree/main/imcui/third_party/MatchAnything)
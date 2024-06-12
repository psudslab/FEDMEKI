# FEDMEKI

This is the official website for FEDMEKI.



## Table of Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Environment Installation](#environment-installation)
- [Guideline](#Guideline)
- [Selected Results](#Selected-Results)
- [Others](#Others)
- [Acknowledgements](#acknowledgements)


## Overview



FEDMEKI is  a new benchmark designed to address the unique challenges of integrating medical knowledge into foundation models under privacy constraints. By leveraging a cross-silo federated learning approach, {\model} circumvents the issues associated with centralized data collection, which is often prohibited under health regulations like the Health Insurance Portability and Accountability Act (HIPAA) in the USA. The platform is meticulously designed to handle multi-site, multi-modal, and multi-task medical data.
<img src="/document/platform_overview.png" width="800">

The FedMEKI package contains:

- Data preprocessing scripts for multiple tasks.
- Implementation of multimodal multi-task federated learning baselines.
- Implementation of medical knowledge injection baselines.

In this package, you need to download the dataset and use the provided scripts to process the data.



## Datasets
We provide data processing scripts to process the data. The dataset processing scripts can be found in the folder `data_preprocess`.

Please note, all the datasets are governed by the corresponding license of their owners. 


## Environment Installation

Anaconda is recommended to install the environment. Anaconda is available at [Anaconda website](https://www.anaconda.com/products/distribution).
The name of the environment is `requirements.txt`. You can install the environment via:

```bash
git clone https://github.com/psudslab/FEDMEKI.git
cd FEDMEKI
conda create --name FEDMEKI
conda activate FEDMEKI
pip install -r requirements.txt
```

## Guideline

We provide the scripts of examples to help you get started with our platform. Also, we provide the key hyperparameters for you to adjust under different settings.

### Running Examples
The running scripts are shown step by step as below:

Step 1
```bash
cd FedMEKI/src/
```
Step 2: Training

```bash
bash /tools/FedMEKI/train.sh
```
the configuration regarding training can be found at FedMEKI/src/config/FedMEKI.yaml

Step 3: Evaluation
```bash 
bash /tools/FedMEKI/eval.sh

```
the configuration regarding evaluation can be directly found at FedMEKI/src/tools/FedMEKI/eval.sh




### Key Hyperparameters

We introduce key hyperpameters as below:

- `--epoch`: federated training epoch
- `--num_clients_values`: number of clients during local training 
- `--llm_ckpt_path`: the path of foundation model checkpoint that you would like to inject knowledge to  
- `--save_path`: the path that you would like to save your injected foundation model and local models to

For the training process, we adopt DeepSpeed (https://github.com/microsoft/DeepSpeed) for the computation acceleration. Please refer to their tutorial (https://www.deepspeed.ai/getting-started/) for relevant hyperparameter setting. 

## Seleted Results
We show the selected results as below. More details can be found in the paper.
- Benchmark performance of single-task evaluation for training tasks.
<img src="/document/single.png" width="800">

- Benchmark performance of multi-task evaluation for training tasks.
<img src="/document/multiple.png" width="800">

## Others
- We will keep maintaining this repository to cover more datasets, more modalities, and more tasks.
- If you have any questions, please open an issues at the repository.
- If you would like to contribute to this repository, please contact [Jiaqi Wang](mailto:jqwang@psu.edu) or [Xiaochen Wang](mailto:xcwang@psu.edu).
- This implementation is partially borrowed from Octavius (https://github.com/OpenGVLab/LAMM/tree/main/src/model/Octavius). We would like to extend our appreciation to the authors of this paper: <em><strong>Octavius: Mitigating Task Interference in MLLMs via LoRA-MoE</strong></em>

## Acknowledgements
FEDMEKI is partially supported by the National Science Foundation under Grant No. 2238275, 2333790, 2348541, and the National Institutes of Health under Grant No. R01AG077016.

Last updated: June 2024.



### CATH-ddG network

---

#### Description

This repo contains code for **CATH-ddG: towards robust mutation effect prediction on protein–protein interactions out of CATH homologous superfamily** by Guanglei Yu, Xuehua Bi, Teng Ma, Yaohang Li and Jianxin Wang.

We proposed a [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187)-inspired $\Delta\Delta G$ predictor using 3D structure and 2D sequences of wildtype $\mathcal{WT}$ and mutant $\mathcal{MT}$ protein complex as input. 

Overview of our **CATH-ddG** architecture is shown below.

<img src="./assets/cover.png" alt="cover" style="width:70%;" />



## Install

#### CATH-ddG Environment

```bash
conda env create -f env.yml
conda activate CATH-ddG
```

The default PyTorch version is 2.3.1, python version is 3.10, and cudatoolkit version is 11.3.1. They can be changed in [`env.yml`](./env.yml).

## Preparation of processed dataset

We generate the mutant structures and  prepare the processed dataset using the following command for SKEMP2 v2.0, HER2, and S285 dataset, respectively.

```bash
python skempi_parallel.py --reset --subset skempi_v2
python skempi_parallel.py --reset --subset HER2
python skempi_parallel.py --reset --subset S285
```

### Trained Weights

***

1. The  trained weights for held-out CATH test set is located in: [CATH_model_0](https://drive.google.com/file/d/1Hs2h_9VBNzFSZ37vDObZ8pu47gI-cwgq/view?usp=sharing), [CATH_model_1](https://drive.google.com/file/d/10HEIJM_NU5Cz-vEcqxYNCbYFdvbzZdv7/view?usp=sharing), [CATH_model_2](https://drive.google.com/file/d/13tAPMEnSLB9uvIWJpWv4JZoSoZvs3xZ_/view?usp=sharing)
2. The  trained weights for held-out PPIFORMER test set is located in: [PPIFORMER](https://drive.google.com/file/d/12XqQ3ucelemtaUB9MSARaKd7g49NZmi1/view?usp=sharing)
3. The  trained weights for case study is located in: [case study](https://drive.google.com/file/d/1oOJjA7Lp7xXQqN1ChNScXQ3YS7wfnmvd/view?usp=sharing)

### Usage

***

#### Evaluate CATH-ddG on held-out CATH test set

```bash
python test_DDAffinity.py ./configs/train/CATH.yml  --device cuda:0
```

#### Evaluate CATH-ddG on held-out PPIFORMER test set

```bash
python test_DDAffinity.py ./configs/train/PPIformer.yml  --device cuda:0
```

#### Evaluate CATH-ddG on RDE-Net work data splitting

```
python test_DDAffinity.py ./configs/train/Case_study.yml  --device cuda:0
```

#### Case Study 1: HER2

```bash
python case_study.py ./configs/inference/case_study_HER2.yml --device cuda:0
```

#### Case Study 2: S285

```
python case_study.py ./configs/inference/case_study_S285.yml --device cuda:0
```

#### Train CATH-ddG on held-out CATH data splitting

```bash
python train_DDAffinity.py ./configs/train/CATH.yml --device cuda:0
```
### Acknowledgements

***

We acknowledge that parts of our code is adapted from [Rotamer Density Estimator (RDE)](https://github.com/luost26/RDE-PPI). Thanks to the authors for sharing their codes. 
# FACT
Assignment for FACT UVA
=======
# FairMI - Reproducibility + Extension
This repository contains source code to AAAI-2023 paper and the extension for:

- Fair Representation Learning for Recommendation: A Mutual Information-Based Perspective

FairMI is a framework based Mutual Information (MI) for embedding fairness in recommendations,
which is optimized by the two-fold MI based objective. We have included the electronics dataset and two models:

1. AutoRec 
2. Non Matrix Factorization
3. Code missing for reproducibility
4. Electronics Dataset

>  **Note from original repo**:  Due to the limitations of github's file transfer, the complete training parameters are published in https://drive.google.com/drive/folders/14A4SqcPVFxBpFdX7TAL1BP_oShwl3kPK?usp=sharing



## Getting Started

### Train & Test

- Training BPRMF baseline on MovieLens: 

```shell
python train_bpr_baseline.py
```

- Training BPRMF baseline on LastFM:

```shell
python train_bpr_baseline_lastfm.py
```

- Training FairMI_BPR on MovieLens: 

```shell
python train_bpr_fairmi.py
```

- Training FairMI_BPR on LastFM:

```shell
python train_bpr_fairmi_lastfm.py
```

- Training GCN baseline on MovieLens: 

```shell
python train_gcn_baseline.py
```

- Training GCN baseline on LastFM:

```shell
python train_gcn_baseline_lastfm.py
```

- Training FairMI_GCN on MovieLens: 

```shell
python train_gcn_fairmi.py
```

- Training FairMI_GCN on LastFM:

```shell
python train_gcn_fairmi_lastfm.py
```

- Training BPRMF baseline with NMF and AutoRec on MovieLens: 

```shell
python train_bpr_baseline_nmf-autorec.py
```

- Training FairMI_BPR with NMF and AutoRec on MovieLens: 

```shell
python train_bpr_fairmi_nmf-autorec.py
```

- Training GCN baseline with NMF and AutoRec on MovieLens: 

```shell
python train_gcn_baseline_nmf-autorec.py
```

- Training FairMI_GCN with NMF and AutoRec on MovieLens: 

```shell
python train_gcn_fairmi_nmf-autorec.py
```

- Training FairMI_Star_GCN on MovieLens: 

```shell
python train_gcn_fairmi_star.py
```

- Training FairMI_Star_BPR on MovieLens: 

```shell
python train_bpr_fairmi_star.py
```

- Training FairMI_BPR_NoLowBound on MovieLens: 

```shell
python train_bpr_fairmi_nolowbound.py
```
- Training FairMI_BPR_NoUpBound on MovieLens: 

```shell
python train_bpr_fairmi_noupbound.py
```

- Training FairMI_GCN_NoLowBound on MovieLens: 

```shell
python train_gcn_fairmi_nolowbound.py
```
- Training FairMI_GCN_NoUpBound on MovieLens: 

```shell
python train_gcn_fairmi_noupbound.py
```

- Training BPRMF baseline on Electronics Dataset:

```shell
python train_bpr_baseline_electronics.py
```

- Training FairMI_BPR on Electronics Dataset:

```shell
python train_bpr_fairmi_electronics.py
```

- Training GCN baseline on Electronics Dataset:

```shell
python train_gcn_baseline_electronics.py
```

- Training FairMI_GCN on Electronics Dataset:

```shell
python train_gcn_fairmi_electrinics.py
```

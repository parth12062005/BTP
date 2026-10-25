# Complete Technical Report: Variant B 3-Loss `EVACLIPSharedHadamardMetaNet`, `best_baseline.pth` vs. `best_of_all_v2.pth` Evaluation, and 4-Block 5-Loss TTA Contribution & Weightage Ablation

---

## 1. Executive Summary

This report documents the mathematical formulation, architectural implementation, and complete experimental results of:
1. **Variant B Shared `1024 -> 768 -> 128` Hadamard `MetaNet` (`EVACLIPSharedHadamardMetaNet`)**:
   - Extracts native 1024D `EVA01-g-14` (`open_clip`, `laion400m_s11b_b41k`) prompt embeddings $\mathbf{t}_i^{\text{EVA}} \in \mathbb{R}^{1024}$ and visual `[CLS]` embeddings $\mathbf{v}_i^{\text{EVA}} \in \mathbb{R}^{1024}$ (via `eva_model.visual.trunk.head`), projects both through a **shared $1024 \to 768$ linear projection** fused with the Q-Former prompt and visual representations $\mathbf{t}_i^{\text{QF}}, \mathbf{v}_i^{\text{QF}} \in \mathbb{R}^{768}$, and maps both into a **shared 128D Hadamard space** ($\mathbf{h}_p(i), \mathbf{h}_v(i) \in \mathbb{R}^{128}$).
   - Trained on `EvalMI-Train` using **strictly the 3 specified losses**:
     $$\mathcal{L}_{\text{MetaNet}} = \mathcal{L}_{\text{MSE}}\!\left(\hat{y}_i^{\text{MG4}},\, y_i^{\text{GT}}\right) + 0.06\,\mathcal{L}_{\text{MOS-BT}}\!\left(\mathbf{H}_p, \mathbf{H}_v, \mathbf{y}^{\text{GT}}\right) + 0.12\,\mathcal{L}_{\cos\text{-Cal}}\!\left(\hat{y}_i^{\cos},\, y_i^{\text{GT}}\right)$$
2. **Side-by-Side Comparison of `best_baseline.pth` vs. `best_of_all_v2.pth` Under 2-Loss `Query_Only` Episodic TTA (`loss = l_prior + 0.5 * l_itc`)**:
   - Adapts **only the 32 Q-Former Query Tokens** ($\mathbf{Q} \in \mathbb{R}^{1 \times 32 \times 768}$, `24,576` parameters; all LayerNorms, Q-Former BERT layers, and `MetaNet` strictly frozen).
   - Evaluates both checkpoints on identical balanced `N = 800` splits (`EvalMI-Val 800`, `AGHI-QA 800`, `AGIQA-20K 800`, `Q-Eval 800`, `HPDv2 800`) and full splits (`AGIQA-20K-Full 1,320`, `Q-Eval-Sub 500`).
3. **Complete 4-Block Ablation of All 5 TTA Loss Components (`27` Unique Weight Combinations Across All 5 Balanced `N = 800` Datasets)**:
   - Dissects the general 5-term TTA objective:
     $$\mathcal{L}_{\text{TTA}}(\mathbf{Q}) = \lambda_{d10}\,\mathcal{L}_{\text{Desc}}^{(10)}(\mathbf{Q}) + \lambda_{d4}\,\mathcal{L}_{\text{Desc}}^{(4)}(\mathbf{Q}) + \lambda_{\text{ITC}}\,\mathcal{L}_{\text{ITC}}(\mathbf{Q}) + \lambda_{\text{Prior}}\,\mathcal{L}_{\text{Prior-MSE}}(\mathbf{Q}) + \lambda_{\text{Cos128}}\,\mathcal{L}_{\text{Cos128-TTA}}(\mathbf{Q})$$
   - Quantifies both **Pure Unblended Visual `pred_mm` (`Pure S15`)** and **Blended TTA (`TTA S5 / S15 / Best`)** across:
     - **Block 1**: Single-Loss Standalone Isolation (`1-by-1` Only)
     - **Block 2**: Cumulative Forward Addition & Leave-One-Out Attribution
     - **Block 3**: Fine-Grained Weight Sensitivity Grids (`3A`: Layer-10 vs. Layer-4 Ratio; `3B`: $\lambda_{\text{ITC}}$ Sweep; `3C`: $\lambda_{\text{Prior}}$ Sweep; `3D`: $\lambda_{\text{Cos128}}$ Sweep)
     - **Block 4**: Optimal Weight Vectors & Practical Takeaways

---

## 2. Part I — Mathematical Formulation of Variant B (`EVACLIPSharedHadamardMetaNet`)

### 2.1 Shared 1024D EVA-CLIP $\to$ 768D $\to$ 128D Hadamard Architecture
For each image-prompt pair $(I_i, P_i)$:
1. **Native 1024D EVA-CLIP Joint Embeddings**:
   $$\mathbf{t}_i^{\text{EVA}} = \frac{\text{Enc}_{\text{text}}^{\text{EVA}}(P_i)}{\|\text{Enc}_{\text{text}}^{\text{EVA}}(P_i)\|_2} \in \mathbb{R}^{1024}, \qquad \mathbf{v}_i^{\text{EVA}} = \frac{W_{\text{head}}^{\text{EVA}}\,\mathbf{v}_{i, \text{[CLS]}}^{\text{ViT}}}{\|W_{\text{head}}^{\text{EVA}}\,\mathbf{v}_{i, \text{[CLS]}}^{\text{ViT}}\|_2} \in \mathbb{R}^{1024}$$
   where $W_{\text{head}}^{\text{EVA}} \in \mathbb{R}^{1024 \times 1408}$ is `eva_model.visual.trunk.head` from `EVA01-g-14`.
2. **Shared $1024 \to 768$ Projection & Fusion with Q-Former Representations**:
   $$\tilde{\mathbf{t}}_i^{\text{EVA}} = \text{Norm}\!\left(\text{LN}_p^{(768)}\!\left(W_{1024 \to 768}\,\mathbf{t}_i^{\text{EVA}}\right)\right), \qquad \tilde{\mathbf{v}}_i^{\text{EVA}} = \text{Norm}\!\left(\text{LN}_v^{(768)}\!\left(W_{1024 \to 768}\,\mathbf{v}_i^{\text{EVA}}\right)\right)$$
   $$\mathbf{t}_i^{(768)} = \text{Norm}\!\left((1 - w_p)\tilde{\mathbf{t}}_i^{\text{EVA}} + w_p\,\mathbf{t}_i^{\text{QF}}\right), \qquad \mathbf{v}_i^{(768)} = \text{Norm}\!\left((1 - w_v)\tilde{\mathbf{v}}_i^{\text{EVA}} + w_v\,\mathbf{v}_i^{\text{QF}}\right)$$
   where $w_p = \sigma(\alpha_p), w_v = \sigma(\alpha_v)$ are learnable scalar gates.
3. **Shared $768 \to 128$ Hadamard Projection & Calibrated 128D Cosine Similarity**:
   $$\mathbf{h}_p(i) = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LN}_p^{(128)}\!\left(W_{768 \to 128}\,\mathbf{t}_i^{(768)} + 0.30\,R_p\,\mathbf{t}_i^{(768)}\right)\right)\right) \in \mathbb{R}^{128}$$
   $$\mathbf{h}_v(i) = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LN}_v^{(128)}\!\left(W_{768 \to 128}\,\mathbf{v}_i^{(768)} + 0.30\,R_v\,\mathbf{v}_i^{(768)}\right)\right)\right) \in \mathbb{R}^{128}$$
   Let $\hat{\mathbf{h}}_p(i) = \mathbf{h}_p(i)/\|\mathbf{h}_p(i)\|_2$ and $\hat{\mathbf{h}}_v(i) = \mathbf{h}_v(i)/\|\mathbf{h}_v(i)\|_2$. Then the 128D coordinate-wise cosine similarity and calibrated MOS prediction are:
   $$s_i^{\cos} = \sum_{c=1}^{128} \hat{h}_{p,c}(i)\,\hat{h}_{v,c}(i) \in [-1, 1], \qquad \hat{y}_i^{\cos} = \alpha_{\cos}\,s_i^{\cos} + \beta_{\cos}$$
4. **Hadamard Interaction & 4-Aspect Softmax Gate ($\tau = 0.65$)**:
   $$\mathbf{z}_{\text{inter}}(i) = \left(\sqrt{128}\,\hat{\mathbf{h}}_p(i)\right) \odot \left(\sqrt{128}\,\hat{\mathbf{h}}_v(i)\right) \in \mathbb{R}^{128}$$
   $$\mathbf{u}_i = \left[\sqrt{128}\hat{\mathbf{h}}_p(i)\;\middle\|\;\sqrt{128}\hat{\mathbf{h}}_v(i)\;\middle\|\;\mathbf{z}_{\text{inter}}(i)\;\middle\|\;s_i^{\cos}\;\middle\|\;\mathbf{s}_{i,\text{block}}^{(4)}\right] \in \mathbb{R}^{389}, \qquad \mathbf{w}_i^{\text{asp}} = \text{Softmax}\!\left(\frac{\text{MLP}_{\text{gate}}(\mathbf{u}_i)}{\tau}\right) \in \Delta^3$$
   $$\hat{y}_i^{\text{MG4}} = 0.92 \sum_{k=1}^{4} w_{i,k}^{\text{asp}}\,y_{i,k}^{\text{AE50-Fused}} + 0.08\,\hat{y}_i^{\cos}$$

### 2.2 Strictly 3-Loss `MetaNet` Training Objective
`EVACLIPSharedHadamardMetaNet` is trained on `EvalMI-Train` (`N = 800`, with Leave-One-Out gallery self-masking) using **strictly the 3 specified losses**:
$$\mathcal{L}_{\text{MetaNet}} = \underbrace{\frac{1}{B}\sum_{i=1}^{B}\left(\hat{y}_i^{\text{MG4}} - y_i^{\text{GT}}\right)^2}_{\text{1. }\mathcal{L}_{\text{MSE}}(\hat{y}_{\text{MG4}},\,y_{\text{GT}})} \;+\; \underbrace{0.06\left[\sum_{c=1}^{128}\left(1 - C_{c,c}^{(w)}\right)^2 + 0.02\sum_{c \neq c'}\left(C_{c,c'}^{(w)}\right)^2\right]}_{\text{2. }0.06\,\mathcal{L}_{\text{MOS-BT}}\text{ (MOS-Weighted }128\times 128\text{ Barlow Twins, }C_{c,c}^{(w)}\to 1\text{)}} \;+\; \underbrace{0.12\,\frac{1}{B}\sum_{i=1}^{B}\left(\hat{y}_i^{\cos} - y_i^{\text{GT}}\right)^2}_{\text{3. }0.12\,\mathcal{L}_{\cos\text{-Cal}}(\hat{y}_{\cos},\,y_{\text{GT}})}$$

---

## 3. Part II — Side-by-Side Comparison: `best_baseline.pth` vs. `best_of_all_v2.pth` Under 2-Loss `Query_Only` TTA (`loss = l_prior + 0.5 * l_itc`)

### 3.1 Table 1A — 128D Coordinate Alignment & Cosine Diagnostics on `best_baseline.pth`
*(From `final_baseline_evaclip_3loss_metanet_diagnostics.csv`)*

| Dataset | High-MOS Mean Cosine $\bar{s}_{\cos}^{\text{High}}$ | Low-MOS Mean Cosine $\bar{s}_{\cos}^{\text{Low}}$ | Separation $\Delta_{\text{High}-\text{Low}}$ | Standalone $s_i^{\cos}$ vs. MOS `SRCC / PLCC` | 128D Diagonal Channel Corr $\frac{1}{128}\sum_{c=1}^{128} C_{c,c}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`EvalMI-Val (800)`** | `+0.6924` | `+0.0725` | **`+0.6199`** | **`0.8146 / 0.8555`** | **`+0.7706`** |
| **`AGHI-QA (800)`** | `+0.6744` | `+0.4712` | **`+0.2032`** | **`0.4252 / 0.4423`** | **`+0.7150`** |
| **`AGIQA-20K (800)`** | `+0.4488` | `+0.1325` | **`+0.3163`** | **`0.7044 / 0.6673`** | **`+0.5877`** |
| **`Q-Eval (800)`** | `+0.6380` | `+0.5174` | **`+0.1206`** | **`0.3358 / 0.3237`** | **`+0.7357`** |
| **`HPDv2 (800)`** | `+0.2683` | `+0.1462` | **`+0.1221`** | **`0.5226 / 0.4730`** (**`78.71%` PairAcc vs. `78.30%` ZS**) | **`+0.6127`** |
| **`AGIQA-20K-Full (1320)`** | `+0.4694` | `+0.1303` | **`+0.3391`** | **`0.7073 / 0.6663`** | **`+0.5969`** |
| **`Q-Eval-Sub (500)`** | `+0.6368` | `+0.5112` | **`+0.1257`** | **`0.3547 / 0.3489`** | **`+0.7385`** |

### 3.2 Table 1B — 128D Coordinate Alignment & Cosine Diagnostics on `best_of_all_v2.pth`
*(From `final_best_of_all_v2_evaclip_3loss_metanet_diagnostics.csv`)*

| Dataset | High-MOS Mean Cosine $\bar{s}_{\cos}^{\text{High}}$ | Low-MOS Mean Cosine $\bar{s}_{\cos}^{\text{Low}}$ | Separation $\Delta_{\text{High}-\text{Low}}$ | Standalone $s_i^{\cos}$ vs. MOS `SRCC / PLCC` | 128D Diagonal Channel Corr $\frac{1}{128}\sum_{c=1}^{128} C_{c,c}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`EvalMI-Val (800)`** | `+0.6550` | `+0.0691` | **`+0.5859`** | **`0.8620 / 0.8400`** | **`+0.5314`** |
| **`AGHI-QA (800)`** | `+0.6019` | `+0.2947` | **`+0.3072`** | **`0.5128 / 0.5228`** | **`+0.4419`** |
| **`AGIQA-20K (800)`** | `+0.3722` | `+0.0627` | **`+0.3095`** | **`0.7085 / 0.6307`** | **`+0.4256`** |
| **`Q-Eval (800)`** | `+0.5480` | `+0.4067` | **`+0.1413`** | **`0.3751 / 0.3726`** *(beats Visual ZS `0.3638`!)* | **`+0.5382`** |
| **`HPDv2 (800)`** | `+0.1333` | `+0.0233` | **`+0.1100`** | **`0.5323 / 0.4494`** (**`79.89%` PairAcc vs. `79.18%` ZS!**) | **`+0.4398`** |
| **`AGIQA-20K-Full (1320)`** | `+0.3956` | `+0.0636` | **`+0.3320`** | **`0.7109 / 0.6266`** | **`+0.4317`** |
| **`Q-Eval-Sub (500)`** | `+0.5471` | `+0.4022` | **`+0.1449`** | **`0.3993 / 0.3963`** *(beats Visual ZS `0.3848`!)* | **`+0.5368`** |

---

### 3.3 Table 2A — 2-Loss `Query_Only` Episodic TTA on `best_baseline.pth` (`24,576` Params, `loss = l_prior + 0.5 * l_itc`)
*(From `final_baseline_evaclip_3loss_metanet_2loss_tta_summary.csv`)*

| Dataset | $N$ | Visual Zero-Shot (`SRCC / PLCC`) | Standalone $s_i^{\cos}$ (`SRCC`) | Step 0 (`SRCC / PLCC`) | Step 5 (`SRCC / PLCC`) | Step 10 (`SRCC / PLCC`) | Step 15 (`SRCC / PLCC`) | Step 25 (`SRCC / PLCC`) | Best TTA (`SRCC / PLCC`) | Gain vs. ZS (`Δ SRCC / Δ PLCC`) | Relative Gain (`% SRCC / % PLCC`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`EvalMI-Val (800)`** | `800` | `0.8653 / 0.9068` | `0.8146` | `0.8653 / 0.9068` | `0.8648 / 0.9062` | `0.8639 / 0.9056` | `0.8622 / 0.9041` | `0.8588 / 0.9012` | **`0.8653 / 0.9067`** | `+0.0000 / -0.0001` | **`+0.01% / -0.01%`** |
| **`AGHI-QA (800)`** | `800` | `0.5159 / 0.5090` | `0.4252` | **`0.5493 / 0.5550`** | `0.5453 / 0.5498` | `0.5466 / 0.5513` | `0.5451 / 0.5496` | `0.5403 / 0.5446` | **`0.5493 / 0.5550`** | **`+0.0334 / +0.0460`** | **`+6.47% / +9.03%`** |
| **`AGIQA-20K (800)`** | `800` | `0.7493 / 0.7526` | `0.7044` | **`0.7548 / 0.7601`** | `0.7536 / 0.7586` | `0.7526 / 0.7575` | `0.7495 / 0.7547` | `0.7408 / 0.7467` | **`0.7548 / 0.7601`** | **`+0.0054 / +0.0075`** | **`+0.72% / +0.99%`** |
| **`Q-Eval (800)`** | `800` | `0.3734 / 0.3562` | `0.3358` | **`0.3888 / 0.3704`** | `0.3849 / 0.3663` | `0.3826 / 0.3636` | `0.3792 / 0.3596` | `0.3712 / 0.3504` | **`0.3888 / 0.3704`** | **`+0.0155 / +0.0142`** | **`+4.14% / +3.99%`** |
| **`HPDv2 (800)`** | `800` | `0.6774 / 0.6390` | `0.5226` | **`0.6898 / 0.6535`** | `0.6852 / 0.6484` | `0.6817 / 0.6446` | `0.6771 / 0.6397` | `0.6660 / 0.6290` | **`0.6898 / 0.6535`** | **`+0.0124 / +0.0146`** | **`+1.83% / +2.28%`** |
| **`AGIQA-20K-Full (1320)`** | `1320` | `0.7364 / 0.7513` | `0.7073` | **`0.7454 / 0.7622`** | `0.7443 / 0.7607` | `0.7433 / 0.7596` | `0.7406 / 0.7570` | `0.7330 / 0.7498` | **`0.7454 / 0.7622`** | **`+0.0090 / +0.0109`** | **`+1.22% / +1.45%`** |
| **`Q-Eval-Sub (500)`** | `504` | `0.3914 / 0.3873` | `0.3547` | **`0.4048 / 0.4031`** | `0.4017 / 0.3991` | `0.3991 / 0.3963` | `0.3951 / 0.3924` | `0.3869 / 0.3834` | **`0.4048 / 0.4031`** | **`+0.0134 / +0.0158`** | **`+3.43% / +4.08%`** |

### 3.4 Table 2B — 2-Loss `Query_Only` Episodic TTA on `best_of_all_v2.pth` (`24,576` Params, `loss = l_prior + 0.5 * l_itc`)
*(From `final_best_of_all_v2_evaclip_3loss_metanet_2loss_tta_summary.csv`)*

| Dataset | $N$ | Visual Zero-Shot (`SRCC / PLCC`) | Standalone $s_i^{\cos}$ (`SRCC`) | Step 0 (`SRCC / PLCC`) | Step 5 (`SRCC / PLCC`) | Step 10 (`SRCC / PLCC`) | Step 15 (`SRCC / PLCC`) | Step 25 (`SRCC / PLCC`) | Best TTA (`SRCC / PLCC`) | Gain vs. ZS (`Δ SRCC / Δ PLCC`) | Relative Gain (`% SRCC / % PLCC`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`EvalMI-Val (800)`** | `800` | `0.8788 / 0.9117` | `0.8620` | `0.8788 / 0.9117` | `0.8786 / 0.9131` | `0.8780 / 0.9139` | `0.8760 / 0.9135` | `0.8648 / 0.9067` | **`0.8788 / 0.9126`** (Step 3) | `+0.0000 / +0.0009` | **`+0.00% / +0.10%`** |
| **`AGHI-QA (800)`** | `800` | `0.5252 / 0.5330` | `0.5128` | `0.5536 / 0.5683` | `0.5507 / 0.5629` | `0.5557 / 0.5666` | `0.5612 / 0.5693` | **`0.5704 / 0.5746`** | **`0.5704 / 0.5746`** (Step 25) | **`+0.0451 / +0.0416`** | **`+8.59% / +7.81%`** |
| **`AGIQA-20K (800)`** | `800` | `0.7647 / 0.7697` | `0.7085` | `0.7711 / 0.7798` | `0.7758 / 0.7827` | `0.7801 / 0.7855` | `0.7817 / 0.7860` | `0.7796 / 0.7815` | **`0.7820 / 0.7840`** (Step 20) | **`+0.0173 / +0.0144`** | **`+2.27% / +1.87%`** |
| **`Q-Eval (800)`** | `800` | `0.3638 / 0.3609` | `0.3751` | **`0.3855 / 0.3769`** | `0.3807 / 0.3709` | `0.3801 / 0.3679` | `0.3784 / 0.3634` | `0.3691 / 0.3530` | **`0.3855 / 0.3769`** (Step 0) / **`0.3807 / 0.3709`** (Step 5) | **`+0.0217 / +0.0160`** | **`+5.96% / +4.43%`** |
| **`HPDv2 (800)`** | `800` | `0.7179 / 0.6829` (`79.18%` PairAcc) | `0.5323` (**`79.89%` PairAcc**) | **`0.7264 / 0.6927`** | `0.7229 / 0.6886` | `0.7188 / 0.6855` | `0.7112 / 0.6786` | `0.6955 / 0.6646` | **`0.7264 / 0.6927`** (Step 0) / **`0.7230 / 0.6882`** (Step 3) | **`+0.0085 / +0.0098`** | **`+1.19% / +1.44%`** |
| **`AGIQA-20K-Full (1320)`** | `1320` | `0.7595 / 0.7735` | `0.7109` | `0.7655 / 0.7850` | `0.7700 / 0.7871` | `0.7737 / 0.7897` | `0.7756 / 0.7899` | `0.7731 / 0.7853` | **`0.7759 / 0.7882`** (Step 20) | **`+0.0164 / +0.0147`** | **`+2.16% / +1.90%`** |
| **`Q-Eval-Sub (500)`** | `504` | `0.3848 / 0.3951` | `0.3993` | **`0.4071 / 0.4134`** | `0.4031 / 0.4075` | `0.4012 / 0.4065` | `0.3980 / 0.4024` | `0.3861 / 0.3851` | **`0.4071 / 0.4134`** (Step 0) / **`0.4031 / 0.4075`** (Step 5) | **`+0.0223 / +0.0183`** | **`+5.79% / +4.63%`** |

---

## 4. Part III — Complete 4-Block Ablation of All 5 TTA Loss Components & Weightages

### 4.1 Mathematical Definition of All 5 Candidate TTA Losses on Query Tokens $\mathbf{Q}$
$$\mathcal{L}_{\text{TTA}}(\mathbf{Q}) = \lambda_{d10}\,\mathcal{L}_{\text{Desc}}^{(10)}(\mathbf{Q}) + \lambda_{d4}\,\mathcal{L}_{\text{Desc}}^{(4)}(\mathbf{Q}) + \lambda_{\text{ITC}}\,\mathcal{L}_{\text{ITC}}(\mathbf{Q}) + \lambda_{\text{Prior}}\,\mathcal{L}_{\text{Prior-MSE}}(\mathbf{Q}) + \lambda_{\text{Cos128}}\,\mathcal{L}_{\text{Cos128-TTA}}(\mathbf{Q})$$
1. **$\mathcal{L}_{\text{Desc}}^{(10)}(\mathbf{Q})$ (Deep Layer-10 Aspect-Weighted AE-50 Cosine Loss)**:
   $$\mathcal{L}_{\text{Desc}}^{(10)}(\mathbf{Q}) = \frac{1}{B}\sum_{i=1}^{B}\sum_{k=1}^{4} w_{i,k}^{\text{asp}}\!\left(1 - \cos\!\left(\hat{\mathbf{z}}_{i,k}^{(10)}(\mathbf{Q}),\, \mathbf{z}_{i,k}^{\text{AE50}}\right)\right)$$
2. **$\mathcal{L}_{\text{Desc}}^{(4)}(\mathbf{Q})$ (Shallow Layer-4 Aspect-Weighted AE-50 Cosine Loss)**:
   $$\mathcal{L}_{\text{Desc}}^{(4)}(\mathbf{Q}) = \frac{1}{B}\sum_{i=1}^{B}\sum_{k=1}^{4} w_{i,k}^{\text{asp}}\!\left(1 - \cos\!\left(\hat{\mathbf{z}}_{i,k}^{(4)}(\mathbf{Q}),\, \mathbf{z}_{i,k}^{\text{AE50}}\right)\right)$$
3. **$\mathcal{L}_{\text{ITC}}(\mathbf{Q})$ (256D Image-Text Contrastive Loss)**:
   $$\mathcal{L}_{\text{ITC}}(\mathbf{Q}) = \frac{1}{B}\sum_{i=1}^{B}\left(1 - \left\langle \frac{W_{\text{proj}}\bar{\mathbf{h}}_{\text{MM}, i}(\mathbf{Q})}{\|W_{\text{proj}}\bar{\mathbf{h}}_{\text{MM}, i}(\mathbf{Q})\|_2},\; \mathbf{t}_{\text{desc}, i}^{\text{fixed}} \right\rangle\right)$$
4. **$\mathcal{L}_{\text{Prior-MSE}}(\mathbf{Q})$ (In-Loop Quality Prior Anchor MSE Loss)**:
   $$\mathcal{L}_{\text{Prior-MSE}}(\mathbf{Q}) = \frac{1}{B}\sum_{i=1}^{B}\left(\hat{y}_{\text{MM}, i}(\mathbf{Q}) - \text{sg}\!\left[(1 - w_{\text{inloop}})\hat{y}_{\text{ZS}, i} + w_{\text{inloop}}\hat{y}_{\text{prior}, i}\right]\right)^2$$
5. **$\mathcal{L}_{\text{Cos128-TTA}}(\mathbf{Q})$ (Live 128D Hadamard Cosine Calibration Loss)**:
   $$\mathcal{L}_{\text{Cos128-TTA}}(\mathbf{Q}) = \frac{1}{B}\sum_{i=1}^{B}\left(\hat{y}_{\cos}\!\left(\mathbf{t}_i^{\text{EVA}}, \mathbf{v}_i^{\text{EVA}}, \mathbf{t}_i^{\text{QF}}, \bar{\mathbf{h}}_{\text{MM}, i}(\mathbf{Q})\right) - \text{sg}\!\left[\tilde{y}_{\text{inloop}, i}\right]\right)^2$$

---

### 4.2 Block 1 — Single-Loss Standalone Isolation (`1-by-1` Only, All Other 4 Weights $= 0$)
*(Evaluated across all 5 balanced `N = 800` splits on `best_of_all_v2.pth`; reporting both **Pure Unblended Visual `pred_mm` at Step 15** and **Blended TTA at Step 15**)*

| Exp ID | Single Active Loss $(\lambda_{d10}, \lambda_{d4}, \lambda_{\text{ITC}}, \lambda_{\text{Prior}}, \lambda_{\text{Cos128}})$ | `EvalMI-Val` `Pure / Best` (`ZS=0.8788`) | `AGHI-QA` `Pure S15 / TTA S15` (`ZS=0.5252`) | `AGIQA-20K` `Pure S15 / TTA S15` (`ZS=0.7647`) | `Q-Eval` `Pure S15 / TTA S15` (`ZS=0.3638`) | `HPDv2` `Pure S15 / TTA S15` (`ZS=0.7179`) | **Mean OOD Pure Gain (S15)** | **Mean OOD TTA Gain (S15)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.0`** | **Zero-Shot / Step 0** `(0, 0, 0, 0, 0)` | `0.8788 / 0.8788` | `0.5252 / 0.5536` | `0.7647 / 0.7711` | `0.3638 / 0.3855` | `0.7179 / 0.7264` | `+0.0000` | `+0.0163` *(S0)* |
| **`1.1`** | **Only $\mathcal{L}_{\text{Desc}}^{(10)}$** `(1, 0, 0, 0, 0)` | `0.8780 / 0.8788` | `0.5338 / 0.5600` | `0.7697 / 0.7757` | **`0.3713 / 0.3920`** | `0.7201 / 0.7267` | **`+0.0058`** | **`+0.0207`** |
| **`1.2`** | **Only $\mathcal{L}_{\text{Desc}}^{(4)}$** `(0, 1, 0, 0, 0)` | **`0.8792 / 0.8794`** | `0.5323 / 0.5599` | `0.7682 / 0.7743` | `0.3622 / 0.3833` | `0.7198 / 0.7274` | `+0.0027` | `+0.0183` |
| **`1.3`** | **Only $\mathcal{L}_{\text{ITC}}$** `(0, 0, 1, 0, 0)` | `0.8755 / 0.8788` | `0.5312 / 0.5592` | **`0.7793 / 0.7834`** | `0.3538 / 0.3766` | `0.6988 / 0.7091` | `-0.0021` | `+0.0142` |
| **`1.4`** | **Only $\mathcal{L}_{\text{Prior-MSE}}$** `(0, 0, 0, 1, 0)` | `0.8787 / 0.8788` | **`0.5458 / 0.5692`** | `0.7727 / 0.7714` | **`0.3850 / 0.4007`** | **`0.7307 / 0.7353`** | **`+0.0156`** | **`+0.0262`** |
| **`1.5`** | **Only $\mathcal{L}_{\text{Cos128-TTA}}$** `(0, 0, 0, 0, 1)` | `0.8691 / 0.8788` | `0.5378 / 0.5673` | `0.7294 / 0.7403` | `0.3690 / 0.3910` | `0.7209 / 0.7271` | `-0.0036` | `+0.0135` |

> [!IMPORTANT]
> **Key Scientific Insights from Block 1**:
> 1. **$\mathcal{L}_{\text{Prior-MSE}}$ Alone (`1.4`) vs. $\mathcal{L}_{\text{ITC}}$ Alone (`1.3`) Explains the Cross-Dataset Gradient Conflict**:
>    - **$\mathcal{L}_{\text{Prior-MSE}}$ alone (`1.4`)** achieves the **highest Mean OOD Pure Visual Gain (`+0.0156`)** and **highest Mean OOD TTA Gain (`+0.0262`)** across all single losses, boosting `AGHI-QA` to **`0.5692`** (`Pure = 0.5458`), `Q-Eval` to **`0.4007`** (`Pure = 0.3850`, **`+10.14%` over ZS!**), `HPDv2` to **`0.7353`** (`Pure = 0.7307`, **`+2.42%` over ZS!**), and `AGIQA-20K` Pure Visual to **`0.7727`**.
>    - **$\mathcal{L}_{\text{ITC}}$ alone (`1.3`)** is the single strongest driver on `AGIQA-20K` (`Pure = 0.7793`, `TTA = 0.7834`), **but actively degrades `Q-Eval` (`Pure = 0.3538`, `-0.0100` drop) and `HPDv2` (`Pure = 0.6988`, `-0.0191` drop)** because the 256D `text_proj` head was trained on `EvalMI` whole-paragraph syntax.
> 2. **$\mathcal{L}_{\text{Desc}}^{(10)}$ (`1.1`) Is the Most Generalizable Pure Semantic Loss**:
>    - Without any scalar target, **Layer-10 AE-50 (`1.1`)** improves Pure Visual `pred_mm` on **all 4 OOD datasets simultaneously** (`+0.0086` on `AGHI-QA`, `+0.0050` on `AGIQA-20K`, `+0.0075` on `Q-Eval`, `+0.0022` on `HPDv2`).

---

### 4.3 Block 2 — Cumulative Forward Addition & Leave-One-Out Attribution

| Exp ID | Configuration $(\lambda_{d10}, \lambda_{d4}, \lambda_{\text{ITC}}, \lambda_{\text{Prior}}, \lambda_{\text{Cos128}})$ | `EvalMI-Val` `Pure / Best` | `AGHI-QA` `Pure S15 / TTA S15` | `AGIQA-20K` `Pure S15 / TTA S15` | `Q-Eval` `Pure S15 / TTA S15` | `HPDv2` `Pure S15 / TTA S15` | **Mean OOD Pure Gain (S15)** | **Mean OOD TTA Gain (S15)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2.1`** | **Dual AE-50 Only** `(0.7, 0.3, 0, 0, 0)` | `0.8781 / 0.8788` | `0.5346 / 0.5616` | `0.7696 / 0.7754` | `0.3680 / 0.3887` | `0.7208 / 0.7273` | `+0.0053` | `+0.0203` |
| **`2.2`** | **2-Loss (`Prior + 0.5 ITC`)** `(0, 0, 0.5, 1.0, 0)` | `0.8760 / 0.8788` | `0.5336 / 0.5612` | `0.7776 / 0.7817` | `0.3558 / 0.3784` | `0.7015 / 0.7112` | `-0.0008` | `+0.0152` |
| **`2.3`** | **Dual AE-50 + ITC** `(0.7, 0.3, 0.5, 0, 0)` | `0.8758 / 0.8788` | `0.5300 / 0.5570` | **`0.7817 / 0.7857`** | `0.3576 / 0.3797` | `0.7042 / 0.7135` | `+0.0005` | `+0.0161` |
| **`2.4`** | **Dual AE-50 + Prior** `(0.7, 0.3, 0, 0.4, 0)` | **`0.8782 / 0.8788`** | `0.5363 / 0.5631` | `0.7694 / 0.7753` | `0.3697 / 0.3901` | **`0.7228 / 0.7287`** | `+0.0066` | `+0.0214` |
| **`2.5`** | **Full w/o $\mathcal{L}_{\text{ITC}}$** `(0.7, 0.3, 0, 0.4, 0.15)` | `0.8754 / 0.8788` | **`0.5400 / 0.5664`** | `0.7700 / 0.7756` | **`0.3722 / 0.3934`** | `0.7224 / 0.7282` | **`+0.0083`** | **`+0.0230`** |
| **`2.6`** | **Full 5-Loss ($\lambda_{\text{ITC}}=0.25$)** `(0.7, 0.3, 0.25, 0.4, 0.15)` | `0.8761 / 0.8788` | `0.5341 / 0.5611` | `0.7786 / 0.7824` | `0.3616 / 0.3835` | `0.7110 / 0.7185` | `+0.0034` | `+0.0185` |
| **`2.7`** | **Full w/o $\mathcal{L}_{\text{Desc}}^{(10)}$** `(0, 0.3, 0.25, 0.4, 0.15)` | `0.8759 / 0.8788` | `0.5341 / 0.5608` | `0.7787 / 0.7829` | `0.3565 / 0.3804` | `0.7059 / 0.7148` | `+0.0009` *(drops `-74%`)* | `+0.0168` |
| **`2.8`** | **Full w/o $\mathcal{L}_{\text{Desc}}^{(4)}$** `(0.7, 0, 0.25, 0.4, 0.15)` | `0.8755 / 0.8788` | `0.5343 / 0.5619` | `0.7799 / 0.7843` | `0.3640 / 0.3848` | `0.7084 / 0.7163` | `+0.0038` *(improves!)* | `+0.0189` |
| **`2.9`** | **Full w/o $\mathcal{L}_{\text{Prior}}$** `(0.7, 0.3, 0.25, 0, 0.15)` | `0.8758 / 0.8788` | `0.5328 / 0.5601` | `0.7788 / 0.7828` | `0.3611 / 0.3827` | `0.7087 / 0.7166` | `+0.0025` *(drops `-26%`)* | `+0.0177` |
| **`2.10`** | **Full w/o $\mathcal{L}_{\text{Cos128}}$** `(0.7, 0.3, 0.25, 0.4, 0)` | `0.8761 / 0.8788` | `0.5308 / 0.5576` | `0.7787 / 0.7827` | `0.3609 / 0.3822` | `0.7107 / 0.7181` | `+0.0024` *(drops `-29%`)* | `+0.0172` |

---

### 4.4 Block 3 — Fine-Grained Weight Sensitivity Grids (`3A`, `3B`, `3C`, `3D`)

#### **Block 3A: Layer-10 vs. Layer-4 Depth Ratio Sweep $(\lambda_{d10} : \lambda_{d4})$**
*(Fixed $\lambda_{\text{ITC}}=0.0, \lambda_{\text{Prior}}=0.40, \lambda_{\text{Cos128}}=0.15$)*

| Exp ID | Ratio $(\lambda_{d10} : \lambda_{d4})$ | `AGHI-QA` `Pure / TTA S15` | `AGIQA-20K` `Pure / TTA S15` | `Q-Eval` `Pure / TTA S15` | `HPDv2` `Pure / TTA S15` | **Mean OOD Pure Gain** | **Mean OOD TTA Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`3A.1`** | **`1.0 : 0.0` (Pure Layer 10)** | `0.5390 / 0.5653` | **`0.7702 / 0.7758`** | **`0.3767 / 0.3957`** | `0.7223 / 0.7278` | **`+0.0092`** | **`+0.0233`** |
| **`3A.2`** | **`0.7 : 0.3` (Default Dual)** | **`0.5400 / 0.5664`** | `0.7700 / 0.7756` | `0.3722 / 0.3934` | **`0.7224 / 0.7282`** | `+0.0083` | `+0.0230` |
| **`3A.3`** | **`0.5 : 0.5` (Equal)** | `0.5394 / 0.5659` | `0.7698 / 0.7756` | `0.3698 / 0.3909` | `0.7222 / 0.7281` | `+0.0074` | `+0.0222` |
| **`3A.4`** | **`0.3 : 0.7` (Shallow-Heavy)** | `0.5383 / 0.5649` | `0.7693 / 0.7752` | `0.3685 / 0.3891` | `0.7224 / 0.7279` | `+0.0067` | `+0.0214` |
| **`3A.5`** | **`0.0 : 1.0` (Pure Layer 4)** | `0.5361 / 0.5632` | `0.7682 / 0.7742` | `0.3661 / 0.3869` | `0.7227 / 0.7282` | `+0.0054` | `+0.0202` |

#### **Block 3B: $\lambda_{\text{ITC}}$ Weight Sweep**
*(Fixed $\lambda_{d10}=0.7, \lambda_{d4}=0.3, \lambda_{\text{Prior}}=0.40, \lambda_{\text{Cos128}}=0.15$)*

| Exp ID | $\lambda_{\text{ITC}}$ Weight | `AGHI-QA` `Pure / TTA S15` | `AGIQA-20K` `Pure / TTA S15` | `Q-Eval` `Pure / TTA S15` | `HPDv2` `Pure / TTA S15` | **Mean OOD Pure Gain** | **Mean OOD TTA Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`3B.1`** | **$\lambda_{\text{ITC}} = 0.00$** | **`0.5400 / 0.5664`** | `0.7700 / 0.7756` | **`0.3722 / 0.3934`** | **`0.7224 / 0.7282`** | **`+0.0083`** | **`+0.0230`** |
| **`3B.2`** | **$\lambda_{\text{ITC}} = 0.10$** | `0.5368 / 0.5631` | `0.7741 / 0.7789` | `0.3655 / 0.3860` | `0.7179 / 0.7245` | `+0.0057` | `+0.0202` |
| **`3B.3`** | **$\lambda_{\text{ITC}} = 0.25$** | `0.5341 / 0.5611` | `0.7786 / 0.7824` | `0.3616 / 0.3835` | `0.7110 / 0.7185` | `+0.0034` | `+0.0185` |
| **`3B.4`** | **$\lambda_{\text{ITC}} = 0.50$** | `0.5323 / 0.5598` | `0.7809 / 0.7853` | `0.3593 / 0.3809` | `0.7060 / 0.7146` | `+0.0017` | `+0.0172` |
| **`3B.5`** | **$\lambda_{\text{ITC}} = 1.00$** | `0.5323 / 0.5595` | **`0.7827 / 0.7861`** | `0.3566 / 0.3794` | `0.7031 / 0.7120` | `+0.0008` | `+0.0163` |

#### **Block 3C: $\lambda_{\text{Prior}}$ Weight Sweep (In-Loop Quality Prior Anchor)**
*(Fixed $\lambda_{d10}=0.7, \lambda_{d4}=0.3, \lambda_{\text{ITC}}=0.00, \lambda_{\text{Cos128}}=0.15$)*

| Exp ID | $\lambda_{\text{Prior}}$ Weight | `EvalMI-Val` `Pure S15` | `AGHI-QA` `Pure / TTA S15` | `AGIQA-20K` `Pure / TTA S15` | `Q-Eval` `Pure / TTA S15` | `HPDv2` `Pure / TTA S15` | **Mean OOD Pure Gain** | **Mean OOD TTA Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`3C.1`** | **$\lambda_{\text{Prior}} = 0.00$** | `0.8749` | `0.5388 / 0.5654` | **`0.7701 / 0.7760`** | `0.3716 / 0.3925` | `0.7206 / 0.7273` | `+0.0074` | `+0.0224` |
| **`3C.2`** | **$\lambda_{\text{Prior}} = 0.20$** | `0.8751` | `0.5395 / 0.5660` | `0.7701 / 0.7759` | `0.3719 / 0.3930` | `0.7219 / 0.7278` | `+0.0079` | `+0.0228` |
| **`3C.3`** | **$\lambda_{\text{Prior}} = 0.40$** | `0.8754` | `0.5400 / 0.5664` | `0.7700 / 0.7756` | `0.3722 / 0.3934` | **`0.7224 / 0.7282`** | `+0.0083` | **`+0.0230`** |
| **`3C.4`** | **$\lambda_{\text{Prior}} = 0.80$** | **`0.8758`** | **`0.5409 / 0.5668`** | `0.7698 / 0.7753` | **`0.3731 / 0.3935`** | **`0.7229 / 0.7278`** | **`+0.0088`** | `+0.0229` |

#### **Block 3D: $\lambda_{\text{Cos128}}$ Weight Sweep (Live 128D Hadamard Cosine Loss)**
*(Fixed $\lambda_{d10}=0.7, \lambda_{d4}=0.3, \lambda_{\text{ITC}}=0.00, \lambda_{\text{Prior}}=0.40$)*

| Exp ID | $\lambda_{\text{Cos128}}$ Weight | `AGHI-QA` `Pure / TTA S15` | `AGIQA-20K` `Pure / TTA S15` | `Q-Eval` `Pure / TTA S15` | `HPDv2` `Pure / TTA S15` | **Mean OOD Pure Gain** | **Mean OOD TTA Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`3D.1`** | **$\lambda_{\text{Cos128}} = 0.00$** | `0.5363 / 0.5631` | `0.7694 / 0.7753` | `0.3697 / 0.3901` | **`0.7228 / 0.7287`** | `+0.0066` | `+0.0214` |
| **`3D.2`** | **$\lambda_{\text{Cos128}} = 0.05$** | `0.5379 / 0.5643` | `0.7696 / 0.7754` | `0.3709 / 0.3913` | `0.7226 / 0.7287` | `+0.0073` | `+0.0220` |
| **`3D.3`** | **$\lambda_{\text{Cos128}} = 0.15$** | `0.5400 / 0.5664` | `0.7700 / 0.7756` | **`0.3722 / 0.3934`** | `0.7224 / 0.7282` | `+0.0083` | `+0.0230` |
| **`3D.4`** | **$\lambda_{\text{Cos128}} = 0.35$** | **`0.5422 / 0.5685`** | **`0.7703 / 0.7760`** | **`0.3739 / 0.3932`** | `0.7222 / 0.7274` | **`+0.0092`** | **`+0.0234`** |

---

### 4.5 Block 4 — Summary of Optimal Weightages

| Regime / Goal | Optimal Weight Tuple $(\lambda_{d10}, \lambda_{d4}, \lambda_{\text{ITC}}, \lambda_{\text{Prior}}, \lambda_{\text{Cos128}})$ | `AGHI-QA (800)` | `AGIQA-20K (800)` | `Q-Eval (800)` | `HPDv2 (800)` | **Mean OOD Pure Gain (S15)** | **Mean OOD TTA Gain (S15)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Minimal 1-Loss Prior-Only TTA (`Exp 1.4`)** | `(0.0, 0.0, 0.0, 1.0, 0.0)` | **`0.5692`** (`+8.38%`) | `0.7714` (`+0.88%`) | **`0.4008`** (**`+10.17%`**) | **`0.7353`** (**`+2.42%`**) | **`+0.0156`** | **`+0.0262`** |
| **2. Uniform Semantic + Prior + 128D Cosine (`Exp 3A.1 / 3D.4`)** | `(1.0, 0.0, 0.0, 0.40, 0.15)` or `(0.7, 0.3, 0.0, 0.40, 0.35)` | **`0.5685`** (`+8.24%`) | **`0.7760`** (`+1.48%`) | **`0.3957`** (`+8.77%`) | **`0.7282`** (`+1.43%`) | **`+0.0092`** | **`+0.0234`** |
| **3. Moderate-Shift `AGIQA-20K` Specialist (`Exp 2.3 / 3B.5`)** | `(0.7, 0.3, 0.50, 0.0, 0.0)` or `(0.7, 0.3, 1.0, 0.40, 0.15)` | `0.5595` | **`0.7861`** (**`+2.80%`**) | `0.3855` | `0.7264` | `+0.0008` | `+0.0163` |

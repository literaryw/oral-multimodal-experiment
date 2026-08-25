# Oral Multimodal Experiment

Experiment code for pathology image + clinical text multimodal fusion in oral cancer grading (TCGA-HNSC).

单模态与多模态融合对比实验：病理图像特征 (UNI, ViT-L/16) + 临床文本特征 (BiomedBERT)，二分类 Low-Grade (G1+G2) vs High-Grade (G3+G4)，全部结果基于逐模型随机种子重置的 5-fold 分层交叉验证。

## 仓库布局

```
code/     # 全部实验脚本（执行顺序见下）
data/     # 数据目录（不入库，运行 Step 1 后自动生成）
```

> 注意：脚本内部以 `code/` 的上级目录为数据根（`BASE_DIR/data/TCGA-HNSC/`），请保持本仓库目录结构不变，在仓库根下自建 `data/` 目录。

## 环境配置

```powershell
conda create -n tcga-oscc python=3.12 -y
conda activate tcga-oscc
pip install -r code\requirements_full.txt
```

详见 `code/environment.md`。

## 执行顺序

按以下顺序运行脚本，每一步产出物是下一步的输入：

### Step 1: 数据下载

```powershell
# 下载TCGA-HNSC临床数据(506例)
python code\download_tcga.py

# 下载预提取UNI特征(需HuggingFace登录)
python code\download_pretrained_features.py
```

产出: `data/TCGA-HNSC/clinical_data.csv`, `data/TCGA-HNSC/pretrained_features/*.parquet`

### Step 2: 特征提取

```powershell
# BiomedBERT文本特征提取 (506×768)
python code\extract_text_features.py

# 整合UNI图像特征 + 文本特征 + 标签 (499×1024 + 499×768)
python code\integrate_features.py
```

产出: `data/TCGA-HNSC/features/`, `data/TCGA-HNSC/integrated/`

### Step 3: 实验

```powershell
# 主实验: 多模态融合 (4模型 × 5折CV, 二分类)
python code\multimodal_fusion_v2.py

# 文本单模态基线
python code\text_baseline.py

# Modality Dropout 消融
python code\modality_dropout_experiment.py

# DNDF vs MLP 4-class对比
python code\dndf_experiment.py
```

产出: 实验日志和指标输出到终端

### Step 4: 真实性审计

```powershell
# 真实Grad-CAM热图 (替代随机噪声)
python code\fix_real_gradcam.py

# 真实Attention权重 (替代硬编码)
python code\fix_real_attention.py

# 真实DNDF参数 + 互信息计算
python code\fix_dndf_and_mi.py

# 保存所有实验输出为JSON
python code\save_experiment_outputs.py
```

产出: `data/TCGA-HNSC/experiment_outputs/`

### Step 5: 论文配图

```powershell
# Fig1-4: 架构图/混淆矩阵/性能对比/消融
python code\generate_figures.py
```

产出: 配图输出目录

### Step 6: 统计检验与稳健性（终稿协议）

```powershell
# 主表多种子稳健性: 3种子完整重跑, seed=42应逐位复现主表
python code\main_multiseed.py

# 逐样本精确McNemar+Bonferroni (三个嵌套队列)
python code\compute_mcnemar_exact.py
python code\compute_mcnemar_full_wide.py

# 留一解剖组(LOSO)与学习曲线多种子
python code\leave_one_site_out.py
python code\learning_curves_multiseed.py
```

产出: `data/TCGA-HNSC/experiment_outputs/*.json`（所有模型CV前逐模型重置随机种子, 协议详见各脚本头部注释）

## 文件清单

| 脚本                              | 阶段 | 功能                         |
| ----------------------------------- | ------ | ------------------------------ |
| `download_tcga.py`                | 数据 | TCGA-HNSC临床数据下载        |
| `download_pretrained_features.py` | 数据 | UNI/GatorTron预提取特征下载  |
| `refilter_oral_subsite.py`        | 数据 | 口腔亚组重过滤(GDC API)      |
| `map_oral_patches.py`             | 数据 | 口腔case→patch索引映射       |
| `extract_text_features.py`        | 特征 | BiomedBERT文本特征(506×768) |
| `integrate_features.py`           | 特征 | 图像+文本+标签整合(499例)    |
| `multimodal_fusion_v2.py`         | 实验 | 多模态融合主实验(4模型×5CV) |
| `text_baseline.py`                | 实验 | 文本单模态基线               |
| `dndf_experiment.py`              | 实验 | DNDF vs MLP 4-class对比      |
| `modality_dropout_experiment.py`  | 实验 | Modality Dropout消融         |
| `modality_dropout_oral.py`        | 实验 | 口腔亚组模态丢弃消融         |
| `modality_dropout_full.py`        | 实验 | 全队列模态丢弃消融(统一协议) |
| `text_ablation_full.py`           | 实验 | 全队列文本消融(统一协议)     |
| `tnm_sensitivity.py`              | 实验 | TNM泄漏敏感性分析            |
| `learning_curves_multiseed.py`    | 实验 | 学习曲线多种子重跑(5种子)    |
| `leave_one_site_out.py`           | 实验 | 留一解剖组LOSO验证           |
| `main_multiseed.py`               | 实验 | 主表多种子稳健性(42/123/456) |
| `oral_experiments_full.py`        | 实验 | 口腔亚组综合实验套件         |
| `compute_mcnemar_exact.py`        | 统计 | 逐样本McNemar(口腔n=246)     |
| `compute_mcnemar_full_wide.py`    | 统计 | 逐样本McNemar(全499/宽332)   |
| `diagnose_crossattn_auc.py`       | 诊断 | CrossAttn AUC根因(ECE校准)   |
| `fix_real_gradcam.py`             | 审计 | 真实Grad-CAM生成             |
| `fix_real_attention.py`           | 审计 | 真实Attention权重提取        |
| `fix_dndf_and_mi.py`              | 审计 | 真实DNDF参数+互信息          |
| `gradcam_oral_expanded.py`        | 审计 | Grad-CAM扩展(G1/G3+忠实性)   |
| `save_experiment_outputs.py`      | 审计 | 实验输出JSON导出             |
| `generate_figures.py`             | 配图 | Fig1-4生成 (从JSON加载)      |

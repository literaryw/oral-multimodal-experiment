# 项目环境记录

## 环境信息

| 项目 | 值 |
|------|-----|
| Conda环境名 | `tcga-oscc` |
| Python版本 | 3.12.13 |
| PyTorch | 2.12.0 (CPU) |
| CUDA | 不适用 (CPU版) |

## 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.12.0 | 深度学习框架 |
| torchvision | 0.27.0 | 视觉模型 |
| transformers | 5.11.0 | BERT/BiomedBERT文本编码 |
| timm | 1.0.27 | UNI/ViT视觉模型 |
| scikit-learn | 1.9.0 | 交叉验证/评估指标/MI计算 |
| pandas | 3.0.3 | 数据处理 |
| numpy | 2.4.6 | 数值计算 |
| matplotlib | 3.10.9 | 论文配图生成 |
| seaborn | 0.13.2 | 统计可视化 |
| opencv-python | 4.13.0 | 图像patch处理 |
| huggingface-hub | 1.19.0 | HuggingFace模型/数据集下载 |
| requests | 2.34.2 | HTTP/TGCA API下载 |
| certifi | 2026.5.20 | SSL证书验证 |

## 一键重建环境

```powershell
# 创建环境
conda create -n tcga-oscc python=3.12 -y

# 激活并安装
conda activate tcga-oscc
pip install torch torchvision transformers datasets accelerate timm opencv-python scikit-learn pandas matplotlib seaborn requests huggingface-hub certifi

# 验证
python -c "import torch,transformers,timm,sklearn; print('OK')"
```

或从快照安装:

```powershell
conda create -n tcga-oscc python=3.12 -y
conda activate tcga-oscc
pip install -r code\requirements_full.txt
```

## 完整依赖快照

完整74个包的列表见 `code/requirements_full.txt`。

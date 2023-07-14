# 多模态情感分析

## 实验任务

- 给定配对的文本和图像，预测对应的情感标签。
- 三分类任务：positive, neutral, negative。

## 实验数据集

- data文件夹：包括所有的训练文本和图片，每个文件按照唯一的guid命名。
- train.txt: 数据的guid和对应的情感标签。
- test_without_label.txt：数据的guid和空的情感标签。

## 文件结构

```
|-- ablation_image.py		运行消融实验模型（仅图像）
|-- ablation_text.py		运行消融实验模型（仅文本）
|-- multi.py				运行多模态模型
|-- multi_image_only_model.py	消融实验（仅图像）调用模型
|-- multi_model.py			多模态调用模型
|-- multi_text_only_model.py	消融实验（仅文本）调用模型
|-- README.md
|-- requirements.txt
|-- test_predictions.txt	测试结果文件
|-- trial_img.py			仅图像单模态运行	
|-- trial_txt.py			仅文本单模态运行
|-- 实验报告.md
|-- 实验报告.pdf
|-- dataset
	|-- backup
		|-- 备份一些文件（无关紧要）
	|-- data
		|-- 数据文件
	|-- test_without_label.txt
	|-- train.txt
|-- images
	|-- 实验报告中用到的图片
|-- output
	|-- 每个测试中的最佳结果以及评价指标变化
```

## 参考库

```python
import os
import chardet
import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision.models import resnet50,resnet101
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from multi_model import MultimodalSentimentAnalysisModel_add,MultimodalSentimentAnalysisModel_atten,MultimodalSentimentAnalysisModel_cat_direct,MultimodalSentimentAnalysisModel_cat_trans
import argparse
```

运行消融实验（其他参数参照代码内容）：

```
python ablation_image.py --ls 0.0001
python ablation_text.py --ls 0.0001
```

运行多模态（其他参数参照代码内容）：

```
python multi.py --ls 0.0001
```

运行单模态（其他参数参照代码内容）：

```
python trial_img.py --ls 0.0001
python trial_txt.py --ls 0.0001
```

# AI-Lab5-Multimodal-sentiment-analysis

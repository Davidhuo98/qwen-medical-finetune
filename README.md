# Qwen3.5-4B 微调实战：LLaMA-Factory 打造医疗AI助手

核心工具链：LLaMA-Factory + [Qwen3.5-4B](https://zhida.zhihu.com/search?content_id=272182167&content_type=Article&match_order=1&q=Qwen3.5-4B&zhida_source=entity) + 医疗问答数据集

Qwen3.5 是阿里最新发布的千问系列模型，4B 参数量刚好卡在"效果够用 + 显存友好"的甜蜜点；LLaMA-Factory 则是目前开源社区最成熟的微调框架，上手简单，坑也相对少。

## 准备工作

先说硬件要求。4B 模型用 [LoRA](https://zhida.zhihu.com/search?content_id=272182167&content_type=Article&match_order=1&q=LoRA&zhida_source=entity) 微调的话，一张 12GB 显存的显卡就够了（比如 RTX 4070）。如果手头只有 8GB 显存的卡，可以上 [QLoRA](https://zhida.zhihu.com/search?content_id=272182167&content_type=Article&match_order=1&q=QLoRA&zhida_source=entity) 量化方案，牺牲一点精度换显存空间。

| 微调方式      | 4B 模型显存需求 | 推荐显卡            |
| ------------- | --------------- | ------------------- |
| LoRA (16-bit) | ~10-12 GB       | RTX 4070 / RTX 3090 |
| QLoRA (8-bit) | ~6-8 GB         | RTX 4060 / RTX 3070 |
| QLoRA (4-bit) | ~4-6 GB         | RTX 3060            |

软件环境这边，建议 Python 3.11+，[PyTorch](https://zhida.zhihu.com/search?content_id=272182167&content_type=Article&match_order=1&q=PyTorch&zhida_source=entity) 2.0 以上。CUDA 版本最好 12.x，兼容性更好。

## 搭建 LLaMA-Factory 环境

LLaMA-Factory 的安装很直接：

```plain
# 克隆仓库 
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git 
cd LLaMA-Factory
# 安装依赖 
pip install -e . 
pip install -r requirements/metrics.txt 
# 如果需要 DeepSpeed 加速（可选） 
pip install -r requirements/deepspeed.txt
```

装完之后可以跑一下测试命令确认环境没问题：

```
llamafactory-cli version
```

## 下载 Qwen3.5-4B 模型

模型从魔搭社区下载，国内速度很快：

```plain
# 安装 modelscope 
pip install modelscope 
# 方式一：Python 代码下载 
from modelscope import snapshot_download 
model_dir = snapshot_download('Qwen/Qwen3.5-4B') 
print(f"模型已下载到: {model_dir}") 
# 方式二：命令行下载 
modelscope download --model Qwen/Qwen3.5-4B --local_dir ./models/Qwen3.5-4B
```

💡 小贴士：模型大约 9.3GB，建议挂个代理或者选择网络好的时间段下载。下载完记得检查一下文件完整性。

## 准备医疗数据集

数据集是微调效果的关键。这里用的是开源的中文医疗问答数据，主要来自几个渠道：

**数据来源推荐：**

GitHub 上有个 llm-medical-data 仓库，整理了几十万条医疗问答数据，覆盖内科、外科、妇产科、儿科等科室。另外 [HuggingFace](https://zhida.zhihu.com/search?content_id=272182167&content_type=Article&match_order=1&q=HuggingFace&zhida_source=entity) 上的 shibing624/medical 数据集也不错，格式比较规范。下载后某些数据集还需要进行格式转换，匹配相应的训练格式。

```
git clone https://github.com/donote/llm-medical-data
git clone https://huggingface.co/datasets/shibing624/medical
```

### 数据集文件清单

| 文件                                                  | 来源                          | 类型 |
| ----------------------------------------------------- | ----------------------------- | ---- |
| andriatria_男科.json                                  | CSV转换                       | SFT  |
| IM_内科.json                                          | CSV转换                       | SFT  |
| obgyn_妇产科.json                                     | CSV转换                       | SFT  |
| oncology_肿瘤科.json                                  | CSV转换                       | SFT  |
| pediatric_儿科.json                                   | CSV转换                       | SFT  |
| surgical_外科.json                                    | CSV转换                       | SFT  |
| GenMedGPT-5k.json                                     | doctorchat                    | SFT  |
| HealthCareMagic-100k.json                             | doctorchat                    | SFT  |
| dialogue_task.json                                    | medicalgpt                    | SFT  |
| dialogue_seed_task.json                               | medicalgpt                    | SFT  |
| prepared_generated_data_for_nhs_uk_qa.json            | opengpt                       | SFT  |
| prepared_generated_data_for_medical_tasks.json        | opengpt                       | SFT  |
| prepared_generated_data_for_nhs_uk_conversations.json | opengpt（多轮拆分）           | SFT  |
| llama_data.json                                       | huatuo_llama_med_chinese_data | SFT  |

LLaMA-Factory 要求的数据格式是这样的（Alpaca格式）：

```plain
[ 
{ 
"instruction": "你是一个专业的医疗助手，请根据患者描述给出建议。", 
"input": "我最近总是头痛，尤其是下午的时候，已经持续一周了。", 
"output": "持续性头痛需要关注。建议您先排除以下几个常见原因：1）睡眠质量，2）用眼过度，3）颈椎问题。如果休息后仍不缓解，建议到医院神经内科就诊，做个头部CT检查。" 
}, 
{ 
"instruction": "你是一个专业的医疗助手，请根据患者描述给出建议。", 
"input": "孩子3岁，发烧38.5度，要不要去医院？", 
"output": "38.5度属于中度发热。可以先在家观察处理：1）物理降温，温水擦拭腋下、腹股沟；2）保持室内通风；3）多喝水。如果体温超过39度、持续超过3天、或出现精神萎靡、抽搐等情况，需要立即就医。" 
},
... 
]
```

解压项目内的压缩包，有些是csv文件，需要转换成json文件：

```bash
cd llm-medical-data/scripts
pip install -r requirements.txt

# 批量转换中文CSV
for f in ../chinese_medical_dialogue_data/*.csv; do
    python csv2json_chinese_medical_dialogue_data.py --rd_csv_path "$f"
done

# 转换opengpt数据
python csv2json_opengpt_data.py tqa \
    --rd_csv_path ../opengpt_data/prepared_generated_data_for_nhs_uk_qa.csv

python csv2json_opengpt_data.py ttask \
    --rd_csv_path ../opengpt_data/prepared_generated_data_for_medical_tasks.csv

# tchat为补全方法，拆分多轮对话为单轮
python csv2json_opengpt_data.py tchat \
    --rd_csv_path ../opengpt_data/prepared_generated_data_for_nhs_uk_conversations.csv
```

把处理好的数据放到 data/ 目录下，然后在 data/dataset_info.json 里注册：

```plain
{
  "andriatria_男科": {
    "file_name": "andriatria_男科.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "IM_内科": {
    "file_name": "IM_内科.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "obgyn_妇产科": {
    "file_name": "obgyn_妇产科.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "oncology_肿瘤科": {
    "file_name": "oncology_肿瘤科.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "pediatric_儿科": {
    "file_name": "pediatric_儿科.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "surgical_外科": {
    "file_name": "surgical_外科.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "dialogue_seed_task": {
    "file_name": "dialogue_seed_task.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "dialogue_task": {
    "file_name": "dialogue_task.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "GenMedGPT_5k": {
    "file_name": "GenMedGPT-5k.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "HealthCareMagic_100k": {
    "file_name": "HealthCareMagic-100k.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "opengpt_qa": {
    "file_name": "prepared_generated_data_for_nhs_uk_qa.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "opengpt_task": {
    "file_name": "prepared_generated_data_for_medical_tasks.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "opengpt_chat": {
    "file_name": "prepared_generated_data_for_nhs_uk_conversations.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "llama_data": {
    "file_name": "llama_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }

}

```

## 开始微调

配置文件是整个流程的核心。在 examples/train_lora/ 下创建一个 qwen35_medical_lora.yaml：

```plain
### 模型配置 ###
model_name_or_path: /root/autodl-fs/LLaMA-Factory/models/Qwen3.5-4B
trust_remote_code: true
### 微调方法 ###
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all
 ### 数据集配置 ###
dataset: andriatria_男科,IM_内科,obgyn_妇产科,oncology_肿瘤科,pediatric_儿科,surgical_外科,dialogue_seed_task,dialogue_task,GenMedGPT_5k,HealthCareMagic_100k,opengpt_qa,opengpt_task,opengpt_chat,llama_data
template: qwen3
cutoff_len: 2048
preprocessing_num_workers: 8
default_system: "你是一位专业的医生，请详细、耐心地回答患者的问题，包括病因分析、治疗建议、注意事项等方面。"
### 训练参数 ###
output_dir: ./output/qwen35_medical_lora
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
logging_steps: 10
save_steps: 500
### 显存优化 ###
bf16: true
gradient_checkpointing: true
max_samples: 1000

```

**参数解读：**
- `lora_rank: 64` —— LoRA 秩，越大表达能力越强，但显存占用也越大
- `lora_target: all` —— 对所有线性层应用 LoRA，效果更好
- `gradient_checkpointing: true` —— 用时间换空间，降低显存占用

一切就绪，启动训练：

```plain
llamafactory-cli train \
examples/train_lora/qwen35_medical_lora.yaml
```

## 测试效果

训练完成后，先在命令行跑个快速测试：

```plain
llamafactory-cli chat \
examples/inference/qwen35_medical_lora.yaml
```

对应的推理配置文件 qwen35_medical_lora.yaml：

```plain
model_name_or_path: ./models/Qwen3.5-4B 
adapter_name_or_path: ./output/qwen35_medical_lora 
template: qwen3 
finetuning_type: lora
```

实测下来，微调后的模型在医疗问答上明显比原版更专业。比如问"孕妇能不能吃螃蟹"，原版模型可能给个模棱两可的回答，微调后的版本会从中医寒凉属性、现代营养学、个体差异等多个角度分析，更像一个有经验的医生。

## 导出和部署

如果效果满意，可以把 LoRA 权重合并到基座模型里，方便后续部署：

```plain
llamafactory-cli export \
examples/merge_lora/qwen35_medical_merge.yaml
```

合并配置：

```plain
model_name_or_path: /root/autodl-fs/LLaMA-Factory/models/Qwen3.5-4B
adapter_name_or_path: /root/autodl-fs/LLaMA-Factory/output/qwen35_medical_lora
template: qwen3
finetuning_type: lora
export_dir: /root/autodl-fs/LLaMA-Factory/output/qwen35_medical_merged
export_size: 2
export_device: cuda
export_legacy_format: false
```

合并后的模型可以直接用 [vLLM](https://zhida.zhihu.com/search?content_id=272182167&content_type=Article&match_order=1&q=vLLM&zhida_source=entity) 或者 SGLang 部署成 API 服务：

```plain
# vLLM 部署 
pip install vllm 
vllm serve ./models/Qwen35-Medical --port 8000 
# 或者 LLaMA-Factory 内置的 API 服务 
API_PORT=8000 llamafactory-cli api examples/inference/qwen35_medical.yaml
```

## 踩坑记录

分享几个我遇到过的问题：

**1. 显存不够用**

把 per_device_train_batch_size 调小，或者启用 gradient_checkpointing。实在不行就上 4-bit 量化。

**2. Loss 不下降**

检查数据格式是否正确，尤其是 dataset_info.json 里的字段映射。另外学习率不要设太大，1e-4 到 5e-5 之间比较稳。

**3. 微调后模型变傻了**

可能是数据质量问题，或者训练轮数太多导致过拟合。适当减少 epoch 数，或者在数据里混入一些通用对话保持泛化能力。

*医疗领域的 AI 应用一定要注意：模型输出仅供参考，不能替代专业医生的诊断。在产品设计时要做好免责声明和人工审核机制。*

整个流程走下来，从环境搭建到模型部署，熟练的话半天就能搞定。LLaMA-Factory 确实把微调的门槛降低了很多，配合 Qwen3.5 这样的高质量基座模型，普通开发者也能做出效果不错的垂直领域 AI 助手。

当然，真要做成产品级的医疗 AI，还需要在数据质量、安全合规、持续迭代等方面下功夫。但至少，迈出第一步没那么难。

来自: [ Qwen3.5-4B 微调实战：LLaMA-Factory 打造医疗AI助手 - 知乎](https://zhuanlan.zhihu.com/p/2021533045312733330)

不想一步步来也可使用本项目文件操作：

```
qwen-medical-finetune/
├── README.md                        # 微调详细操作
├── train_config/
│   └── qwen35_medical_lora.yaml     # 训练配置文件
├── data/
│   ├── dataset_info.json            # 数据集注册文件
└── └── convert_medical.py           # 一键进行数据转换脚本
```


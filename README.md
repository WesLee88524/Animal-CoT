# Animal-CoT
### **Pose-to-Context Chain-of-Thought Reasoning with Inverse Verification for Animal Behavior Recognition**

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>
#### Kou Yi, Yuhao Li, Guo Liu, Bohan Zhang, Yingqiu Huo

#### **Northwestern Polytechnical University，Northwest A&F University**

Code, datasets and pretrained models are coming!

- `2026/07/14`: Our code and datasets are released! Feel free to contact us if u have any problem!
- `2025/04/26`:  Our Paper has been accepted as an Oral paper in ICIC 2026!

<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Abstract. Animal behavior recognition is crucial for animal science, smart agriculture, and wildlife monitoring. However, existing keypoint- and detection-based methods rely on implicit feature learning and lack structured reasoning, while recent vision–language models remain suboptimal for animal behavior understanding due to limited domain knowledge and high intra-class variability. To
address these challenges, we present the first systematic exploration of multi-step Chain-of-Thought reasoning for animal behavior recognition, proposing Animal-CoT, a framework that decomposes behavior understanding into interpretable steps, including pose perception, contextual reasoning, behavior hypothesis generation, and inverse verification. This iterative reasoning process reduces semantic ambiguity and enhances robustness in challenging visual conditions. We further extend existing datasets with explicit multi-step reasoning annotations, resulting, to the best of our knowledge, in the first animal behavior recognition datasets that enable structured multimodal training. Extensive experiments on multiple benchmarks show that Animal-CoT consistently outperforms traditional
methods and state-of-the-art multimodal VLMs, achieving up to 49.7% F1-score improvement on CBVD-5 test set, which demonstrates the effectiveness of structured Chain-of-Thought reasoning for animal behavior recognition. 
</details>

# Animal-CoT

## Intro

- **Animal-CoT** introduces the **first Chain-of-Thought (CoT) reasoning datasets for Animal Behavior Recognition (ABR)** by extending existing datasets with structured step-by-step language annotations. Each video is annotated with pose, context, interaction, reasoning process, and final behavior labels, enabling interpretable multimodal learning.

- We propose **Animal-CoT**, a multimodal reasoning framework that fine-tunes **Qwen2.5-VL-7B-Instruct** on our CoT-annotated datasets. During training, the model learns to recognize animal behaviors through explicit reasoning rather than direct classification, while no additional supervision is required during inference.

<div align=center>
    <img src="source/framework.png" width="90%"/>
</div>

- We construct two reasoning datasets:

| Dataset | Species | Videos | Frames | QA Pairs | Reasoning | Modalities |
|:-------:|:-------:|:------:|:------:|:--------:|:---------:|:----------:|
| CBVD-L | 1 | 5,730 | 343,800 | 5,730 | ✓ | Video + Text |
| KABR-L | 3 | 2,490 | 498,000 | 2,490 | ✓ | Video + Text |
| **Total** | **4** | **8,220** | **841,800** | **8,220** | ✓ | Video + Text |

- Compared with existing animal behavior datasets, **CBVD-L** and **KABR-L** are the **first datasets equipped with multi-step Chain-of-Thought reasoning annotations**, bridging the gap between animal behavior recognition and multimodal reasoning.

<div align=center>
    <img src="source/dataset_pipeline.png" width="90%"/>
</div>

---

## Animal-CoT Framework

Our annotation pipeline consists of three stages:

1. **Automatic CoT Annotation**
   - Videos are first segmented into single-action clips.
   - A frozen **Qwen2.5-VL-32B-Instruct** generates structured five-step Chain-of-Thought descriptions, including:
     - Pose Attention
     - Contextual Attention
     - Hypothesis Generation
     - Inverse Verification
     - Iterative Refinement

2. **Manual Verification**
   - Five experts manually inspect all generated reasoning.
   - Incorrect or hallucinated annotations are removed or corrected to ensure logical consistency and factual accuracy.

3. **Model Fine-tuning**
   - We fine-tune **Qwen2.5-VL-7B-Instruct** using LoRA.
   - GroundingDINO and ByteTrack are adopted to provide tracking-based video representations before reasoning.

<div align=center>
    <img src="source/training_pipeline.png" width="90%"/>
</div>

---

## Performance on Benchmarks

### CBVD-5 Test Set

| Method | F1-score ↑ | mAP ↑ | rAP ↑ | Recall ↑ |
|:------|:---------:|:----:|:----:|:---------:|
| Asbar |0.142|0.260|0.086|0.243|
| ABIF |0.118|0.206|0.166|0.200|
| CSP |0.400|0.506|0.280|0.395|
| InternVL3.5-8B |0.352|0.365|0.314|0.495|
| Kimi-VL-A3B-Thinking |0.348|0.346|0.257|0.422|
| Qwen2.5-VL-7B-Instruct |0.339|0.378|0.254|0.399|
| Gemma-3-12B-IT |0.358|0.401|0.250|0.335|
| **Animal-CoT (Ours)** | **0.536** | **0.516** | **0.368** | **0.520** |

---

### KABR Test Set

| Method | F1-score ↑ | mAP ↑ | Recall ↑ |
|:------|:---------:|:----:|:---------:|
| ABIF |0.108|0.140|0.140|
| Animal-CLIP |0.381|0.353|0.387|
| CSP |0.401|0.355|0.415|
| InternVL3.5-8B |0.204|0.246|0.350|
| Qwen2.5-VL-7B-Instruct |0.258|0.270|0.389|
| Kimi-VL-A3B-Thinking |0.231|0.265|0.316|
| Gemma-3-12B-IT |0.292|0.257|0.401|
| **Animal-CoT (Ours)** | **0.449** | **0.385** | **0.454** |

- On **CBVD-5**, Animal-CoT improves the best traditional method by **34.0%** in F1-score and **31.4%** in rumination AP.

- On **KABR**, Animal-CoT achieves **12.0%** higher F1-score and **8.5%** higher mAP than the strongest traditional baseline, demonstrating superior cross-species reasoning ability.

<div align=center>
    <img src="source/performance.png" width="60%"/>
</div>

---

## Visualization

Qualitative comparisons on CBVD-5 and KABR.

Compared with conventional VLMs, Animal-CoT performs explicit pose-context reasoning before predicting behaviors, producing more interpretable and reliable decisions while effectively reducing hallucinations.

<div align=center>
    <img src="source/visualization.png" width="90%"/>
</div>

---

## Model Checkpoints

| Dataset | Model |
|:--------|:------|
| CBVD-L | Coming Soon |
| KABR-L | Coming Soon |

---




# 1. Animal-CoT Directory

This directory contains the code and resources for our paper:

**"Pose-to-Context Chain-of-Thought Reasoning with Inverse Verification for Animal Behavior Recognition"**

> **Note:** The paper is not publicly available yet.

GitHub Repository:
**[https://github.com/WesLee88524/Animal-CoT](https://github.com/WesLee88524/Animal-CoT)**

The `Animal-CoT` directory contains four subdirectories:

```
Animal-CoT
├── dataset
├── evaluation
├── model_weight
└── ablation
```

## dataset

This directory contains the datasets and related scripts for **CBVD-5** and **KABR**.

### CBVD-5

The original CBVD-5 dataset can be downloaded from:

[https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset)

> **Note:** The original dataset is **not included** in this repository. Please download it yourself if needed.

#### cot

Contains the Chain-of-Thought (CoT) annotations for the training and test sets, together with the script used to generate them.

* `labels2COT.py`

  * Script for generating CoT descriptions.
  * The code contains detailed comments and usage instructions. (The same applies to most scripts in this repository. If anything is unclear, AI coding assistants can also be very helpful 😊.)

* `train_CoT.json`

  * CoT descriptions for the training set.

* `test_CoT.json`

  * CoT descriptions for the test set.

#### finetune

Contains inference results generated by the fine-tuned **Qwen2.5-VL-7B** model.

* `run.py`

  * Runs inference on the test set using the fine-tuned Qwen2.5-VL-7B model and outputs a JSON file containing:

    * video filename
    * behavior label
    * prompt
    * model response

* `CBVD_test.json`

  * Output generated by `run.py`.

* `CBVD_test.csv`

  * CSV version of `CBVD_test.json` for evaluation.
  * Contains three columns:

    * ID
    * Ground Truth
    * Answer

#### videos

Contains the processed tracking videos and their corresponding annotation files.

---

### KABR

The original KABR dataset can be downloaded from:

[https://huggingface.co/datasets/imageomics/KABR](https://huggingface.co/datasets/imageomics/KABR)

> **Note:** The original dataset is **not included** in this repository. Please download it separately.

#### annotation

Contains behavior labels and dataset annotations.

* `classes.json`

  * Behavior label definitions.

* `train.csv`, `val.csv`

  * Original annotation files provided by KABR.

* `train.json`, `val.json`

  * Processed annotations containing:

    * video filename
    * behavior label

#### cot

Contains CoT annotations and generation scripts.

* `labels2COT.py`

  * Script for generating CoT descriptions.

* `train_CoT.json`

  * Training-set CoT descriptions.

* `test_CoT.json`

  * Test-set CoT descriptions.

#### finetune

Contains inference results produced by the fine-tuned **Qwen2.5-VL-7B** model.

* `run.py`

  * Performs inference on the test set and outputs a JSON file containing:

    * video filename
    * ground-truth behavior label
    * animal species
    * prompt
    * model response

* `KABR_test.json`

  * Output generated by `run.py`.

* `KABR_test.csv`

  * CSV version used for evaluation.
  * Contains three columns:

    * ID
    * Ground Truth
    * Answer

#### videos

Contains the processed training and validation videos derived from the original dataset.

---

## evaluation

Contains the evaluation scripts and comparison tables.

The comparison tables include the evaluation metrics of all experiments reported in the paper.

---

## model_weight

Contains:

* Fine-tuned model checkpoints
* Model checkpoints used in the ablation studies

---

## ablation

Contains:

* JSON and CSV files for the five-step ablation experiments
* JSON outputs generated by the original (non-fine-tuned) Qwen2.5 model

---

# 2. LLaMA-Factory Directory

**LLaMA-Factory** is an open-source framework for efficient fine-tuning of large language models and vision-language models.

GitHub Repository:

[https://github.com/hiyouga/LLaMAFactory](https://github.com/hiyouga/LLaMAFactory)

The official repository already provides comprehensive documentation and tutorial videos. Therefore, this directory only includes the code and data specifically related to this project.

In this project, **LLaMA-Factory** is used to perform **LoRA fine-tuning** on **Qwen2.5-VL-7B**.

```
LLaMA-Factory
└── data
```

## data

Contains the JSON files used for LoRA fine-tuning.

### LoRA Fine-tuning Configuration

```bash
--stage sft \
--do_train True \
--model_name_or_path <path_to_pretrained_model> \
--preprocessing_num_workers 16 \
--finetuning_type lora \
--template qwen2_vl \
--flash_attn auto \
--dataset_dir data \
--dataset <your_dataset_name> \
--cutoff_len 4096 \
--learning_rate 5e-05 \
--num_train_epochs <num_epochs> \
--max_samples 100000 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--logging_steps 50 \
--save_steps 2000 \
--warmup_ratio 0.03 \
--packing False \
--enable_thinking True \
--report_to none \
--output_dir <output_directory> \
--bf16 True \
--plot_loss True \
--trust_remote_code True \
--ddp_timeout 180000000 \
--include_num_input_tokens_seen True \
--optim adamw_torch \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0 \
--lora_target all \
--freeze_vision_tower True \
--freeze_multi_modal_projector True \
--video_fps 1.0 \
--video_maxlen 32 \
--image_max_pixels 589824 \
--image_min_pixels 1024 \
--video_max_pixels 65536 \
--video_min_pixels 256 \
--val_size 0.01 \
--eval_strategy steps \
--eval_steps 1000 \
--per_device_eval_batch_size 1
```

## Citation

If you find this work useful, please consider citing:

```BibTeX
@inproceedings{Kou2026AnimalCOT,
  title={Pose-to-Context Chain-of-Thought Reasoning with Inverse Verification for Animal Behavior Recognition},
  author={Kou, Yi and Li, Yuhao and Liu, Guo and Zhang, Bohan and Huo,Yingqiu},
  booktitle={2026 International Conference on Intelligent Computing},
  pages={1-12},
  year={2026}
}

```


## License
This project is released under the Apache license. See [LICENSE](LICENSE.txt) for additional details.


> Replace `<path_to_pretrained_model>`, `<your_dataset_name>`, `<num_epochs>`, and `<output_directory>` with your own settings before training.

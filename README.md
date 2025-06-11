# Jetson-CV-Opt

**Jetson Computer Vision Optimization**

This repository contains test configuration files and data related to our work on optimizing computer vision workloads for NVIDIA Jetson platforms.

Our framework enables high-throughput, NPU-accelerated inference for ONNX-based models on Jetson AGX Orin by combining mixed-precision inference, updated calibration tables, and activation replacement techniques—achieving minimal accuracy loss.

<!-- graphics -->

## 📚 Table of Contents

- [Benchmark Results](#-benchmark-results) 
  * [Classification](#classification)
  * [Detection](#detection)
- [Installation](#-installation)   
  * [Requirements](#-requirements)
  * [Prerequisites](#%EF%B8%8F-prerequisites)
  * [Directory Structure](#-directory-structure)
  * [Installation & Build](#%EF%B8%8F-installation--build)
  * [Usage](#-usage)
- [Training Hyperparameters](#training-hyperparameters)
  * [Activation Fine-Tuning](#activation-fine-tuning)
  * [YOLO Quantization-Aware Training (QAT)](#yolo-quantization-aware-training-qat)

## 📊 Benchmark Results

The following results present the performance of various classification and detection models optimized using the proposed methodology on the NVIDIA Jetson AGX Orin.
Each model was evaluated under three configurations:

* **acc**: Accuracy-focused optimization
* **pareto**: Pareto-optimal trade-off between accuracy and throughput
* **fps**: Throughput-maximized configuration

For classification tasks, Top-1 accuracy on ImageNet-1K was measured.
For detection tasks, mAP was evaluated on both the COCO 2017 validation and test sets.
All input sizes match each model’s original training configuration.

### Classification

| Model Name               | Target | Acc.  | FPS  |
|--------------------------|--------|-------|------|
| convmixer_768_32         | acc    | 80.17 | 163  |
| convmixer_768_32         | pareto | 79.78 | 274  |
| convmixer_768_32         | fps    | 78.52 | 301  |
| convmixer_1536_20_silu   | acc    | 79.95 | 54   |
| convmixer_1536_20_silu   | pareto | 78.72 | 112  |
| convmixer_1536_20_silu   | fps    | 75.4  | 123  |
| efficientformerv2_l_silu | acc    | 82.84 | 520  |
| efficientformerv2_l_silu | pareto | 82.24 | 639  |
| efficientformerv2_l_silu | fps    | 74.57 | 822  |
| fastvit_ma36_silu        | acc    | 84.14 | 322  |
| fastvit_ma36_silu        | pareto | 83.15 | 346  |
| fastvit_ma36_silu        | fps    | 80.91 | 432  |
| mobileone_s1             | acc    | 75.59 | 1506 |
| mobileone_s1             | pareto | 75.58 | 1550 |
| mobileone_s1             | fps    | 74.87 | 1571 |
| res2net50d               | acc    | 80.23 | 866  |
| res2net50d               | pareto | 80.22 | 906  |
| res2net50d               | fps    | 80.15 | 922  |
| res2net101d              | acc    | 81.17 | 512  |
| res2net101d              | pareto | 81.13 | 528  |
| res2net101d              | fps    | 81.11 | 715  |

### Detection

All classification results are reported using a confidence threshold of 0.1 during evaluation.

| Model Name    | Target | mAP(test) | mAP(val.) | FPS |
|---------------|--------|-----------|-----------|-----|
| yolov9_t      | acc    | 35.4%     | 35.0%     | 508 |
| yolov9_t      | pareto | 35.2%     | 34.7%     | 694 |
| yolov9_t      | fps    | 35.0%     | 34.6%     | 732 |
| yolov9_c      | acc    | 49.4%     | 49.2%     | 161 |
| yolov9_c      | pareto | 49.2%     | 48.9%     | 181 |
| yolov9_c      | fps    | 48.8%     | 48.4%     | 222 |
| yolov9_c_relu | acc    | 48.6%     | 48.1%     | 163 |
| yolov9_c_relu | pareto | 48.4%     | 48.1%     | 247 |
| yolov9_c_relu | fps    | 48.2%     | 48.1%     | 309 |
| yolov9_e      | acc    | 52.0%     | 51.7%     | 49  |
| yolov9_e      | pareto | 51.9%     | 51.5%     | 52  |
| yolov9_e      | fps    | 51.4%     | 51.4%     | 61  |

---
<br/><br/><br/>

## 🔧 Installation

### ✅ Requirements

* **CMake** ≥ 3.2
* **Make**

### 📦 Python Requirements

The following Python packages are required to run the benchmark and evaluation scripts:

* `polygraphy==0.49.9`
* `onnx==1.16.1`
* `onnx_graphsurgeon==0.5.2`
* `pycocotools==2.0.8`
* `faster_coco_eval==1.6.5`

All required packages are listed in the `requirements.txt` file.  
Install all dependencies with:

```bash
pip install -r requirements.txt
```

### ⚙️ Prerequisites

Before building and running the application on **NVIDIA Jetson AGX Orin**, make sure to:

1. **Enable MAXN Power Mode**
   Set the device to maximum performance mode to ensure consistent and reproducible results:

   ```bash
   sudo jetson_clocks
   ```

   This locks the CPU, GPU, and memory frequencies and removes dynamic power limits.

2. **Grant Access to DLA Utilization Logs**
   Run the following script to enable DLA utilization monitoring for JEDI:

   ```bash
   sudo ./dla_perm.sh
   ```

   This is required for monitoring DLA activity during inference.

3. **Jetpack and TensorRT Versions**
   All experiments were conducted on **JetPack 6.1** and **TensorRT 10.7**.

4. **Power and Energy Measurement**
   Power usage for all components was monitored via tegrastats (automatically triggered, no user action needed).

   Energy usage was computed by multiplying the average power (W) with total execution time.

5. **Model Format**
   Benchmark networks must be exported to the ONNX format before execution. Pre-converted models can be downloaded from the [Releases](https://github.com/cap-lab/Jetson-CV-Opt/releases) tab. Models from Hugging Face timm are supported, but ensure that they use static input shapes (i.e., batch size = 1).

---

### 📂 Directory Structure

```
.
├── calibration_tables   # Calibration tables for INT8 quantization, incl. DLA support
├── configs              # Benchmark configuration files
├── data
│   ├── coco2017         # COCO 2017 validation & test data
│   └── imagenet12       # ImageNet-1K validation data
├── engines              # Generated TensorRT engine files (.rt)
├── onnx                 # ONNX models
└── run
```
**To prepare the datasets and models, follow these steps:**

1. **First**, download or symlink the [coco2017](https://cocodataset.org/#download) and [imagenet12](https://www.image-net.org/download.php) datasets into the `data` folder.
2. **Next**, download the required ONNX models from the [Releases](https://github.com/cap-lab/Jetson-CV-Opt/releases) page and place them in the `onnx` directory.

### 🏗️ Installation & Build

1. Clone and Initialize Submodules

```bash
git clone https://github.com/cap-lab/Jetson-CV-Opt.git
cd Jetson-CV-Opt
git submodule update --init --recursive
```

2. Build the Project

```bash
mkdir -p run/build
cd run/build
cmake ..
make -j
```

### 🚀 Usage

1. Prepare Data and Models

Download or symlink [imagenet12](https://www.image-net.org/download.php) and [coco2017](https://cocodataset.org/#download) datasets to the data directory.

Download ONNX models from the [Releases](https://github.com/cap-lab/Jetson-CV-Opt/releases) page and place them in the onnx directory.

2. Run Inference Examples

- Classification
```bash
./run/build/bin/proc -c configs/mobileone_s1_fps.cfg
```

- Detection
```bash
./run/build/bin/proc -c configs/yolov9_t_fps.cfg -r coco_results.json
```

3. Evaluate Detection Results

```bash
python evaluate.py coco_results.json [instance_val.json]
```

---
<br/><br/><br/>


## Training Hyperparameters

This section summarizes the key training hyperparameters and fine-tuning procedures used for quantization and activation replacement experiments.
Classification models were trained on four RTX 4090 GPUs, two RTX 3090 GPUs, or a single A6000. Detection models (YOLOv9) were trained on a single NVIDIA A6000 GPU.

### Activation Fine-Tuning

#### Results

All results are evaluated on the ImageNet validation set.
|      Model Name     | GeLU  | ReLU  | SiLU  |
|:-------------------:|-------|-------|-------|
| ConvMixer-1536/20   | 81.37 | 79.21 | 79.95 |
| EfficientFormerV2-L | 83.63 | 81.92 | 82.85 |
| FastVit-MA36        | 84.61 | 83.86 | 84.18 |

#### Hyperparameter Settings

NVIDIA Orin DLA does not support certain activation functions, such as GeLU. To maximize DLA compatibility, all GeLU activations were replaced with SiLU, followed by brief fine-tuning. For comparison, results with ReLU activations are also reported.

- Seed: 42 (for all runs, for reproducibility)

| Parameter (ReLU)    | ConvMixer-1536/20 | EfficientFormerV2-L | FastVit-MA36   |
| ------------------- | ----------------- | ------------------- | -------------- |
| input-size          | 3,224,224         | 3,224,224           | 3,256,256      |
| sched               | cosine            | cosine              | cosine         |
| epochs              | 30                | 300                 | 100            |
| decay-epochs        | -                 | 90                  | 90             |
| decay-rate          | 0.1               | 0.1                 | 0.1            |
| batch-size          | 64                | 128                 | 128            |
| amp                 | true (float16)    | true (float16)      | true (float16) |
| lr (initial)        | 3e-4              | 1e-5                | 3e-6           |
| warmup-epochs       | 0                 | 5                   | 5              |
| warmup-lr           | -                 | 1e-5                | 1e-6           |
| opt                 | adamW             | adamW               | adamW          |
| weight-decay        | 2e-5              | 0.025               | 0.05           |
| drop-path           | -                 | -                   | 0.2            |
| cooldown-epochs     | -                 | -                   | 10             |
| workers             | 32                | 32                  | 32             |
| GPU (Fine-Tuning)   | RTX 3090 x 2      | RTX 3090 x 2        | RTX 4090 x 4   |
| Fine-Tuning Time    | 400H              | 73H                 | 29H            |
<!-- 399H 22m / 72H 50m / 28H 37m -->

| Parameter (SiLU) | ConvMixer-1536/20 | EfficientFormerV2-L | FastVit-MA36   |
| ---------------- | ----------------- | ------------------- | -------------- |
| input-size       | 3,224,224         | 3,224,224           | 3,256,256      |
| sched            | cosine            | cosine              | cosine         |
| epochs           | 30                | 30                  | 100            |
| decay-epochs     | -                 | 90                  | 90             |
| decay-rate       | 0.1               | 0.1                 | 0.1            |
| batch-size       | 64                | 128                 | 64             |
| amp              | true (float16)    | true (float16)      | true (float16) |
| lr (initial)     | 1e-5              | 1e-5                | 3e-6           |
| warmup-epochs    | 0                 | 5                   | 5              |
| warmup-lr        | -                 | 1e-5                | 1e-6           |
| opt              | adamW             | adamW               | adamW          |
| weight-decay     | 0.025             | 0.025               | 0.05           |
| drop-path        | -                 | -                   | 0.2            |
| cooldown-epochs  | -                 | -                   | 10             |
| workers          | 8                 | 8                   | 32             |
| GPU (Fine-Tuning)| A6000             | A6000               | RTX 4090 x 4   |
| Fine-Tuning Time |                   |                     | 31h            |

---

### YOLO Quantization-Aware Training (QAT)

#### Results

All results are evaluated on the COCO 2017 validation set.
|    Model Name   | FP32 | INT8 | QAT  |
|:---------------:|------|------|------|
| Yolov9-T        | 35.1 | 29.4 | 34.7 |
| Yolov9-C        | 49.2 | 42.4 | 48.8 |
| Yolov9-C (ReLU) | 48.1 | 46.8 | 48.1 |
| Yolov9-E        | 51.7 | 43.5 | 51.5 |

#### Hyperparameter Settings

All training parameters follow the [YOLOv9 QAT repository](https://github.com/levipereira/yolov9-qat).
- Image size: 640 × 640

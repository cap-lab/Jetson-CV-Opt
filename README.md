# Jetson-CV-Opt

**Jetson Computer Vision Optimization**

This repository contains test configuration files and data related to our work on optimizing computer vision workloads for NVIDIA Jetson platforms.

## 📚 Table of Contents

- [Benchmark Results](#benchmark-results) 
- [Installation](#installation)   

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

| Model Name    | Target | mAP(val.) | mAP(test) | FPS |
|---------------|--------|-----------|-----------|-----|
| yolov9_t      | acc    | 35.4%     | 35.0%     | 508 |
| yolov9_t      | pareto | 35.2%     | 34.7%     | 694 |
| yolov9_t      | fps    | 35.0%     | 34.6%     | 732 |
| yolov9_c      | acc    | 49.0%     | 49.2%     | 161 |
| yolov9_c      | pareto | 49.2%     | 48.9%     | 181 |
| yolov9_c      | fps    | 48.8%     | 48.6%     | 222 |
| yolov9_c_relu | acc    | 48.6%     | 48.1%     | 163 |
| yolov9_c_relu | pareto | 48.4%     | 48.1%     | 247 |
| yolov9_c_relu | fps    | 48.2%     | 48.1%     | 309 |
| yolov9_e      | acc    | 52.0%     | 51.7%     | 49  |
| yolov9_e      | pareto | 51.9%     | 51.5%     | 52  |
| yolov9_e      | fps    | 51.4%     | 51.4%     | 61  |

## 🔧 Installation

### ✅ Requirements

* **CMake** ≥ 3.2
* **Make**

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
   Benchmark networks must be exported to the ONNX format before execution. Pre-converted models can be downloaded from the Releases tab. Models from Hugging Face timm are supported, but ensure that they use static input shapes (i.e., batch size = 1).

---

### 📥 1. Clone and Initialize Submodules

```bash
git submodule update --init --recursive
```

### 🏗️ 2. Build the Project

```bash
mkdir -p run/build
cd run/build
cmake ..
make -j
```

### 🚀 3. Run an Example

```bash
./run/build/bin/proc -c configs/yolov9_t_fps.cfg
```

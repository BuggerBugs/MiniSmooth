## MiniSmooth

> Combining reconstruction-based Post-training Quantization with activation smoothing for improved low-bit accuracy.

### Overview

Reconstruction-based Post-training Quantization (PTQ) methods have shown strong performance in low-bit settings. However, these techniques do not address accuracy degradation caused by **activation outliers**.

MiniSmooth integrates smoothing techniques with state-of-the-art reconstruction-based PTQ methods (Adaround, BRECQ, QDrop) to address this gap. The implementation is evaluated on the MQBench profiling benchmark.

### How It Works

The smoothing formulation follows [SmoothQuant](https://arxiv.org/abs/2211.10438), and is adapted 
for use in CNNs (see also: [Convolution Smooth](https://ieeexplore.ieee.org/abstract/document/10955493/)).
Unlike prior work, MiniSmooth combines smoothing with reconstruction-based PTQ methods.

The smoothing is implemented from scratch (no public reference code was found for CNN 
smoothing) and incorporated into MQBench; the reconstruction methods (Adaround, BRECQ, 
QDrop) are provided by MQBench.

Per-channel smoothing is applied as follows, where s is the scaling factor:

```
activations = activations × (1/s)
weights     = weights     ×  s
```

This shifts the quantization difficulty from activation outliers to the weights, where quantization is generally easier to manage.

#### Smoothing Factor

The per-channel scaling factor `s` is computed as:

```
s = act_max^α / weight_max^(1−α)
```

where **α** (migration intensity) controls how much difficulty is shifted to the weights — a higher α shifts more difficulty to the weight side.

### Results

Evaluated on **ResNet-50** at **W2A4** (2-bit weights, 4-bit activations) on academic backend with **α = 0.7**:

| Method   | Accuracy Improvement |
|----------|----------------------|
| Adaround | **+1.3%**            |
| BRECQ    | **+0.9%**            |
| QDrop    | **+0.7%**            |

MiniSmooth yields notable accuracy improvements at W2A4 compared to baselines for the above methods (without smoothing), and has **negligible impact** at higher bit-width configurations.

> **Note:** The current implementation incurs heavy deployment overhead and increased GPU utilization.

### Implementation

The main MiniSmooth implementation is located at:

```
imagenet_example/PTQ/ptq/smoothquant.py
```


---

## License and Attribution

This project is built on top of [MQBench](https://github.com/ModelTC/MQBench) and inherits its Apache 2.0 License.

- **Original work:** Copyright (c) 2021 ModelTC/MQBench contributors

See [LICENSE](LICENSE) for the full Apache 2.0 license text.

For MQBench documentation, installation instructions, and citation information, visit:
- [MQBench GitHub Repository](https://github.com/ModelTC/MQBench)
- [MQBench Documentation](https://mqbench.readthedocs.io/en/latest/)

---

## Replicating MiniSmooth Accuracy Experiment Results

### 1. Setup

- Clone this repository to Aspire2A. To install dependencies, see the last section below, or refer to MQBench Github Repository.
- Download ImageNet-1k 2012 from Kaggle for the validation and train sets:
  - **Train subset (for calibration):** https://www.kaggle.com/datasets/tusonggao/imagenet-train-subset-100k
  - Validation Set should be easily obtained on kaggle.  
  - Place images into `ImageNet-ILSVRC2012/train/` and `ImageNet-ILSVRC2012/val/` folders respectively.
- Download the ResNet-18 and ResNet-50 pth.tar from MQBench:
  https://github.com/ModelTC/MQBench/releases/tag/pre-trained
  Place them into a dedicated directory.

### 2. Identify the Run to Replicate

- Open `MiniSmoothDataResults/FYP_RESULTS_R18.xlsx` or `FYP_RESULTS_SUMMARY_R50.xlsx`.
- Navigate to the **ALL_RAW_DATA** tab and find the run under the **Run Name** column. Names of the runs follow this format:

  ```
  reconstructionTechnique_resnetModel_WeightQuantBits_ActQuantBits_NoSmoothQuant|SmoothQuantAlpha=0.5|SmoothQuantAlpha=0.7
  ```

### 3. Configure

- For the selected run, find the config file under `imagenet_example/PTQ/configs/<reconstructionTechnique>/`.
- Edit the following fields:
  - `model:path` — path to resnet pth.tar
  - `data:path` — path to ImageNet-ILSVRC2012 directory (that contains train and val subdirs)
  - `quantize:smoothquant:alpha` and `quantize:smoothquant:enabled`
  - `seed - values we used were`:
    - **ResNet-18:** `1005`, `89`, `42`, `2000`, `2001`
    - **ResNet-50:** `1005`, `89`, `42`

### 4. Run (in /imagenet_example/PTQ/ptq)

```bash
python3 ptq.py --config ../configs/adaround/FYP_CONFIG_FILE.yaml
```

See `MiniSmoothAspire2a/jobscript.sh` for a reference job script. After execution, inspect the output — accuracy should match the values in the Excel sheet. Repeat for multiple seeds and configs.

> **Note:** `MiniSmoothAspire2a/FYPRuns.sh` automates mass job submission in tandem with `imagenet_example/PTQ/ptq/runExperiments.py`, but requires many directory path changes before reuse. Feel free to use it as a reference for mass job submission.

---

## Replicating TensorRT Latency Experiment Results

### 1. Setup

Repeat steps 1 from the Accuracy section above, but on the **Jetson Nano**.

### 2. Install TensorRT

- Download for your jetson nano on CUDA site
- If you face errors while executing trt, try updating NVIDIA drivers.


### 3. Generate ONNX and Clip Range Files

Run (in /imagenet_example/PTQ/ptq) PTQ on the following configs to produce corresponding ONNX and clip range files (1 pair should be generated per ptq.py run) :

```bash
python3 ptq.py --config ../configs/adaround/FYP_r50_8_8_trt_NOSQ.yaml
python3 ptq.py --config ../configs/adaround/FYP_r50_8_8_trt_SQ05.yaml
```

> You can run this on Aspire2A and transfer the output files to the Jetson Nano.

### 4. Profile

Open three terminals, then run them in the following order:

| Terminal | Command |
|----------|---------|
| 1 | `python MiniSmoothJetsonNano/tegrastats_monitor.py` |
| 2 | `htop` *(track GPU shared memory usage manually)* |
| 3 | Run inference (see below) |

```bash
python MiniSmoothJetsonNano/onnx2trt.py \
  --onnx-path <onnx_filename>.onnx \
  --trt <output_trtengine_name>.trt \
  --data <validation_set_path> \
  --evaluate \
  --explicit \
  --clip-range-file <clip_range_file_path>
```

tegrastats_monitor.py should auto-terminate and print benchmarking results once onnx2trt.py is complete.

---

## MiniSmooth Validation Experiments

*These are secondary - mostly supplementary*

a. Correctness of MiniSmooth: Verify that scaling is applied as intended (empirical 
scaled activation and weight values match the theoretical ones).  
b. Effect of MiniSmooth: Observe whether MiniSmooth actually reduces activation 
variance in practice.  
c. MiniSmooth deployment correctness: Verify that implementation persists 
correctly after the deployment pipeline is employed. 

- **Val Experiment b** — uses `verify_smoothquant.py`, `verify_ptq.py`, `FYPDataCollectRuns.py`, `RunDataCollect.py`
- **Val Experiment a** — after inserting scaling hooks, simply re-pass the calibration set through the model and check that resulting activations actually match the theoretically calculated ones
- Results and ipynbs for data processing, val experiments a and b: `MiniSmoothDataResults/val/`
- Val Experiment c is trivial and results should be easily obtained (intentionally remove hooks prior to deployment to ensure it breaks model accuracy)

---

## Environment Setup Issues

You might face dependency version issues on MQBench. I used the following for Aspire2A:

```bash
conda create -n mqbench python=3.10 -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0"

# Then from the MQBench directory:
pip install -r requirements.txt
pip install -e .
```
---



# Notebook Descriptions

## jetconstituteCode.ipynb

This notebook contains the main jet constituent classification experiments using several different neural network architectures.

### Models Evaluated

* hls4ml CNN
* ConvSNN
* ConvSNN with 2 convolutional layers (feature extraction version)
* Simple MLP-Mixer

### Purpose

The notebook compares conventional CNN-based approaches with spiking neural network (SNN) architectures and MLP-Mixer models on the jet constituent classification task.

### Results Included

* Training and validation performance
* Classification accuracy
* Model comparisons between CNN, ConvSNN, ConvSNN (2-conv feature extractor), and MLP-Mixer architectures

---

## MLPMixer_RELUcode.ipynb

This notebook evaluates a conventional MLP-Mixer architecture using ReLU activations.

### Hidden Dimensions Evaluated

* Hidden Dimension = 16
* Hidden Dimension = 64

### Purpose

This notebook serves as the non-spiking baseline for comparison against the spiking MLP-Mixer implementations.

### Important Note

Since ReLU networks do not use temporal spike accumulation, changing the number of timesteps does not affect the model output.

As a result, performance versus timestep appears as a flat line.

In other words:

```text
Timestep changes only affect spike counting in SNN models.
For ReLU models, timestep has no effect on inference results.
```

---

## MLPMixerCode.ipynb

This notebook contains the Spiking Neural Network (SNN) version of the MLP-Mixer architecture using Leaky Integrate-and-Fire (LIF) neurons.

### Hidden Dimensions Evaluated

* Hidden Dimension = 16
* Hidden Dimension = 64
* Hidden Dimension = 128
* Hidden Dimension = 256

### Experiments Included

#### 1. Hidden Dimension Comparison

Performance comparison across different hidden dimensions.

#### 2. Timestep Analysis

Investigation of the effect of timestep count on model performance.

The following metrics are evaluated as a function of timestep:

* Accuracy
* ROC Curve
* AUC (Area Under Curve)

#### 3. ROC and AUC Evaluation

ROC curves and AUC scores are generated for different timestep settings to identify the optimal temporal integration window.

### Key Finding

Among the timestep values evaluated:

```text
T = 20
```

provided the best overall performance and was selected as the final configuration for subsequent experiments.

### Purpose

This notebook investigates how temporal spike accumulation influences classification performance in SNN-based MLP-Mixer models and determines the optimal timestep configuration.

---

# Notes on Earlier FC Replacement Attempts

The notebooks:

```text
hls4ml-snn-example-base-Copy1 (1).ipynb
hls4ml-snn-example.ipynb
```

were earlier development attempts to replace the fully connected (FC) layer from the original hls4snn example with our Conv1D-based architecture.

These notebooks successfully explored model conversion and architecture modifications, but ultimately failed during RTL co-simulation (RTL cosim). Therefore, they are retained only as intermediate development/debugging attempts and are not part of the final working resource-estimation flow.

---

# hls4snn FPGA Resource Estimation Flow

This guide describes how to set up **hls4snn**, run SNN-to-HLS conversion, synthesize the generated design using **Vitis HLS**, and obtain FPGA resource utilization estimates (FF, LUT, DSP, BRAM, URAM).

## 1. Clone the repository

```bash
git clone https://github.com/bmdillon/hls4snn.git
cd hls4snn
```

## 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Python dependencies

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
numpy \
scipy \
pandas \
matplotlib \
pyyaml \
jupyter \
notebook \
ipykernel \
torch \
torchvision \
torchaudio \
snntorch

python -m pip install -e .
```

## 4. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install \
    --user \
    --name=hls4snn \
    --display-name "Python (hls4snn)"
```

## 5. Source the Xilinx tools

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /tools/Xilinx/Vivado/2023.2/settings64.sh
```

Verify:

```bash
which vitis_hls
which vivado
which vitis-run
```

## 6. Launch Jupyter

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open:

```text
examples/snn_example/hls4ml-snn-example.ipynb
```

Select kernel:

```text
Python (hls4snn)
```

## 7. Configure Xilinx paths inside the notebook

```python
import os

os.environ["PATH"] = (
    "/tools/Xilinx/Vitis_HLS/2023.2/bin:"
    "/tools/Xilinx/Vitis/2023.2/bin:"
    "/tools/Xilinx/Vivado/2023.2/bin:"
    + os.environ["PATH"]
)

os.environ["XILINX_VITIS"] = "/tools/Xilinx/Vitis/2023.2"
os.environ["XILINX_HLS"] = "/tools/Xilinx/Vitis_HLS/2023.2"
os.environ["XILINX_VIVADO"] = "/tools/Xilinx/Vivado/2023.2"
```

Verify:

```python
!which vitis_hls
!which vivado
!which vitis-run
```

## 8. Convert a trained PyTorch SNN model

```python
state = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state)
model.eval()
```

```python
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    output_dir="my_snn_project",
    project_name="my_snn",
    backend="Vitis",
    io_type="io_stream",
    hls_config=hls_config,
    part="xczu7ev-ffvc1156-2-e",
    clock_period=5,
)
```

```python
hls_model.write()
```

## 9. Run synthesis

```python
report = hls_model.build(
    reset=True,
    csim=False,
    synth=True,
    cosim=False,
    validation=False,
    export=False,
    vsynth=True,
    log_to_stdout=True,
)
```

---

# FPGA Resource Estimation

The primary objective is obtaining FPGA resource estimates:

* FF (Flip-Flops)
* LUT (Lookup Tables)
* DSP
* BRAM
* URAM
* Latency
* Initiation Interval (II)

## Monitor synthesis progress

```bash
top
```

or

```bash
ps -ef | grep vitis_hls
```

Typical output:

```text
vitis_hls    100% CPU
clang        100% CPU
```

Large SNN models may require tens of minutes or several hours to synthesize.

## Locate the synthesis report

```bash
find . -name csynth.rpt
```

Typical location:

```text
examples/my_snn_resource/my_snn/my_snn_prj/solution1/syn/report/csynth.rpt
```

Open:

```bash
less examples/my_snn_resource/my_snn/my_snn_prj/solution1/syn/report/csynth.rpt
```

## Extract FPGA resource usage

Locate:

```text
Performance & Resource Estimates
```

Example:

```text
+---------------------------------------------------------------------------+
| Modules & Loops | FF | LUT | DSP | BRAM | URAM |
+---------------------------------------------------------------------------+
|+ my_snn*        |575 |2264 |  0  |  0   |  0   |
+---------------------------------------------------------------------------+
```

The first row corresponds to the total synthesized design.

| Resource | Usage |
| -------- | ----- |
| FF       | 575   |
| LUT      | 2264  |
| DSP      | 0     |
| BRAM     | 0     |
| URAM     | 0     |

Latency and Initiation Interval (II) are reported in the same section.

## Resource Breakdown by Layer

Example:

```text
| lif1        | FF 82  | LUT 858 |
| conv1       | FF 68  | LUT 476 |
| fc_out_conv | FF 43  | LUT 170 |
```

This helps identify the most resource-intensive layers.

## Expected Synthesis Time

| Model Size                     | Typical Runtime        |
| ------------------------------ | ---------------------- |
| Small SNN example              | 1–5 min                |
| Medium Conv1D SNN              | 5–30 min               |
| Large Conv1D + LIF network     | 30–120+ min            |
| Long sequence (SEQ_LEN = 3000) | Several hours possible |

## Notes

* High CPU usage from `vitis_hls` and `clang` is expected.
* Vitis HLS uses CPU and RAM only; GPUs are not used.
* Resource estimation does not require RTL simulation (`csim=False`, `cosim=False`).
* Large sequence lengths can significantly increase synthesis time.
* If synthesis becomes impractical, start with shorter sequences (128, 256, or 512) and scale upward.
* The generated HLS project automatically includes the trained PyTorch weights loaded from the checkpoint; no manual weight export is required.

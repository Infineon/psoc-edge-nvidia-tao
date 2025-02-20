# PeopleNet Conversion Notebook

![NVIDIA TAO](https://img.shields.io/badge/NVIDIA-TAO_Toolkit-76B900?style=flat-square&logo=nvidia)
![Infineon](https://img.shields.io/badge/Infineon-Tooling-0058CC?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square)

This Jupyter notebook provides a streamlined workflow for acquiring NVIDIA's pre-trained PeopleNet models and converting them for deployment on Infineon hardware with Ethos-U acceleration.

## Overview

The notebook covers the essential steps to:

1. **Configure the environment** for TAO model conversion
2. **Install the NGC CLI** for accessing NVIDIA's model registry
3. **Download pre-trained PeopleNet models** in quantized ONNX format
4. **Convert models** using Infineon's tooling for Ethos-U acceleration

## Prerequisites

Before running this notebook, ensure you have:

- **NVIDIA GPU** with CUDA support for running the TAO container
- **Docker** installed and configured
- **Access to NGC** (NVIDIA GPU Cloud) - create an account if needed
- **Infineon tooling package** installed in your Python environment
- At least **16GB of disk space** for models and outputs

## Environment Setup

The notebook starts by configuring the environment variables needed for the workflow:

```python
# Setting up env variables
import os

%env NUM_GPUS=1
%env USER_EXPERIMENT_DIR=/workspace/tao-experiments/peoplenet_onnx
%env DATA_DOWNLOAD_DIR=/workspace/tao-experiments/data

# Set your local project directory
os.environ["LOCAL_PROJECT_DIR"] = "/teamspace/studios/this_studio/peoplenet"

os.environ["LOCAL_DATA_DIR"] = os.path.join(
    os.getenv("LOCAL_PROJECT_DIR", os.getcwd()),
    "data"
)
os.environ["LOCAL_EXPERIMENT_DIR"] = os.path.join(
    os.getenv("LOCAL_PROJECT_DIR", os.getcwd()),
    "peoplenet"
)

# Make the experiment directory 
! mkdir -p $LOCAL_EXPERIMENT_DIR
```

> **Note**: Be sure to update `LOCAL_PROJECT_DIR` to your specific project path.

## NGC CLI Installation

The NGC CLI is required to download models from NVIDIA's model registry:

```python
# Installing NGC CLI on the local machine
%env CLI=ngccli_cat_linux.zip
!mkdir -p $LOCAL_PROJECT_DIR/ngccli

# Remove any previously existing CLI installations
!rm -rf $LOCAL_PROJECT_DIR/ngccli/*
!wget "https://ngc.nvidia.com/downloads/$CLI" -P $LOCAL_PROJECT_DIR/ngccli
!unzip -u "$LOCAL_PROJECT_DIR/ngccli/$CLI" -d $LOCAL_PROJECT_DIR/ngccli/
!rm $LOCAL_PROJECT_DIR/ngccli/*.zip 
os.environ["PATH"]="{}/ngccli/ngc-cli:{}".format(os.getenv("LOCAL_PROJECT_DIR", ""), os.getenv("PATH", ""))
```

## Model Download

The notebook downloads the pre-trained, quantized PeopleNet model:

```python
!mkdir -p $LOCAL_EXPERIMENT_DIR/quantized_onnx_model

!ngc registry model download-version "nvidia/tao/peoplenet:pruned_quantized_decrypted_v2.3.4" \
    --dest $LOCAL_EXPERIMENT_DIR/quantized_onnx_model
```

This specific model version (v2.3.4) includes:
- INT8 quantization for efficient deployment
- Pruning for reduced model size
- Pre-trained weights for people, bag, and face detection

## Infineon Tooling Conversion

The heart of the notebook is the model conversion using Infineon's tooling:

```python
from ifx_tooling import run_ifx_tooling, ModelConversionError
import os
from pathlib import Path

qat_onnx_model_path = os.path.join(os.environ['LOCAL_EXPERIMENT_DIR'], 
                                  "quantized_onnx_model/peoplenet_vpruned_quantized_decrypted_v2.3.4/resnet34_peoplenet_int8.onnx")
ifx_tooling_output_path = os.path.join(os.environ['LOCAL_EXPERIMENT_DIR'], "ifx_tooling")

config = {
    'vela_accelerator': 'ethos-u55-128',
    'vela_system_config': 'PSE84_M55_U55_400MHz',
    'vela_memory_mode': 'Sram_Only',
    'compress_to_fp16': False,
    'vela_ini_file_path': os.path.join(os.environ['LOCAL_PROJECT_DIR'], "vela.ini")
}

try:
    output_paths = run_ifx_tooling(
        onnx_model_path=qat_onnx_model_path,
        input_shape=[1, 3, 544, 960],
        output_dir=ifx_tooling_output_path,
        config=config
    )
    print("Generated artifacts:", output_paths)
except ModelConversionError as e:
    print(f"Conversion failed: {e}")
```

### Configuration Options

The configuration parameters control how the model is optimized for Ethos-U:

| Parameter | Description | Value |
|-----------|-------------|-------|
| `vela_accelerator` | Target Ethos-U accelerator | ethos-u55-128 |
| `vela_system_config` | Hardware configuration | PSE84_M55_U55_400MHz |
| `vela_memory_mode` | Memory usage strategy | Sram_Only |
| `compress_to_fp16` | FP16 compression flag | False |
| `vela_ini_file_path` | Path to Vela config file | vela.ini |

## Next Steps

After completing the model conversion, you can:

1. Move the converted model to the `models/converted` directory
2. Use the Python implementation in `peoplenet_tflite_demo` to run detections
3. For detailed instructions on using the detection system, refer to the README in the `peoplenet_tflite_demo` directory
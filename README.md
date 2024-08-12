# qwen2-export-onnx

## Overview

This is a versatile tool designed to convert and run language models in the ONNX format. This project facilitates the exportation of models to ONNX, as well as the subsequent loading and inference using these models. Additionally, it helps to efficiently convert Large Language Models (LLMs) like QWEN for deployment on edge computing platforms.

## System Requirements

- **Operating System**: Ubuntu is recommended for optimal compatibility and performance.
- **Memory**: A minimum of 32GB of RAM is advised to ensure smooth execution.
- **CPU/GPU**: The project is designed to be hardware-agnostic, meaning it should run effectively on most modern systems without specific hardware requirements.

## Project Structure

```
qwen2-export-onnx
|   chat_onnx.py
|   export_onnx_qwen.py
|
+---output_onnx
|
\---[Your huggingface model dir] ( local model from huggingface )
        [model, config and tokenizer files]
```

### File Descriptions

- **`chat_onnx.py`**: Script for loading and running the ONNX model, using the configuration and tokenizer files from your model directory.
- **`export_onnx_qwen.py`**: Script to convert your existing model into the ONNX format.
- **`output_onnx`**: Directory where the exported ONNX models are saved.
- **`[Your huggingface model dir]`**: Placeholder for your model directory containing the original model files, configuration, and tokenizer.

## Getting Started

### Step 1: Export the Model to ONNX

To begin, convert your model to the ONNX format by running:

```bash
python export_onnx_qwen.py [Your huggingface model dir]
```
example:
```bash
export_onnx_qwen.py huggingface-qwen
```

- **`[Your model dir]`**: Replace this with the path to your model directory.
- The converted ONNX models will be saved in the `output_onnx` directory as `language-model.onnx` and `logits-model.onnx`.

### Step 2: Run the Model

After exporting the model, you can run it using:

```bash
python chat_onnx.py [Your huggingface model dir]
```

- **`chat_onnx.py`**: This script will automatically search for the `language-model.onnx` and `logits-model.onnx` files in the `output_onnx` directory.
- The command-line arguments will load the tokenizer and configuration files necessary for inference. This step is essential because different models have varying structures, such as the number of layers, KV cache head counts, and hidden sizes (e.g., the QWEN 2 0.5B model has 24 layers, while the 1.5B model has 28 layers). The configuration file ensures correct model execution during inference.


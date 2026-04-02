# Understanding Mechanistic Interpretability Under Model Compression for Transformer Based Language Models
---

## Repository Structure & Code Organization
The experimental pipeline is divided into distinct Jupyter Notebooks based on the compression technique. Inside each notebook, the code is logically separated by **Markdown headings** to clearly define individual experiments, setup steps, and evaluation loops.

* `Pruning.ipynb`: Contains the implementation and evaluation of magnitude and random pruning techniques across different sparsity levels.
* `Quantization.ipynb`: Contains the code for applying and testing quantization methods.
* `Distillation.ipynb`: Contains the pipeline for Knowledge Distillation experiments.
* `Attribution Pruning.ipynb`: Contains the code required to run the ablation study and analyze specific component importance.

## Results & Outputs
The individual results and generated images for each experiment are saved in their respective output folders as uploaded.

---

## Pre-trained Models & Reproducibility
Due to GitHub's file size constraints, the heavy model weights (`.safetensors`) generated during these experiments are not stored in this repository. 

To ensure strict reproducibility for the evaluation metrics reported in this study, all pruned and quantized models have been frozen and hosted publicly on the Hugging Face Hub.

**[Access the Pruned Models Here](https://huggingface.co/Nabeelah04/pruned_models) or (https://drive.google.com/drive/folders/1hHhax3B8KTWD8i88Q85wlHd1CwBTJ6I5?usp=sharing)**
**[Access the Quantized Models Here](https://huggingface.co/Nabeelah04/quantized_models) or (https://drive.google.com/drive/folders/1aWaCGkP8-dCT2SahRq-gv7RxuqbgIrpe?usp=sharing)**

### How to Load the Models
You can load any of the saved models directly via the `transformers` library without needing to download them manually:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Example: Loading a pruned model from Hugging Face
model_id = "Nabeelah04/pruned_models/magnitude_attention_heads_sparsity30"

tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

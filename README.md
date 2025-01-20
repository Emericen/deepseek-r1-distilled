<p align="left">
    &nbspEnglish&nbsp | <a href="README_CN.md">中文</a>
</p>

# ✨ Tiny Qwen

A minimal, easy-to-read PyTorch re-implementation of Qwen2 and Qwen2.5, the open source multi-modal LLM.

If you find [Transformers](https://github.com/huggingface/transformers) code verbose and challenging to interprete, this repo is for you! Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [litGPT](https://github.com/Lightning-AI/litgpt), it supports text-only versions (Instruct, Coder, Math, etc.) and text+vision versions (VL). It also supports all full prevision Qwen2+ model at any size. Just choose a repo id from Hugging Face Hub [here](https://huggingface.co/Qwen). 

Keep in mind you'll likely need multiple GPU for models bigger than 32B. Stay tuned for FSDP support in the coming days. If you run into any issues, open a PR or create an issue.

## **Interested in building vision-based AI Agents?**

I’m passionate about automating computer use to free up human labor and would love to collaborate with like-minded people. If this sound like you, please don't hesitate to reach out to me 🤗 ([my bio](https://github.com/Emericen))!

# 🦋 Quick Start

I recommend installing torch with cuda enabled (see [here](https://pytorch.org/get-started/locally/)). After that, simply run:

```bash
pip install -r requirements.txt
```

You can use the code base like the following:

```python
from models.model import Qwen2, Qwen2VL
from models.processor import Processor
from PIL import Image

# text-only models
model_name = "Qwen/Qwen2.5-3B"
model = Qwen2.from_pretrained(repo_id=model_name, device="cuda")
processor = Processor(repo_id=model_name)

context = [
    "<|im_start|>user\nwhat is the meaning of life?<|im_end|>\n<|im_start|>assistant\n"
]
inputs = processor(context, device="cuda")
output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=64)
output_text = processor.tokenizer.decode(output[0].tolist())

# text + vision models
model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VL.from_pretrained(repo_id=model_name, device="cuda")
processor = Processor(
    repo_id=model_name,
    vision_config=model.config.vision_config,
)

context = [
    "<|im_start|>user\n<|vision_start|>",
    Image.open("images/test-image.jpeg"),
    "<|vision_end|>What's on this image?<|im_end|>\n<|im_start|>assistant\n",
]
inputs = processor(context, device="cuda")
output = model.generate(
    input_ids=inputs["input_ids"],
    pixels=inputs["pixels"],
    d_image=inputs["d_image"],
    max_new_tokens=64,
)
output_text = processor.tokenizer.decode(output[0].tolist())
```

# 🛠️ Fine-tune / Post-train Your Own

See `train/train_sft.py` for simple SFT example. Any library compatible with `torch.nn.Module` wourld work, but here I used [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/index.html) for its flexibility and simplcity. Also see `train/train_mnist.py` for inspiration on how to use this library.

To run any of the training scripts, just run:

```bash
PYTHONPATH=. python train/train_mnist.py
```

or

```bash
PYTHONPATH=. python train/train_sft.py
```

[English](README.md) | [中文](README_CN.md)

# ✨ Tiny Qwen

一个简洁易读的 PyTorch 代码库，用于重新实现 Qwen2 和 Qwen2.5（开源多模态大模型）。

如果你觉得 [Transformers](https://github.com/huggingface/transformers) 代码太庞大难读，那么这个仓库可能更适合你！灵感来源于 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [litGPT](https://github.com/Lightning-AI/litgpt)，可同时支持纯文本模型（如 Instruct、Coder、Math 等）以及文本 + 图像（VL）。还支持任何全精度的 Qwen2+ 模型，尺寸不限。只需从 [Hugging Face](https://huggingface.co/Qwen) 选择一个 repo id 即可。

注意：大于 32B 的模型通常需要多块 GPU。我们会在今后加入 FSDP 支持。如果遇到问题，请随时提 Issue 或提交 PR。

此外，我在找志同道合的人合伙一起构建视觉 AI Agent。如果你对此感兴趣，请随时联系我🤗~ (我的主页在 [这里](https://github.com/Emericen))

---

## 🦋 快速开始

推荐先安装带 CUDA 的 PyTorch（见 [官方文档](https://pytorch.org/get-started/locally/)）。然后：

```bash
pip install -r requirements.txt
```

使用示例：

```python
from models.model import Qwen2, Qwen2VL
from models.processor import Processor
from PIL import Image

# 纯文本模型
model_name = "Qwen/Qwen2.5-3B"
model = Qwen2.from_pretrained(repo_id=model_name, device="cuda")
processor = Processor(repo_id=model_name)

context = [
    "<|im_start|>user\nwhat is the meaning of life?<|im_end|>\n<|im_start|>assistant\n"
]
inputs = processor(context, device="cuda")
output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=64)
output_text = processor.tokenizer.decode(output[0].tolist())

# 文本 + 图像模型
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

---

## 🛠️ 微调 / 自定义训练

查看 `train/train_sft.py` 以了解如何简单地对 Qwen 模型进行 SFT（有监督微调）。任意兼容 `torch.nn.Module` 的库都行，此处我用 [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/index.html) 来做训练。也可以参见 `train/train_mnist.py` 以获取思路。

运行示例：

```bash
PYTHONPATH=. python train/train_mnist.py
```

或

```bash
PYTHONPATH=. python train/train_sft.py
```

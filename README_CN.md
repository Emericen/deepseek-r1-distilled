[English](README.md) | [中文](README_CN.md)

# ✨ DeepSeek R1 Distilled

本代码库提供了基于 DeepSeek R1 蒸馏而来的 6 个小型模型的极简实现。DeepSeek R1 是一个通过大规模强化学习训练来执行思维链推理的LLM。这里的 6 个模型是基于 Qwen 和 Llama 的微调版本，使用 80万条由 DeepSeek R1 生成的思维链数据进行训练。简易期间，这里只使用了 SFT 进行微调，但强化学习可以进一步提升模型性能。

支持的模型：
- `DeepSeek-R1-Distill-Qwen-1.5B`
- `DeepSeek-R1-Distill-Qwen-7B`
- `DeepSeek-R1-Distill-Qwen-14B`
- `DeepSeek-R1-Distill-Qwen-32B`
- `DeepSeek-R1-Distill-Llama-8B`
- `DeepSeek-R1-Distill-Llama-70B`

更多信息请参考官方 [DeepSeek R1 原始仓库](https://github.com/deepseek-ai/DeepSeek-R1) 和 [DeepSeek R1 报告](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)。


另外，我在找志同道合的人来合伙构建基于视觉的 AI Agent。如果你对此感兴趣，请随时联系我🤗~ (我的主页在 [这里](https://github.com/Emericen))

## 🦋 快速开始

推荐先安装带 CUDA 的 PyTorch（见 [官方文档](https://pytorch.org/get-started/locally/)）。然后：

```bash
pip install -r requirements.txt
```

本代码库的使用方式如下：
```python
from model.model import DeepSeekR1Distilled, Processor

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model = DeepSeekR1Distilled.from_pretrained(repo_id=model_name, device_map="auto")
processor = Processor(repo_id=model_name)

context = ["55^0.12 等于多少?"]
# 正确答案是 1.61749714485

inputs = processor(context, device="cuda")
output = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=1024,
    temperature=0.6, 
    # 原文建议 temperature 为 0.5 到 0.8
)
output_text = processor.tokenizer.decode(output[0].tolist())
print(output_text)
```

结果（截取）
```
55^0.12 等于多少? 用自然对数计算
嗯，我现在要计算55的0.12次方，也就是55^0.12，用自然对数来计算。好，那我先回忆一下相关的数学知识。

首先，我记得指数和对数之间有关系，特别是自然对数，可以用它来表达指数运算。一般来说，a^b 可以写成 e^(b * ln a)，对吧？那也就是说，55^0.12 = e^(0.12 * ln 55)。嗯，这个方法对吗？让我再确认一下。

对的，这个方法没错。那接下来，我需要计算ln 55，也就是55的自然对数，然后再乘以0.12，最后再求e的这个结果次方。好，那我先来计算ln 55。

我不知道ln 55的具体数值是多少，可能需要用计算器或者近似值来计算。不过，我可以先回忆一下或者用泰勒展开来近似 吗？不过，这样可能会比较麻烦，或许直接用计算器更简单，不过这里假设我只能用手算，那我得想想怎么计算。

或者，我可以拆分一下，55等于5乘以11，所以ln 55 = ln(5*11) = ln5 + ln11。那我记得，ln5大约是1.6094，ln11大约 是2.3979。那加起来的话，ln5 + ln11 ≈1.6094 +2.3979=4.0073。所以，ln55≈4.0073。

对吗？让我再检查一下，因为有时候我记得不太准。比如，ln10≈2.3026，那么ln5≈ln(10/2)=ln10 - ln2≈2.3026 - 0.6931≈1.6095，对的，所以ln5≈1.6094。ln11，我知道比ln10大，ln11≈2.3979，对吗？对的，我记得没错，所以结合起来，ln55≈4.0073。

那接下来，0.12乘以ln55，也就是0.12乘以4.0073。我来计算一下，0.12 ×4=0.48，0.12×0.0073≈0.000876，所以总和大约是0.48 + 0.000876≈0.480876。所以，0.12 × ln55≈0.480876。

那么，接下来，我需要计算e^0.480876。嗯，这个值是多少呢？我记得e^0.480876可以用泰勒展开来近似计算，或者用已知的近似值。让我想想，e^0.480876，可以用泰勒展开式吗？

泰勒展开式在x=0处展开，e^x = 1 + x + x²/2! +x³/3! +x^4/4! +…，但因为x=0.480876不算特别小，所以可能需要更多的项才能得到比较准确的近似值。

或者，我可以记得一些关键点的e^x值，比如e^0.4≈1.4918，e^0.5≈1.6487，e^0.48≈1.6161，e^0.480876可能接近这个值
```
🤯🤯🤯

## 有没有人想一起构建视觉 AI Agent？

我在找志同道合的人来合伙。如果你对此感兴趣，请随时联系我🤗~ (我的主页在 [这里](https://github.com/Emericen))
[English](README.md) | [中文](README_CN.md)

# ✨ DeepSeek R1 Distilled

一个简洁易读的 DeepSeek R1 Distilled 的 PyTorch 实现。


此外，我在找志同道合的人合伙一起构建视觉 AI Agent。如果你对此感兴趣，请随时联系我🤗~ (我的主页在 [这里](https://github.com/Emericen))

---

## 🦋 快速开始

推荐先安装带 CUDA 的 PyTorch（见 [官方文档](https://pytorch.org/get-started/locally/)）。然后：

```bash
pip install -r requirements.txt
```

你可以像下面这样使用这个代码库：
```python
from model.model import DeepSeekR1Distilled
from model.processor import Processor

# text-only models
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model = DeepSeekR1Distilled.from_pretrained(repo_id=model_name, device_map="auto")
processor = Processor(repo_id=model_name)

context = ["<|im_start|>user\nwhat is 55^0.12<|im_end|>\n<|im_start|>assistant\n"]
# 55^0.12 is roughly 1.61749714485

inputs = processor(context, device="cuda")
output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=1024)
output_text = processor.tokenizer.decode(output[0].tolist())
print(output_text)
```

结果（截取）
```
Okay, so I need to figure out what 55 raised to the power of 0.12 is. Hmm, let's see. I remember that 
exponents can be a bit tricky, especially when they're not whole numbers. So, 0.12 is the same as 
12/100, which simplifies to 3/25. That means 55^(0.12) is the same as the 25th root of 55 cubed. Wait, 
is that right? Let me double-check. Yeah, because when you have a fractional exponent like a/b, it's 
the same as taking the bth root of a^a. So, 55^(3/25) is indeed the 25th root of 55 cubed.

But calculating the 25th root of something seems complicated. Maybe there's a better way to approach 
this. I think using logarithms could help. If I take the natural logarithm of 55, multiply it by 0.12, 
and then exponentiate the result, that should give me the answer. Let me write that down:

ln(55) ≈ 4.007333146

Now, multiplying that by 0.12:

4.007333146 * 0.12 ≈ 0.48088

So, e^(0.48088) should be approximately equal to 55^0.12. Let me calculate e^0.48088. I know that 
e^0.4 is about 1.4918, and e^0.48 is roughly 1.6161. Since 0.48088 is just a bit more than 0.48, maybe 
around 1.617 or 1.618. Wait, that's close to the golden ratio, but I don't think that's relevant 
here. Let me use a calculator for a more precise value.

Alternatively, I could use the common logarithm instead. Let's try that. Log base 10 of 55 is 
approximately 1.7403627. Multiplying that by 0.12 gives:

1.7403627 * 0.12 ≈ 0.2088435

Now, 10 raised to the power of 0.2088435. I know that 10^0.2 is about 1.5849, and 10^0.2088435 should 
be slightly higher. Maybe around 1.616 or so. That seems consistent with the natural logarithm method.

Wait, both methods gave me approximately the same result, around 1.616. That makes me more confident 
that the answer is correct.
```
🤯🤯🤯
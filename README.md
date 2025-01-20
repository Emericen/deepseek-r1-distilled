<p align="left">
    &nbspEnglish&nbsp | <a href="README_CN.md">中文</a>
</p>

# ✨ DeepSeek R1 Distilled

A minimal re-implementation of DeepSeek R1 Distilled, test-time scaled reasoning LLM.

## **Interested in building vision-based AI Agents?**

I’m passionate about automating computer use to free up human labor and would love to collaborate with like-minded people. If this sound like you, please don't hesitate to reach out to me 🤗 ([my bio](https://github.com/Emericen))!

# 🦋 Quick Start

I recommend installing torch with cuda enabled (see [here](https://pytorch.org/get-started/locally/)). After that, simply run:

```bash
pip install -r requirements.txt
```

You can use the code base like the following:

```python
from model.model import DeepSeekR1Distilled, Processor

# Available model options:
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# - deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# - deepseek-ai/DeepSeek-R1-Distill-Llama-70B

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model = DeepSeekR1Distilled.from_pretrained(repo_id=model_name, device_map="auto")
processor = Processor(repo_id=model_name)

context = ["<|im_start|>user\nwhat is 55^0.12<|im_end|>\n<|im_start|>assistant\n"]
# 55^0.12 is roughly 1.61749714485

inputs = processor(context, device="cuda")
output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=128)
output_text = processor.tokenizer.decode(output[0].tolist())

print(output_text)
```
Output (truncated):
```
Okay, so I need to figure out what 55 raised to the power of 0.12 is. Hmm, let's see. I remember that exponents can be a bit tricky, especially when they're not whole numbers. So, 0.12 is the same as 12/100, which simplifies to 3/25. That means 55^(0.12) is the same as the 25th root of 55 cubed. Wait, is that right? Let me double-check. Yeah, because when you have a fractional exponent like a/b, it's the same as taking the bth root of a^a. So, 55^(3/25) is indeed the 25th root of 55 cubed.

But calculating the 25th root of something seems complicated. Maybe there's a better way to approach this. I think using logarithms could help. If I take the natural logarithm of 55, multiply it by 0.12, and then exponentiate the result, that should give me the answer. Let me write that down:

ln(55) ≈ 4.007333146

Now, multiplying that by 0.12:

4.007333146 * 0.12 ≈ 0.48088

So, e^(0.48088) should be approximately equal to 55^0.12. Let me calculate e^0.48088. I know that e^0.4 is about 1.4918, and e^0.48 is roughly 1.6161. Since 0.48088 is just a bit more than 0.48, maybe around 1.617 or 1.618. Wait, that's close to the golden ratio, but I don't think that's relevant here. Let me use a calculator for a more precise value.

Alternatively, I could use the common logarithm instead. Let's try that. Log base 10 of 55 is approximately 1.7403627. Multiplying that by 0.12 gives:

1.7403627 * 0.12 ≈ 0.2088435

Now, 10 raised to the power of 0.2088435. I know that 10^0.2 is about 1.5849, and 10^0.2088435 should be slightly higher. Maybe around 1.616 or so. That seems consistent with the natural logarithm method.

Wait, both methods gave me approximately the same result, around 1.616. That makes me more confident that the answer is correct. But just to be thorough, maybe I should use a calculator to compute 55^0.12 directly. Let me do that.

Using a calculator, 55^0.12 equals approximately 1.616. So, that matches my earlier calculations. Therefore, I can conclude that 55 raised to the power of 0.12 is approximately 1.616.
```
🤯🤯🤯


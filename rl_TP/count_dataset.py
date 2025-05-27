from transformers import AutoTokenizer
from datasets import load_from_disk
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/liuyq/X-R1/Qwen3-8B")

# 定义一个函数来计算 token 数
def count_tokens(example):
    # 假设你的文本在 'text' 列中
    tokens = tokenizer(example['prompt'][0]['content'], truncation=False, padding=False)
    return len(tokens['input_ids'])

dataset = load_from_disk("data/rl_TP2")
for i, example in enumerate(dataset):
    print(f"样本 {i}:")
    # print(f"Prompt: {example['prompt']}")
    print(f"token数: {count_tokens(example)}")
    # 其他操作...
    if i >= 30:  # 仅示例，实际可删除
        break

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载原始模型
model_path = "/home/wangb/lyq/rl/X-R1/Qwen3-8B/Qwen/Qwen3-8B"  # 替换为你的原始模型路径
adapter_path = "/home/wangb/lyq/rl/X-R1/models/Qwen3-8B-GRPO-TP1"  # 替换为你的 Adapter 路径


# 加载原始模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配设备
    torch_dtype=torch.bfloat16  # 使用 bfloat16 数据类型
)

# 加载 Adapter
model = PeftModel.from_pretrained(model, adapter_path)
print("Adapter loaded successfully.")
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'

import re
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from prompts import SYSTEM_PROMPT


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a text generation model with optional system prompt.")
parser.add_argument("--model_name", type=str, required=True, help="Path or name of the model to load.")
args = parser.parse_args()

def format_correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
    '''
    格式正确性奖励函数
    该函数检查 completion 是否符合先 Thought 后 Action 的格式要求。
    '''
    responses = [completion[0]['content'] for completion in completions]
    pattern = r"Thought \d+.*[\n]?Action \d+.*"
    rewards = []
    q = prompts[0][-1]['content']
    print('-' * 20,  f"Question:\n{q}", f"\nResponse:\n{responses[0]}")
    for r in responses:
        if re.search(pattern, r):
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards


def action_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    奖励函数：检查completion中的Action格式是否正确

    参数:
        prompts (list): 输入的prompt列表
        completions (list): 模型生成的completion列表
        answer (list): 正确答案列表（在此任务中可能不需要）
        **kwargs: 其他可选参数

    返回:
        list[float]: 奖励值列表（1.0表示格式正确，-1.0表示格式错误）
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    # 定义合法的Action类型和参数模式
    valid_action_types = {
        "RestaurantInfo": r"RestaurantInfo\[([^\]]+)\]",
        "RestaurantSearch": r"RestaurantSearch\[(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(\d+)\]",
        "AttractionInfo": r"AttractionInfo\[([^\]]+)\]",
        "AttractionDistance": r"AttractionDistance\[([^,]+),\s*([^\]]+)\]",
        "AttractionSearch": r"AttractionSearch\[(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(\d+)\]",
        "Planner": r"Planner\[([^\]]+)\]"
    }

    for response in responses:
        # 检查是否存在Action
        action_match = re.search(r"Action \d+: (.*?)(?=\n|$)", response)
        if not action_match:
            rewards.append(-1.0)  # 没有Action，格式错误
            continue

        action_content = action_match.group(1).strip()

        # 检查Action类型和参数格式
        is_valid = False
        for action_type, pattern in valid_action_types.items():
            if re.fullmatch(pattern, action_content):
                is_valid = True
                break

        if is_valid:
            rewards.append(1.0)  # 格式正确
        else:
            rewards.append(-1.0)  # 格式错误

    return rewards

def action_diversity_reward_func(prompts, completions, last_action, **kwargs) -> list[float]:
    '''
    Action 多样性奖励函数
    此函数检查当前的 Action 与上一步的 Action 是否不同，若不同则给予奖励。
    '''
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for r, la in zip(responses, last_action):
        current_action_match = re.search(r'Action \d+: (.*?)(?=\n|$)', r)
        if current_action_match:
            current_action = current_action_match.group(1)
            if la and current_action == la:
                rewards.append(-2.0)
            else:
                rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards

def action_function_repetition_reward_func(prompts, completions, last_action, **kwargs) -> list[float]:
    '''
    Action 函数重复惩罚函数
    该函数检查当前 Action 与上一步的 Action 函数是否相同，若相同则给予负奖励。
    :param completions:
    :param last_action:
    :param kwargs:
    :return:
    '''
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for r, la in zip(responses, last_action):
        current_action_match = re.search(r'Action \d+: (\w+)\[', r)
        if current_action_match:
            current_func = current_action_match.group(1)
            if la:
                last_action_match = re.search(r'(\w+)\[', la)
                if last_action_match and last_action_match.group(1) == current_func:
                    rewards.append(-0.2)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def search_action_increase_reward_func(prompts, completions, last_action, **kwargs) -> list[float]:
    '''
    搜索类Action数量增加检查函数
    检查当前Action和上一步Action是否为AttractionSearch或RestaurantSearch，
    并且第三个参数(数量)是否比上一步增加了至少3，否则给予负奖励

    :param completions: 当前完成的动作列表
    :param last_action: 上一步的动作
    :param kwargs: 其他参数
    :return: 奖励值列表
    '''
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for r, la in zip(responses, last_action):
        # 检查当前Action
        current_action_match = re.search(r'Action \d+: (\w+)\[(.+)\]', r)

        if current_action_match:
            current_func = current_action_match.group(1)
            current_params = current_action_match.group(2).split(',')

            # 只处理AttractionSearch和RestaurantSearch
            if current_func in ['AttractionSearch', 'RestaurantSearch'] and len(current_params) >= 3:
                try:
                    current_count = int(current_params[2].strip())
                except (ValueError, IndexError):
                    rewards.append(0.0)
                    continue

                # 检查上一步Action
                if la:
                    last_action_match = re.search(r'(\w+)\[(.+)\]', la)
                    if last_action_match and last_action_match.group(1) in ['AttractionSearch', 'RestaurantSearch']:
                        last_params = last_action_match.group(2).split(',')
                        if len(last_params) >= 3:
                            try:
                                last_count = int(last_params[2].strip())
                                if current_count >= last_count + 3:
                                    rewards.append(0.0)  # 满足条件，不给奖励也不惩罚
                                else:
                                    rewards.append(-0.3)  # 不满足增加至少3的条件
                            except (ValueError, IndexError):
                                rewards.append(-0.3)  # 参数解析失败
                        else:
                            rewards.append(-0.3)  # 参数数量不足
                    else:
                        rewards.append(0.0)  # 上一步不是搜索类Action
                else:
                    rewards.append(0.0)  # 没有上一步Action
            else:
                rewards.append(0.0)  # 当前不是搜索类Action
        else:
            rewards.append(0.0)  # 没有匹配到Action

    return rewards


output_dir = "models/" + args.model_name + "-GRPO-TP3"
run_name = args.model_name + "-GRPO-TP3"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1, # 每个设备的训练批次大小
    gradient_accumulation_steps=8, # 梯度累积步数
    num_generations=4, # 每个prompt生成的样本数
    max_prompt_length=4096,
    max_completion_length=256,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda",
)

# 1. load a pretrained model

model_path = "./" + args.model_name
if args.model_name == "Qwen3-8B":
    model_path = "/home/liuyq/X-R1/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

# 定义LoRA配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# 使用get_peft_model得到PEFT模型
peft_model = get_peft_model(model, lora_config)


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
print("Model and tokenizer loaded successfully.")
dataset = load_from_disk("data/rl_TP2")
print("Datasets initialized successfully.")
trainer = GRPOTrainer(
    model=peft_model,
    processing_class=tokenizer,
    reward_funcs=[
        format_correctness_reward_func,
        action_format_reward_func,
        action_diversity_reward_func,
        action_function_repetition_reward_func,
        search_action_increase_reward_func,],
    args=training_args,
    train_dataset=dataset,
)
print("Trainer initialized successfully.")
trainer.train()
print("Train successful.")
trainer.save_model(output_dir)
print("Save successful.")
# python rl_TP/train_grpo_TP.py --model_name Qwen3-8B
# nohup python rl_TP/train_grpo_TP.py --model_name Qwen3-8B > grpo_output_TP3.log 2>&1 &
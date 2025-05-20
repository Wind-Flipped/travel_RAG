import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'

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
        "RestaurantInfo": r"RestaurantInfo$$([^$$]+)\]",
        "RestaurantSearch": r"RestaurantSearch$$([\d.,]+),\s*([\d.,]+),\s*(\d+)$$",
        "AttractionInfo": r"AttractionInfo$$([^$$]+)\]",
        "AttractionDistance": r"AttractionDistance$$([^$$]+),\s*([^\]]+)\]",
        "AttractionSearch": r"AttractionSearch$$([\d.,]+),\s*([\d.,]+),\s*(\d+)$$",
        "Planner": r"Planner$$([^$$]+)\]"
    }

    for response in responses:
        # 检查是否存在Action
        action_match = re.search(r"Action \d+:(.*?)(?=\n|$)", response)
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
        current_action_match = re.search(r'Action \d+:(.*?)(?=\n|$)', r)
        if current_action_match:
            current_action = current_action_match.group(1)
            if la and current_action == la:
                rewards.append(-0.2)
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
        current_action_match = re.search(r'Action \d+:(\w+)\[', r)
        if current_action_match:
            current_func = current_action_match.group(1)
            if la:
                last_action_match = re.search(r'Action \d+:(\w+)\[', la)
                if last_action_match and last_action_match.group(1) == current_func:
                    rewards.append(-0.2)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def search_param_increase_reward_func(prompts, completions, last_action, **kwargs) -> list[float]:
    """
    奖励函数：检查当前Action和上一步Action是否为AttractionSearch或RestaurantSearch，
    并且第三个参数是否比上一步增加了至少3。如果没有满足条件，则给予负奖励。

    参数:
        prompts (list): 输入的prompt列表
        completions (list): 模型生成的completion列表
        answer (list): 正确答案列表（在此任务中可能不需要）
        last_action (str): 上一步的Action（由调用者传入）
        **kwargs: 其他可选参数

    返回:
        list[float]: 奖励值列表
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response, la in zip(responses, last_action):
        # 提取当前Action
        current_action_match = re.search(r'Action \d+:(.*?)(?=\n|$)', response)
        if not current_action_match:
            rewards.append(-1.0)  # 格式错误
            continue

        current_action = current_action_match.group(1)

        # 检查是否为AttractionSearch或RestaurantSearch
        if not (current_action.startswith("AttractionSearch[") or current_action.startswith("RestaurantSearch[")):
            rewards.append(0.0)  # 不是这两种Action，不奖励也不惩罚
            continue

        # 提取当前Action的参数
        current_params_match = re.search(r'$$(.*?)$$', current_action)
        if not current_params_match:
            rewards.append(-1.0)  # 参数格式错误
            continue

        current_params = current_params_match.group(1).split(',')
        if len(current_params) < 3:
            rewards.append(-1.0)  # 参数不足
            continue

        try:
            current_param3 = float(current_params[2].strip())
        except ValueError:
            rewards.append(-1.0)  # 参数不是数字
            continue

        # 提取上一步Action
        if not la:
            rewards.append(0.0)  # 没有上一步Action
            continue

        last_action_match = re.search(r'Action \d+:(.*?)(?=\n|$)', la)
        if not last_action_match:
            rewards.append(0.0)  # 上一步Action格式错误，与当前步无关
            continue

        last_action_content = last_action_match.group(1)

        # 检查上一步是否为AttractionSearch或RestaurantSearch
        if not (last_action_content.startswith("AttractionSearch[") or last_action_content.startswith(
                "RestaurantSearch[")):
            rewards.append(0.0)  # 上一步不是这两种Action，不奖励也不惩罚
            continue

        # 提取上一步Action的参数
        last_params_match = re.search(r'$$(.*?)$$', last_action_content)
        if not last_params_match:
            rewards.append(0.0)  # 上一步参数格式错误
            continue

        last_params = last_params_match.group(1).split(',')
        if len(last_params) < 3:
            rewards.append(0.0)  # 上一步参数不足
            continue

        try:
            last_param3 = float(last_params[2].strip())
        except ValueError:
            rewards.append(0.0)  # 上一步参数不是数字
            continue

        # 计算参数增加量
        param_increase = current_param3 - last_param3

        # 给予奖励或惩罚
        if param_increase >= 3:
            rewards.append(0.3)  # 满足条件，给予奖励
        else:
            # 差距越小，惩罚越重（最小惩罚-1.0）
            penalty = -min(1.0, (3 - param_increase) * 0.5)
            rewards.append(penalty)


    return rewards  # 确保长度匹配


output_dir = "models/" + args.model_name + "-GRPO-TP2"
run_name = args.model_name + "-GRPO-TP2"

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
    per_device_train_batch_size=4, # 每个设备的训练批次大小
    gradient_accumulation_steps=4, # 梯度累积步数
    num_generations=4, # 每个prompt生成的样本数
    max_prompt_length=256,
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
    model_path = "./Qwen3-8B/Qwen/Qwen3-8B"
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
dataset = load_from_disk("/home/wangb/cyo/graduation/data/rl_TP2")
print("Datasets initialized successfully.")
trainer = GRPOTrainer(
    model=peft_model,
    processing_class=tokenizer,
    reward_funcs=[
        format_correctness_reward_func,
        action_format_reward_func,
        action_diversity_reward_func,
        action_function_repetition_reward_func,],
    args=training_args,
    train_dataset=dataset,
)
print("Trainer initialized successfully.")
trainer.train()
print("Train successful.")
trainer.save_model(output_dir)
print("Save successful.")
# python rl_TP/train_grpo_TP.py --model_name Qwen3-8B
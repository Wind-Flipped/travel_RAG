import json
import os
from datasets import Dataset
from prompts import SYSTEM_PROMPT

def build_cumulative_prompts_from_json_files(folder_path):
    """
    从指定文件夹中的JSON文件构建累积prompts（不包含最后一个step）

    参数:
        folder_path (str): 包含JSON文件的文件夹路径

    返回:
        list: 包含所有累积prompts的列表
    """
    all_prompts = []
    last_prompts = []
    last_thoughts = []
    last_actions = []
    last_observations = []
    query_prompts = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        main_prompt = item.get("Qwen3-8B_two-stage_query", "")
                        if not main_prompt:
                            continue  # 跳过没有主prompt的文件
                        all_prompts.append(main_prompt)
                        last_prompts.append("") # 如果是第一个prompt，添加一个空字符串
                        last_thoughts.append("")
                        last_actions.append("")
                        last_observations.append("")
                        query_prompts.append(main_prompt)

                        action_logs = item.get("Qwen3-8B_two-stage_action_logs", [])
                        if not isinstance(action_logs, list):
                            continue  # 跳过没有action logs的文件

                        # 初始化历史记录（只包含主prompt）
                        history = [main_prompt]

                        # 处理每个step（除了最后一个）
                        for i in range(len(action_logs) - 1):  # 不处理最后一个step
                            if isinstance(action_logs[i], dict):
                                step_entry = (
                                    f"Thought {action_logs[i].get('step', '')}: {action_logs[i].get('thought', '')}。"
                                    f"Action {action_logs[i].get('step', '')}: {action_logs[i].get('action', '')}\n"
                                    f"Observation {action_logs[i].get('step', '')}: {action_logs[i].get('observation', '')}"
                                )
                                history.append(step_entry)
                                last_prompts.append(step_entry)
                                last_thoughts.append(action_logs[i].get('thought', ''))
                                last_actions.append(action_logs[i].get('action', ''))
                                last_observations.append(action_logs[i].get('observation', ''))
                                query_prompts.append(main_prompt)

                                # 构建累积prompt
                                cumulative_prompt = "\n".join(history)
                                all_prompts.append(cumulative_prompt)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return all_prompts, last_prompts, last_thoughts, last_actions, last_observations, query_prompts


# 使用示例
folder_path = "/home/wangb/cyo/graduation/logs_Qwen3-8B/test/zero_shot_zh_new_prompt"  # 替换为你的JSON文件夹路径
prompts, last_prompts, last_thoughts, last_actions, last_observations, query_prompts = build_cumulative_prompts_from_json_files(folder_path)

# 打印前几个prompts作为示例
for i, prompt in enumerate(prompts[:15], 1):
    print(f"Prompt {i}:\n{prompt}\n{'-' * 50}")

# 存储为对话格式（如果需要）
# dataset_dict = {
#     "prompt": [[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": "查询："+ query_prompt + "\n" + prompt + " /no_think"},
#     ] for query_prompt, prompt in zip(query_prompts, prompts)],
#     "last_prompt": [prompt for prompt in last_prompts],
#     "last_thought": [thought for thought in last_thoughts],
#     "last_action": [action for action in last_actions],
#     "last_observation": [observation for observation in last_observations],
#     "query_prompt": [prompt for prompt in query_prompts],
# }

dataset_dict = {
    "prompt": [[
        {"role": "user", "content": SYSTEM_PROMPT + "查询：" + prompt + " /no_think"},
    ] for query_prompt, prompt in zip(query_prompts, prompts)],
    "last_prompt": [prompt for prompt in last_prompts],
    "last_thought": [thought for thought in last_thoughts],
    "last_action": [action for action in last_actions],
    "last_observation": [observation for observation in last_observations],
    "query_prompt": [prompt for prompt in query_prompts],
}

# 转换为 Dataset
dataset = Dataset.from_dict(dataset_dict)
print(dataset)
dataset.save_to_disk("/home/wangb/cyo/graduation/data/rl_TP2")  # 保存为 Dataset 目录（推荐）
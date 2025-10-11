from zhipu import Zhipu
import json
from tqdm import tqdm
from openai import OpenAI

llm = Zhipu(api_key="ff9747c0ac6262a47b8c9a80b253624b.pPgwRsgvGv5sjuph").get_model()
aili_llm = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-09df1e424131460f81a5a431ccb9af55",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
with open(f"../databases/xihu_1/key_place2_td_requests.json", 'r', encoding='utf-8') as f:
    all_data = json.load(f)
    print("总共有" + str(len(all_data)) + "条数据")


def zero_format_prompt(question):
    template = '''
你好，你是一个旅游线路规划的专家。用户希望你能帮忙推荐路线。请根据用户输入来回答用户的问题。
--
###用户输入：
问题: {question}

#输出：
- 返回一个json格式的数据，具有以下键值对：
    - **route**：你推荐的路线,是一个地点列表，用逗号分开。
    - **reason**：选择这条路线的原因。
- 你的返回应该是以下json格式的数据：
        {{
        "route": "你推荐的路线",
        "reason": "选择这条路线的原因",
        }}

#注意：
-路线是一个地点列表，用逗号分开，不要有多余内容。
-总是使用中文回答。输出格式请符合输出规范。
- 返回不包括任何其他内容。
- 以一个 JSON返回。
-路线中的地点和景点名之间请用','隔开
-选取路线和景点中包含的地点至多出现一次，即不要重复浏览。
'''
    return template.format(question=question)


def only_answer_prompt(question):
    template = '''
你好，你是一个旅游线路规划的专家。用户希望你能帮忙推荐路线。请根据用户输入来回答用户的问题。
--
###用户输入：
问题: {question}

#输出：
- 返回一个json格式的数据，具有以下键值对：
    - **route**：你推荐的路线,是一个地点列表，用逗号分开。
- 你的返回应该是以下json格式的数据：
        {{
        "route": "你推荐的路线",
        }}

#注意：
-路线是一个地点列表，用逗号分开，不要有多余内容。
-总是使用中文回答。输出格式请符合输出规范。
- 返回不包括任何其他内容。
- 以一个 JSON返回。
-路线中的地点和景点名之间请用','隔开
-选取路线和景点中包含的地点至多出现一次，即不要重复浏览。
'''
    return template.format(question=question)


baselines = ["only_answer"]
model_name = "qwen-plus"
print(model_name)
for num in range(1):
    for baseline in baselines:
        for item in tqdm(all_data, desc=baseline):
            if baseline == "zero_shot":
                request = zero_format_prompt(item['ai_input'])
            elif baseline == "only_answer":
                request = only_answer_prompt(item['ai_input'])
            else:
                break
            response = aili_llm.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": request
                    }
                ],
            )
            response = response.choices[0].message.content
            print(response)
            item['response'] = response

        with open(f"../output/baselines_{baseline}_td_{model_name}_{num}.json", 'w',
                  encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

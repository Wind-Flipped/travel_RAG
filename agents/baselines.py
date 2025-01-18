import os
os.environ["https_proxy"] = "http://localhost:7890"
os.environ["http_proxy"] = "http://localhost:7890"
from zhipuai import ZhipuAI
import json
from tqdm import tqdm
from evaluate import Evaluator
from openai import OpenAI
import time


class LLMs:
    def __init__(self, model_name: str = 'glm-4-air', temperature: float = 0.9,
                 api_key = "c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE",
                 rag_database: str = "/home/wangb/cyo/graduation/rag/databases/hangzhou") -> None:
        # 初始化大模型
        self.model_name = model_name
        self.temperature = temperature
        if 'glm-4' in model_name:
            self.model = ZhipuAI(api_key=api_key)
        elif model_name == 'deepseek-chat':
            print("Using deepseek-chat")
            self.model = OpenAI(api_key="sk-a416cf4db0f246ae9fd6f9c620e11d9f", base_url="https://api.deepseek.com")
        elif 'gpt-4o' in model_name:
            print("Using gpt-4o")
            self.model = OpenAI(api_key="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", base_url="https://openrouter.ai/api/v1")


        self.prompt_token = 0
        self.completion_token = 0

    # 定义chat方法
    def __call__(self, prompt: str, stop: list = None):
        response = self.model.chat.completions.create(
            model=self.model_name,
            stop = stop,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        self.prompt_token += response.usage.prompt_tokens
        self.completion_token += response.usage.completion_tokens

        return response.choices[0].message.content


    def get_model(self):
        return self.model

    def get_tokens(self):
        return self.prompt_token, self.completion_token
def zero_format_prompt(question):
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
        "route": "你推荐的路线"
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


def zero_format_prompt2(question):
    PLANNER_ZERO_SHOT_ZH = """您是一位熟练的规划师。根据提供的信息和查询，请为我提供一份详细的计划，包括餐厅名称和景点名称等具体内容。用户查询里想要去的景点必须要包含在计划内。
您必须遵循示例中给出的格式。此外，所有细节应符合常识。符号 '-' 表示该信息不必要。
返回的结果以json格式输出，包括"交通"，"早餐"，"上午景点"，"午餐"，"下午景点"，"晚餐"，"晚上景点"属性。若在一个时间段内要去多个景点，请按照顺序依次写出，并用符号 ',' 分隔。
- 返回一个json格式的数据，具有以下键值对：
    - **交通**：推荐的出行方式。
    - **早餐**: 选择早上吃什么，可以不吃早餐。
    - **上午景点**：推荐上午去游玩的景点，必须要有至少一个景点。这里只能填写查询到的景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
    - **午餐**: 选择中午吃什么，必须要有一个餐厅。这里只能填写餐厅的具体名称，如果有括号内容也需要将括号内容填入。不能填写景点。
    - **下午景点**：推荐下午去游玩的景点，必须要有至少一个景点。这里只能填写查询到的景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
    - **晚餐**: 选择晚上吃什么，必须要有一个餐厅。这里只能填写餐厅的具体名称，如果有括号内容也需要将括号内容填入。不能填写景点。
    - **晚上景点**：推荐晚上去游玩的景点，可以没有。如果有的话，这里只能填写景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
***** 示例 *****
查询：您能为4人制定一份在杭州的1日游计划，预算为1200元吗？ 
旅行计划： 
{{
    "交通": "地铁" ,
    "早餐": "-",
    "上午景点": "西湖",
    "午餐": "湘香蒸菜馆(湘湖路店)",
    "下午景点":  "杭州博物馆, 净慈寺, 雷峰塔",
    "晚餐": "成都遇见你·串串吧(萧山店)",
    "晚上景点": "-"
}}

***** 示例结束 *****

查询：{query}
旅行计划：

"""
    return PLANNER_ZERO_SHOT_ZH.format(query=question)


if __name__ == '__main__':
    start_time = time.time()
    react_llm_name = "glm-4-air"
    llm = LLMs(model_name= react_llm_name)
    type = "simulated"
    if type == "real":

        with open(f"./rag/databases/hangzhou/key_place2_requests.json", 'r',
                  encoding='utf-8') as f:
            all_data = json.load(f)
            print("总共有" + str(len(all_data)) + "条数据")

        evaluator = Evaluator(have_truth=True)
        i = 0
        for item in all_data:
            request = zero_format_prompt(item["input"])
            query = item["input"]
            response = llm(request)
            print(response)
            item['zero_shot'] = response
            evaluator.evaluate_baseline(response, item["target_place"], query, item["route"])
        evaluator.print_real_result(mode="baseline_zero_shot", model_name=react_llm_name)

        with open(f"./logs/baselines_zero_shot{react_llm_name}.json", 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

    else:
        with open(f"./data/base_request.json", 'r',
                  encoding='utf-8') as f:
            all_data = json.load(f)
            print("总共有" + str(len(all_data)) + "条数据")

        evaluator = Evaluator(have_truth=True)
        i = 0
        for item in all_data:
            request = zero_format_prompt2(item["query"])
            query = item["query"]
            response = llm(request)
            print(response)
            evaluator.evaluate(agent_output=response, externel_data=item)
            item['zero_shot'] = response
        evaluator.print_result(mode="baseline_zero_shot_simulated", model_name=react_llm_name)

        with open(f"./logs/baselines_zero_shot_simulated_{react_llm_name}.json", 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)


    end_time = time.time()
    prompt_token, completion_token = llm.get_tokens()
    print(f"Prompt tokens: {prompt_token}, Completion tokens: {completion_token}")
    print(f"Total time: {end_time - start_time} seconds")
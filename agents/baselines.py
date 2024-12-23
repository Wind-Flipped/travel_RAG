from zhipuai import ZhipuAI
import json
from tqdm import tqdm
from evaluate import Evaluator
import time

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

if __name__ == '__main__':
    start_time = time.time()
    llm = ZhipuAI(api_key="c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE")

    with open(f"/home/wangb/cyo/graduation/rag/databases/hangzhou/key_place2_requests.json", 'r',
              encoding='utf-8') as f:
        all_data = json.load(f)
        print("总共有" + str(len(all_data)) + "条数据")

    evaluator = Evaluator(have_truth=True)
    i = 0
    for item in all_data:
        # if i > 2:
        #     break
        # i += 1
        request = zero_format_prompt(item['ai_input'])
        query = item['ai_input']
        response = llm.chat.completions.create(
            model="glm-4-air",
            messages=[
                {
                    "role": "user",
                    "content": request
                }
            ],
        )
        response = response.choices[0].message.content
        print(response)
        item['zero_shot'] = response
        evaluator.evaluate_baseline(response, item["target_place"], query, item["route"])
    evaluator.print_real_result(mode="zero_shot")

    with open(f"/home/wangb/cyo/graduation/logs/baselines_zero_shot.json", 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
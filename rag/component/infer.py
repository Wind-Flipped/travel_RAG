import json
import re
from tqdm import tqdm
from llms import LLMs
import os

llm = LLMs(load_ratio=1, rag_database=["../databases/xihu_1",
                                       "../databases/hangzhou_poi"])

with open(f"../databases/xihu_1/key_place2_td_requests.json", 'r', encoding='utf-8') as f:
    all_data = json.load(f)
    requests = all_data
    print("总共有" + str(len(requests)) + "条数据")
top_k = 3
cluster = 3
error_num = 0
for index, item in tqdm(enumerate(requests)):
    llm.clear_history()
    request = item["ai_input"]
    # print('-' * 40)
    # print(item["input"])
    # print(request)
    pos_input, neg_input = llm.request_split(request)
    routes, pois = llm.query_databases(pos_input, neg_input, index, top_k, cluster)
    for route in routes:
        post_index = route.find("帖子")
        content_before_post = route[:post_index]
        # print(content_before_post)
    question = llm.question_format_prompt(input, routes, pois)
    response = llm.llm(question)
    path_plan = llm.path_plan(response)
    num = 0
    while num < 3:
        try:
            question_validation = llm.validation_format_prompt(path_plan, response, request)
            # print(question_validation)
            response_validation = llm.llm(question_validation)
            # print(response_validation)
            match = re.search(r'(\{.*})', response_validation, re.DOTALL)
            if match:
                extracted_json = match.group(1)
                data = json.loads(extracted_json)
            else:
                print("output wrong json")
                break
            # print(data["change"])
            if not data["change"]:
                response = response_validation
                print("输出无需校验")
                break
            path_plan = llm.path_plan(response_validation)
            response = response_validation
            num += 1
            name = f"index_{num}"
            item[name] = response_validation
        except (json.JSONDecodeError, ValueError, KeyError, json.decoder.JSONDecodeError) as e:
            # 捕获 JSON 解析或匹配异常，打印并重新执行从 llm.llm 开始的逻辑
            # print(f"JSON 解析错误或匹配失败: {e}")
            # print("重新生成 response_validation...")
            # print(response_validation)
            error_num += 1
            num += 1
            continue  # 回到循环开始，从 llm.llm(question_validation) 重新执行
    # print(response)
    item["response"] = response
output_filename = f"../output/output_requests_place2_td_k{top_k}_n{cluster}_xihu_2.json"
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(requests, f, ensure_ascii=False, indent=4)

print("error_num")
print(error_num)

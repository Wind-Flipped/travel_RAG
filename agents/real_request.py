import json
import random
import re
import pandas as pd
from llms import LLMs
from tqdm import tqdm


class RealRequest():
    def __init__(self, api_key = "8cf93821658b7df312645b6dc443b871.aY8DFiq17G0NrVmx"):

        with open("/home/wangb/cyo/graduation/rag/databases/hangzhou/document.json", "r", encoding="utf-8") as f1:
            self.data1 = json.load(f1)

        self.request = []
        self.model = LLMs(api_key=api_key)

    def extract_route(self, item):
        # 匹配“路线:”后面的地点，多个地点以逗号分隔
        match = re.search(r"路线[:：](.+)帖子", item)
        if match:
            # 提取并按逗号分隔，过滤掉空结果
            return [place.strip() for place in match.group(1).split(",") if place.strip()]
        return []


    def extract_key(self, item):
        match = re.search(r"关键词[:：](.+?)路线", item)
        if match:
            return [place.strip() for place in match.group(1).split(",") if place.strip()]
        return []

    def generate(self):
        for data in tqdm(self.data1):
            route = self.extract_route(data)
            key_word = self.extract_key(data)
            target_places = random.sample(route, 1)
            input_txt = "请帮我推荐一条杭州一日游的旅游路线，要求是：" + '，'.join(key_word) + '。除此，我还希望去' + '，'.join(
                target_places) + '游玩'
            print(input_txt)
            content = "请改写下面的文本，使之更像人提出的需求，不要包含除文本中提及的地点之外的地点。只输出结果。文本是：" + input_txt
            ai_input = self.model(content)
            self.request.append({
                "input": input_txt,
                "ai_input": ai_input,
                "route": route,
                "target_place": target_places
            })

        output_file = "/home/wangb/cyo/graduation/rag/databases/hangzhou/key_place2_requests.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.request, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    real_request = RealRequest()
    real_request.generate()
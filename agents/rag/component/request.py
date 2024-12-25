from zhipuai import ZhipuAI
import json
import re


class Request:
    def __init__(self, model, model_name="glm-4-air"):
        # 初始化大模型
        self.model = model
        self.pos_poi, self.neg_poi, self.pos_subrequest, self.neg_subrequest = [], [], [], []
        self.model_name = model_name

    def request2llm(self, request):
        prompt = f'''
您好，请帮我基于一段用户需求描述，将分解为正向需求和负向需求。请直接根据给出的**用户输入**返回按照下面的格式的结果，不需要编写任何代码。

---

### 输出格式：

返回一个json格式的数据，具有以下键值对：
- **pos_req**：正向需求，代表用户想要的，但不包括任何负向需求。
- **neg_req**：负向需求，一般是用户不想要的，不想去的，所有负向的没有针对具体地点的需求必须提取到这个字段内，比如“不辣的”应该提取出“辣的”，“不要人多的”应该提取出“人多”，“讨厌嘈杂的”应该提取出“嘈杂”。

- 你的返回应该是以下json格式的数据：
    {{
        "pos_req": "正向需求",
        "neg_req": "负向需求" 
    }}

- 其中，正面需求和负面需求不能有重合，比如pos_req里有"辣的",neg_req里有"不辣的"。
- 用户有时只会描述不想要什么（负向需求），针对没有**正向需求**的情况，应该基于**负向需求**总结一个**正向需求**。比如用户可能说‘不想吃辣的’，这时候输出应该包括："pos"对应"吃的"，"neg"对应"辣的"。
- 不要包含重复的独立的需求，要保证每个独立的需求是对应用户需求中不同的关键点。
- 请确保所有负向需求被提取到"neg"字段对应的内容中。
- "pos"字段的内容**不能包含否定的表达**（比如"不"，"不想"，"不要"等等），因为所有否定表达应该已经被提取到了"neg"字段内。

### 输出规范
- 返回一个json格式的数据，其中包含"pos_req"和"neg_req"两个键值对。
- 以一个 JSON返回。
- 输出中不应包含任何其他信息，确保你的返回可以被 json.loads 解析。

### 用户输入
{request}
'''
        response = self.model.chat.completions.create(
            model=self.model_name,  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ]
        )
        return response.choices[0].message.content

    def extract_requests(self, request):
        pattern = r"\{([^}]+)\}"
        match = re.search(pattern, self.request2llm(request))
        data = match.group(1) if match else None
        data = "{" + data + "}"
        data = json.loads(data)
        return data["pos_req"], data["neg_req"]


if __name__ == "__main__":
    question = "我想去南天门和南锣鼓巷玩，不想去王府井，但是又想避开人多的地方，晚上想去吃火锅或者烤肉，吃完去逛一逛天安门"

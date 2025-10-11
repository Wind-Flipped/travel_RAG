from zhipuai import ZhipuAI
from embedding import Zhipuembedding
from request import Request
from databases import Vectordatabase
from poi2route import Poi2Route
import re
import json


class LLMs:
    def __init__(self, model_name: str = 'Zhipu', temperature: float = 0.9,
                 rag_database: list[str] = ["/home/wangb/cyo/graduation/rag/databases/xihu_1",
                                            "/home/wangb/cyo/graduation/rag/databases/hangzhou_poi"],
                 load_ratio: float = 1) -> None:
        print("load_ratio", load_ratio)
        self.model_name = model_name
        self.temperature = temperature
        self.model = ZhipuAI(api_key="ff9747c0ac6262a47b8c9a80b253624b.pPgwRsgvGv5sjuph")
        self.request_generator = Request(self.model)
        self.db_route = Vectordatabase()
        self.db_route.load_vector(rag_database[0], load_ratio)
        self.db_poi = Vectordatabase()
        self.db_poi.load_vector(rag_database[1])
        self.embedding_model = Zhipuembedding(api_key="ff9747c0ac6262a47b8c9a80b253624b.pPgwRsgvGv5sjuph")
        self.path_planner = Poi2Route()
        self.history = []

    def llm(self, input2llm):
        self.history.append({"role": "user", "content": input2llm})
        answer = self.model.chat.completions.create(
            model="glm-4-air",  # 填写需要调用的模型名称
            messages=self.history
        )
        self.history.append({"role": "assistant", "content": answer.choices[0].message.content})
        return answer.choices[0].message.content

    def clear_history(self):
        self.history = []

    def request_split(self, input_txt):
        return self.request_generator.extract_requests(input_txt)

    def path_plan(self, input_text):
        return self.path_planner.input2answer(input_text)

    def query_route(self, pos_question, neg_question):
        info = self.db_route.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True)
        return info

    def query_poi(self, pos_question, neg_question):
        info = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True)
        return info

    # 定义chat方法
    def query_databases(self, pos_question, neg_question, invisible, top_k=3, cluster=3):
        routes = self.db_route.query_both(pos_question, neg_question, self.embedding_model, top_k, cluster, True,
                                          invisible)
        pois = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, top_k, cluster, True)
        return routes, pois

    def question_format_prompt(self, question, route_info, poi_info):
        # 这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
        template = '''
你好，你是一个旅游线路规划的专家。请根据用户输入并参考以上下文来回答用户的问题。如果你不知道答案，就说你不知道。
--
###用户输入：
问题: {question}

###可参考的上下文
-可参考的上下文分为两部分。
第一部分是是多条由关键词，路线和帖子组成的数据，代表旅行线路。
第二部分是多点景点，景点描述组成的数据，代表POI点。

-第一部分：
···
{info_route}
···

-第二部分：
···
{info_poi}
···

#输出：
- 返回一个json格式的数据，具有以下键值对：
    - **route**：从第一部分上下文选择的路线，仅包含路线。路线是一个地点列表，地点之间用','隔开。
    - **poi**: "从第二部分上下文选择的景点描述，仅包含景点名.
    - **reason**：选择这两个文本的原因。
    - **advantage**: 一段文本解释这条路线的优点。
- 你的返回应该是以下json格式的数据：
        {{
        "route": "从第一部分上下文选择的路线，仅包含路线",
        "poi": "从第二部分上下文选择的景点描述，仅包含景点名",
        "reason": "选择这个路线和景点的原因",
        "advantage": "一段文本解释这条路线的优点"
        }}

#注意
-如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
-总是使用中文回答。输出格式请符合输出规范。
- 返回不包括任何其他内容。
- 以一个 JSON返回。
-路线中的地点和景点名之间请用','隔开
-route字段和poi字段中的文本必须是参考上下文中某一个数据的路线或景点名，不能够修改。
-仅允许在route字段中出现一条路线。poi中可以为空，也可以出现0~3个景点名。
-如果你认为在第一部分的路线都不满足用户需求，你可以将多条路线合并或者删除路线中的某些地点，但请不要打乱路线中地点的顺序，尽可能保留原先线路的游览顺序。
-选取路线和景点中包含的地点至多出现一次，即不要重复浏览。
-即使你要对第一部分选取的线路做修改，也不要把第二部分的景点插入到第一部分选取的路线中。对于route字段，你只允许使用第一部分的路线进行合并、删减或不做更改。
-如果第一部分的旅游攻略中包含多条线路，你只允许推荐关键词之后帖子之前的那条线路。
'''
        format_route_info = "\n".join(f"-第{index + 1}条路线攻略: {item}" for index, item in enumerate(route_info))
        format_poi_info = "\n".join(f"-第{index + 1}条景点描述: {item}" for index, item in enumerate(poi_info))
        return template.format(question=question, info_route=format_route_info, info_poi=format_poi_info)

    def validation_format_prompt(self, validation, route, request):
        # 这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
        template = '''
对与你生成的路线{route},我通过调用工具得到了以下校验结果，请根据校验结果和用户需求改进路线并输出。如果用户没有交通时间需求和交通距离需求，则无需改进路线。如果用户有交通时间需求和交通距离需求，请确保路线的交通时间和交通距离和用户所提出的交通时间和交通距离差不多。
--
###校验结果：
{validation}

### 用户需求：
{request}

#输出：
- 返回一个json格式的数据，具有以下键值对：
    - **has_request**：用户是否有交通距离需求或交通时间需求。如果用户有交通距离需求或交通时间需求，填True，否则填False。
    - **route**：根据校验结果和用户需求和之前的候选路线和候选地点改进的路线。路线是一个地点列表，地点之间用','隔开。
    - **reason**：改进路线的原因。如果用户没有交通时间需求和交通距离需求，则无需改进路线。
    - **change**：如果改进路线，则填True，否则填False。如果用户没有交通时间需求和交通距离需求，则填False。
    - **advantage**: 一段文本解释这条路线的优点。
    - **consistency**：改进前的路线是否与用户的交通距离需求和交通时间需求差不多，如果差不多或者没有交通时间需求和交通距离需求，填True，否则填False。
- 你的返回应该是以下json格式的数据：
        {{
        "has_request":"如果用户有交通距离需求或交通时间需求，填True，否则填False",
        "route": "根据校验结果和用户需求和之前的候选路线和候选地点改进的路线",
        "reason": "改进路线的原因",
        "change": "如果改进路线，则填True，否则填False",
        "advantage": "一段文本解释这条路线的优点",
        "consistency"："改进前的路线是否与用户的交通距离需求和交通时间需求差不多，如果差不多或者没有交通时间需求和交通距离需求，填True，否则填False",
        }}

#注意
-改进路线时可以考虑之前的候选路线和候选景点，而不是简单地删减地点。
-如果用户没有交通时间需求和交通距离需求，则无需改进路线,"change"字段置为False。
-总是使用中文回答。输出格式请符合输出规范。
- 返回不包括任何其他内容。
- 以一个 JSON返回。
-路线中的地点和景点名之间请用','隔开。
-route字段和poi字段中的文本必须是参考上下文中某一个数据的路线或景点名，不能够修改。
-可以不改进路线，只需要将字段"change"置为False。
-新的路线必须基于之前参考的上下文中的景点和线路攻略。
-选取路线和景点中包含的地点至多出现一次，即不要重复浏览。
-如果用户有交通时间需求和交通距离需求，请确保路线的交通时间和交通距离和用户需求尽可能一致。
    '''
        return template.format(validation=validation, route=route, request=request)


if __name__ == "__main__":
    llm = LLMs()
    input = "请推荐一条亲子线路,多吃好吃的，希望交通时间在10min内,不要去西湖"
    pos_input, neg_input = llm.request_split(input)
    routes, pois = llm.query_databases(pos_input, neg_input)
    question = llm.question_format_prompt(input, routes, pois)
    response = llm.llm(question)
    path_plan = llm.path_plan(response)
    while True:
        question_validation = llm.validation_format_prompt(path_plan, response, input)
        # print(question_validation)
        response_validation = llm.llm(question_validation)
        print(response_validation)
        match = re.search(r'(\{.*\})', response_validation, re.DOTALL)
        if match:
            extracted_json = match.group(1)
            data = json.loads(extracted_json)
        else:
            print("output wrong json")
            break
        print(data["change"])
        if not data["change"]:
            print("输出无需校验")
            break
        path_plan = llm.path_plan(response_validation)
        response = response_validation

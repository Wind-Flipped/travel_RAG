from zhipuai import ZhipuAI
from embedding import Zhipuembedding
# from data_chunker import ReadFile
# from databases import Vectordatabase
from prompts import zeroshot_react_agent_prompt, zeroshot_react_agent_prompt_zh, zeroshot_react_agent_prompt_reformat_zh
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))

from rag.component.embedding import Zhipuembedding
from rag.component.data_chunker import ReadFile
from rag.component.databases import Vectordatabase
from rag.component.request import Request

class LLMs:
    def __init__(self, model_name: str = 'glm-4-air', temperature: float = 0.9,
                 rag_database: str = "/home/wangb/cyo/graduation/rag/databases/hangzhou") -> None:
        # 初始化大模型
        self.model_name = model_name
        self.temperature = temperature
        self.model = ZhipuAI(api_key="c006413c47710730c9d9196b57c9ce81.CdBwOpUKukSkBHZG")

        # 加载向量数据库，embedding模型
        # self.db = Vectordatabase()
        # self.db.load_vector(rag_database)
        # self.embedding_model = Zhipuembedding()

    # 定义chat方法
    def __call__(self, prompt: str, stop: list = None):
        response = self.model.chat.completions.create(
            model=self.model_name,  # 填写需要调用的模型名称
            stop = stop,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        return response.choices[0].message.content

    def chat(self, pos_question: str, neg_question: str, answer_type: str = "explain"):
        info = self.db.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True)
        # for tmp in info:
        #     print(tmp)
        #     print('----------------------------')

        if answer_type == "explain":
            prompt = format_prompt_explain(pos_question + neg_question, info)
        else:
            prompt = format_prompt(pos_question + neg_question, info)
        # print(prompt)

        response = self.model.chat.completions.create(
            model= self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content

    def get_model(self):
        return self.model


def format_prompt_explain(question: str, info: str) -> str:
    # 这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
    template = """使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文是多条由关键词，路线和帖子组成的数据：
        ···
        {info}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
    format_info = "\n".join(f"第{index + 1}条数据: {item}" for index, item in enumerate(info))
    return template.format(question=question, info=format_info)


def format_prompt(question: str, info: str) -> str:
    # 这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
    template = """使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文是多条由关键词，路线和帖子组成的数据：
        ···
        {info}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        答案以一个地点列表给出，用逗号分开，不要输出多余内容。
        有用的回答:"""
    format_info = "\n".join(f"第{index + 1}条数据: {item}" for index, item in enumerate(info))
    return template.format(question=question, info=format_info)

class VectorDatabase:
    def __init__(self, model = ZhipuAI(api_key="c006413c47710730c9d9196b57c9ce81.CdBwOpUKukSkBHZG"),
                 model_name = "glm-4-air",
                 rag_database: list[str] = ["/home/wangb/cyo/graduation/rag/databases/hangzhou",
                                            "/home/wangb/cyo/graduation/rag/databases/hangzhou_poi"]) -> None:
        # Load vector database and embedding model
        self.db_route = Vectordatabase()
        self.db_route.load_vector(rag_database[0])
        self.db_poi = Vectordatabase()
        self.db_poi.load_vector(rag_database[1])
        self.embedding_model = Zhipuembedding()
        self.model_name = model_name
        self.request_split = Request(model=model, model_name="glm-4-air")

    def get_related_route_info(self, query: str):
        # Use Request to divide query into pos_question and neg_question)
        pos_question, neg_question = self.request_split.extract_requests(query)
        routes, pois = self.query_databases(pos_question, neg_question)
        return self.query_zh(routes, pois)

    def query_route(self, pos_question, neg_question):
        info = self.db_route.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True)
        return info

    def query_poi(self, pos_question, neg_question):
        info = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True)
        return info

    def query_databases(self, pos_question, neg_question):
        routes = self.db_route.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True)
        pois = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True)

        return routes, pois
    def query_zh(self, route_info, poi_info):
        format_route_info = "\n".join(f"-第{index + 1}条路线攻略: {item}" for index, item in enumerate(route_info))
        format_poi_info = "\n".join(f"-第{index + 1}条景点描述: {item}" for index, item in enumerate(poi_info))

        return format_route_info, format_poi_info

class ReactAgent:
    def __init__(self,
                 mode: str = 'zero_shot_reformat_zh',
                 tools = None,
                 max_steps: int = 10,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name='glm-4-plus',
                 planner_llm_name='glm-4-plus',
                 query = ''):
        self.answer = ''
        self.mode = mode
        if self.mode == 'zero_shot':
            self.agent_prompt = zeroshot_react_agent_prompt
        elif self.mode == 'zero_shot_zh':
            self.agent_prompt = zeroshot_react_agent_prompt_zh
        elif self.mode == 'zero_shot_reformat_zh':
            self.agent_prompt = zeroshot_react_agent_prompt_reformat_zh
        self.query = query
        self.scratchpad = ''
        self.llm = LLMs(rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")
        self.vector_database = VectorDatabase()
        self.route_info, self.poi_info = self.vector_database.get_related_route_info(self.query)
    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad)
        elif self.mode == 'zero_shot_reformat_zh':
            return self.agent_prompt.format(
                query=self.query,
                route_info=self.route_info,
                scratchpad=self.scratchpad)

    def prompt_agent(self) -> str:
        prompt = self._build_agent_prompt()
        print("-------")
        print(prompt)
        print("-------")
        request = format_step(self.llm(prompt))
        print("-------")
        print(request)
        print("-------")
        return request

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

if __name__ == "__main__":
    llm = LLMs(rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")
    pos_input = "请推荐一条亲子线路"
    neg_input = "不要去西湖"
    input = "Please recommend a one-day tour in Hangzhou with a budget of less than 200 yuan."
    input = "请推荐预算在200元以下的杭州一日游。"
    # response = llm.chat(pos_input, neg_input, "explain")
    agent = ReactAgent(query=input)
    response = agent.prompt_agent()
    print('-' * 100)
    print(response)


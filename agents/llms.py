from zhipuai import ZhipuAI
import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
from prompts import zeroshot_react_agent_prompt, zeroshot_react_agent_prompt_zh, zeroshot_react_agent_prompt_reformat_zh



from rag.component.embedding import Zhipuembedding
from rag.component.data_chunker import ReadFile
from rag.component.databases import Vectordatabase
from rag.component.request import Request

class LLMs:
    def __init__(self, model_name: str = 'glm-4-air', temperature: float = 0.9,
                 api_key = "c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE",
                 rag_database: str = "/home/wangb/cyo/graduation/rag/databases/hangzhou") -> None:
        # 初始化大模型
        self.model_name = model_name
        self.temperature = temperature
        self.model = ZhipuAI(api_key=api_key)

        # 加载向量数据库，embedding模型
        # self.db = Vectordatabase(api_key=api_key)
        # self.db.load_vector(rag_database)
        # self.embedding_model = Zhipuembedding()

    # 定义chat方法
    def __call__(self, prompt: str, stop: list = None):
        response = self.model.chat.completions.create(
            model=self.model_name,
            stop = stop,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        return response.choices[0].message.content


    def get_model(self):
        return self.model


class VectorDatabase:
    def __init__(self, model = ZhipuAI(api_key="c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE"),
                 index=-1,
                 model_name = "glm-4-air",
                 rag_database: list[str] = ["/home/wangb/cyo/graduation/rag/databases/hangzhou",
                                            "/home/wangb/cyo/graduation/rag/databases/hangzhou_poi"]) -> None:
        # Load vector database and embedding model
        self.db_route = Vectordatabase()
        self.db_route.load_vector(rag_database[0])
        self.db_poi = Vectordatabase()
        self.db_poi.load_vector(rag_database[1])
        self.embedding_model = Zhipuembedding(api_key="c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE")
        self.model_name = model_name
        self.request_split = Request(model=model, model_name="glm-4-air")

    def run(self, pos_question, number = 3, index=-1):
        return self.db_poi.query_both(pos_question, None, self.embedding_model, number, 1, True, invisible=index)

    def test_query(self, query: str, index=-1):
        try:
            pos_question, neg_question = self.request_split.extract_requests(query)
            print("pos_question")
            print(pos_question)
            print("neg_question")
            print(neg_question)
        except:
            pos_question, neg_question = query, None
        # pos_question, neg_question = query, None
        print("__________++++++++++++++++_______________")
        routes = self.db_route.query_both(pos_question, neg_question, self.embedding_model, 4, 1, True, invisible=index)

        pois = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, 4, 1,  True, invisible=index)
        return self.query_zh(routes, pois)
    def get_related_route_info(self, query: str, index=-1):
        # Use Request to divide query into pos_question and neg_question)
        try:
            pos_question, neg_question = self.request_split.extract_requests(query)
        except:
            pos_question, neg_question = query, None
        routes, pois = self.query_databases(pos_question, neg_question, index=index)
        return self.query_zh(routes, pois)

    def query_route(self, pos_question, neg_question, index=-1):
        info = self.db_route.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True, invisible=index)
        return info

    def query_poi(self, pos_question, neg_question, index=-1):
        info = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, 3, 3, True, invisible=index)
        return info

    def query_databases(self, pos_question, neg_question, index=-1):
        routes = self.db_route.query_both(pos_question, neg_question, self.embedding_model, 2, 2, True, invisible=index)
        pois = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, 2, 2, True, invisible=index)

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


from zhipuai import ZhipuAI
import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./agents")))
from agents.prompts import zeroshot_react_agent_prompt, zeroshot_react_agent_prompt_zh, zeroshot_react_agent_prompt_reformat_zh
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from agents.rag.component.embedding import Zhipuembedding
from agents.rag.component.data_chunker import ReadFile
from agents.rag.component.databases import Vectordatabase
from agents.rag.component.request import Request

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

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
        elif 'Qwen3' in model_name:
            print("Using Qwen3")
            model_path = "/home/wangb/lyq/rl/X-R1/Qwen3-8B/Qwen/Qwen3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )


        self.prompt_token = 0
        self.completion_token = 0

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][
                       len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

    # 定义chat方法
    def __call__(self, prompt: str, stop: list = None):
        if "Qwen3" in self.model_name:
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return content
        else:
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

class VectorDatabase:
    def __init__(self, model = ZhipuAI(api_key="c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE"),
                 index=-1,
                 model_name = "glm-4-air",
                 rag_database: list[str] = ["rag/databases/hangzhou",
                                            "rag/databases/hangzhou_poi",
                                            "rag/databases/hangzhou/key_place2_requests.json"]) -> None:
        # Load vector database and embedding model
        self.db_route = Vectordatabase()
        self.db_route.load_vector(rag_database[0])
        self.db_route.load_real_route(rag_database[2])
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
        routes = self.db_route.query_both(pos_question, neg_question, self.embedding_model, 4, 1, True, invisible=index)

        pois = self.db_poi.query_both(pos_question, neg_question, self.embedding_model, 4, 1,  True, invisible=index)
        return self.query_zh(routes, pois)

    def get_route_info_with_place(self, query: str, place: str, index=-1):
        routes = self.db_route.query_with_route(query, place, self.embedding_model, 2, 2, True, invisible=index)
        pois = self.db_poi.query_both(query, None, self.embedding_model, 2, 1, True, invisible=index)
        return self.query_zh(routes, pois)

    def get_route_info_with_bm25(self, query: str, place: str, index=-1):
        routes = self.db_route.query_with_route_bm25(query, place, self.embedding_model, 2, 4, invisible=index)
        pois = self.db_poi.query_both(query, None, self.embedding_model, 2, 1, True, invisible=index)
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


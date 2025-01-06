import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
from agents.prompts import planner_agent_prompt, planner_agent_prompt_zh, planner_zero_shot_zh, planner_reflection_zh
# from env import ReactEnv,ReactReflectEnv

import re
import time
from enum import Enum
from typing import List, Union, Literal
import argparse
# from zhipuai import ZhipuAI
from agents.llms import LLMs

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# class LLMs:
#     def __init__(self, model_name: str = 'glm-4-air', temperature: float = 0.9,
#                  api_key = "c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE",
#                  rag_database: str = "/home/wangb/cyo/graduation/rag/databases/hangzhou") -> None:
#         # 初始化大模型
#         self.model_name = model_name
#         self.temperature = temperature
#         self.model = ZhipuAI(api_key=api_key)
#
#         # 加载向量数据库，embedding模型
#         # self.db = Vectordatabase(api_key=api_key)
#         # self.db.load_vector(rag_database)
#         # self.embedding_model = Zhipuembedding()
#
#     # 定义chat方法
#     def __call__(self, prompt: str, stop: list = None):
#         response = self.model.chat.completions.create(
#             model=self.model_name,
#             stop = stop,
#             messages=[
#                 {"role": "user", "content": prompt},
#             ]
#         )
#
#         return response.choices[0].message.content
#
#
#     def get_model(self):
#         return self.model
class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 # args,
                 model_name: str = 'glm-4-air',
                 mode = 'zero_shot_reformat_zh',
                 ) -> None:
        self.mode = mode
        if mode == 'our':
            mode = "zero_shot_zh"
            self.mode = mode
        if mode == 'zero_shot_reformat_zh' or mode == 'route_RAG_zh' or mode == 'route_bm25_RAG_zh':
            self.agent_prompt = planner_agent_prompt_zh
        elif mode == "zero_shot_zh":
            self.agent_prompt = planner_zero_shot_zh
        elif mode == "reflection_zh":
            self.agent_prompt = planner_reflection_zh

        self.scratchpad: str = ''
        self.model_name = model_name


        if 'glm-4' in model_name or "deepseek-chat" in model_name:
            self.llm = LLMs(model_name=self.model_name, rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")
        else:
            print("LLM's name is getting wrong")
            self.llm = LLMs(model_name=self.model_name, rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")

        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, route=None) -> str:
        # print(self._build_agent_prompt(text, query))
        if self.mode == 'zero_shot_reformat_zh' or self.mode == 'route_RAG_zh' or self.mode == 'route_bm25_RAG_zh':
            return str(self.llm(self._build_agent_prompt(text, query, route)))
        elif self.mode == "zero_shot_zh" or self.mode == "reflection_zh":
            return str(self.llm(self._build_agent_prompt(text, query, route)))

    def _build_agent_prompt(self, text, query, route) -> str:
        if self.mode == 'zero_shot_reformat_zh' or self.mode == 'route_RAG_zh' or self.mode == 'route_bm25_RAG_zh':
            return self.agent_prompt.format(
                text=text,
                query=query,
                route=route)
        elif self.mode == "zero_shot_zh" or self.mode == "reflection_zh":
            return self.agent_prompt.format(
                text=text,
                query=query)

'''
class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation"]})
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
                reflect_prompt: PromptTemplate = reflect_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        if model_name in ['gemini']:
            self.react_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
            self.reflect_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        else:
            self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
            self.reflect_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_agent(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.react_llm.invoke(self._build_agent_prompt()).content)
                else:
                    return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def prompt_reflection(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.reflect_llm.invoke(self._build_reflection_prompt()).content)
                else:
                    return format_step(self.reflect_llm([HumanMessage(content=self._build_reflection_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_reflection_prompt())
                print(len(self.enc.encode(self._build_reflection_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad,
                            reflections = self.reflections_str)
    
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ''
        self.env.reset()

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
'''
if __name__ == '__main__':
    planner = Planner(model_name='glm-4-air', mode='reflection_zh')
    text = "杭州有西湖、灵隐寺、千岛湖等景点，绿茶餐厅人均50元，肯德基人均40元，海底捞火锅人均100元，巴西烤肉人均70元，胜发饺子人均30元，西湖到灵隐寺的距离为6km，灵隐寺到千岛湖的距离为7km"
    query = "我想去杭州一日游，想要去西湖，总共有4个人，预算480元，交通距离不超过12km"
    result = planner.run(text, query)
    print(result)

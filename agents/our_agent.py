import re, string, os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
from typing import List, Dict, Any
from pandas import DataFrame
from our_prompt import planner_instruction, solver_instruction, trimmer_instruction, evaluator_instruction
import json
from tqdm import tqdm
import argparse
import os
from llms import LLMs, VectorDatabase
import tools.apis as apis
from tools.notebook.apis import Notebook
from evaluate import Evaluator
import time

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


# pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

actionMapping = {'RestaurantInfo':'restaurant_info', 'AttractionInfo':'attraction_info',
                 'RestaurantDistance':'restaurant_distance', 'AttractionDistance':'attraction_distance',
                 'RestaurantSearch': 'restaurant_search', 'AttractionSearch':'attraction_search',
                 'Notebook':'notebook','Planner':'planner', 'AttractionRetrieval': 'attraction_retrieval',
                 'Finish':'finish'}

class Planner:
    def __init__(self,
                 mode: str = 'zero_shot_reformat_zh',
                 llm_name='glm-4-air'
                 ):
        self.mode = mode
        self.llm_name = llm_name
        self.llm = LLMs(model_name=llm_name)
        self.result = ''
        if mode == 'our':
            self.agent_prompt = planner_instruction

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.query = ''

    def run(self, query, reset=True):
        if reset:
            self.__reset_agent()
        self.query = query
        result = self.prompt_agent()

        return result

    def prompt_agent(self) -> str:
        while True:
            try:
                # print(self._build_agent_prompt())
                if 'glm-4' in self.llm_name:
                    request = format_step(self.llm(self._build_agent_prompt()))
                else:
                    # request = format_step(self.llm([HumanMessage(content=self._build_agent_prompt())]).content)
                    request = " "
                # print(request)
                return request
            except:
                print("Error !")
                return "Error !"

    def _build_agent_prompt(self) -> str:
        if self.mode == 'our':
            return self.agent_prompt.format(
                query=self.query)

class Trimmer():
    def __init__(self,
                 mode: str = 'zero_shot_reformat_zh',
                 llm_name='glm-4-air',
                 solver_agent=None,
                 max_step=3
                 ):
        self.mode = mode
        self.llm_name = llm_name
        self.llm = LLMs(model_name=llm_name)
        self.result = ''
        self.finished = False
        self.json_log = [{}]
        self.solver = solver_agent
        self.step_n = 1
        self.max_step = max_step
        if mode == 'our':
            self.agent_prompt = trimmer_instruction

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.query = ''
        self.result = ''
        self.json_log = [{}]

    def run(self, query, tasks: list, plans: list, reset=True):
        if reset:
            self.__reset_agent()
        self.query = query
        self.tasks = tasks
        self.plans = plans
        self.task_str = ''
        self.plan_str = ''
        for task in tasks:
            self.task_str += task + '\n'
        for plan in plans:
            self.plan_str += plan + '\n'

        result = {}
        while not self.finished and self.step_n <= self.max_step:
            result = self.step()

        return result, self.json_log

    def step(self):
        result = self.prompt_agent()
        match = re.search(r'(\{.*\})', result, re.DOTALL)
        if match:
            extracted_json = match.group(1)
            data = json.loads(extracted_json)
            tag = data["评价"]
            if tag == 'Success':
                self.finished = True
                print(result)
                return result
            else:
                sub_task = data["查询"]
                print(sub_task)
                self.tasks.append(sub_task)
                solver_result, scratchpad, action_log = self.solver.run(query=sub_task)
                self.json_log[-1][f'{args.model_name}_subtask_{self.step_n}'] = sub_task
                self.json_log[-1][f'{args.model_name}_results_logs_{self.step_n}'] = scratchpad
                self.json_log[-1][f'{args.model_name}_results_{self.step_n}'] = solver_result
                self.json_log[-1][f'{args.model_name}_action_logs_{self.step_n}'] = action_log
                self.json_log.append({})
                self.step_n += 1
                return {}



    def prompt_agent(self) -> str:
        while True:
            try:
                # print(self._build_agent_prompt())
                if 'glm-4' in self.llm_name:
                    request = format_step(self.llm(self._build_agent_prompt()))
                else:
                    # request = format_step(self.llm([HumanMessage(content=self._build_agent_prompt())]).content)
                    request = " "
                # print(request)
                return request
            except:
                print("Error !")
                return "Error !"

    def _build_agent_prompt(self) -> str:
        if self.mode == 'our':
            return self.agent_prompt.format(
                query=self.query,
                task=self.task_str,
                plan=self.plan_str)

class Evaluator():
    def __init__(self,
                 mode: str = 'our',
                 llm_name="deepseek-chat"):
        self.mode = mode
        self.llm_name = llm_name
        if 'glm-4' in llm_name:
            self.llm = LLMs(model_name=llm_name, rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")
        elif 'deepseek' in llm_name:
            self.llm = LLMs(model_name=llm_name, api_key='sk-a0750ae6f78a4ddfb648d18e65b20ce0' ,rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")

        if mode == 'our':
            self.agent_prompt = evaluator_instruction

        self.json_log = [{}]
        self.thought = ''
        self.action = ''
        self.observation = ''

    def run(self, query, thought, action, observation, reset=True):
        if reset:
            self.__reset_agent()
        self.query = query
        self.thought = thought
        self.action = action
        self.observation = observation
        result = self.prompt_agent()
        match = re.search(r'(\{.*\})', result, re.DOTALL)
        try:
            if match:
                extracted_json = match.group(1)
                data = json.loads(extracted_json)
                analysis = data["分析"]
                tag = data["评价"]
                summary = data["总结"]
                print(f"Analysis: {analysis}")
                print(f"Tag: {tag}")
                print(f"Summary: {summary}")
                self.json_log[-1]['analysis'] = analysis
                self.json_log[-1]['tag'] = tag
                self.json_log[-1]['summary'] = summary
                self.json_log.append({})
            else:
                summary = observation
        except Exception as e:
            summary = observation
            print(e)
            print("json evaluator is getting wrong")
        return summary

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.query = ''
        self.result = ''
        self.thought = ''
        self.action = ''
        self.observation = ''
        self.json_log = [{}]


    def prompt_agent(self) -> str:
        while True:
            try:
                # print(self._build_agent_prompt())
                if 'glm-4' in self.llm_name or 'deepseek' in self.llm_name:
                    request = format_step(self.llm(self._build_agent_prompt()))
                else:
                    # request = format_step(self.llm([HumanMessage(content=self._build_agent_prompt())]).content)
                    request = " "
                # print(request)
                return request
            except:
                print("Error !")
                return "Error !"

    def _build_agent_prompt(self) -> str:
        if self.mode == 'our':
            return self.agent_prompt.format(
                query=self.query,
                thought=self.thought,
                action=self.action,
                observation=self.observation)

    def get_logs(self):
        return self.json_log

class Solver:
    def __init__(self,
                 mode: str = 'zero_shot_reformat_zh',
                 tools: List[str] = None,
                 max_steps: int = 10,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name='glm-4-air',
                 #  logs_path = '../logs/',
                 city_file_path='../database/background/citySet.txt'
                 ) -> None:

        self.answer = ''
        self.max_steps = max_steps
        self.mode = mode

        if 'glm-4' in react_llm_name:
            self.llm = LLMs(model_name= react_llm_name, rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")
        else:
            print("LLM's name is getting wrong")
            self.llm = LLMs(model_name= react_llm_name, rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")

        if self.mode == 'our':
            self.agent_prompt = solver_instruction


        self.vector_database = VectorDatabase(model=self.llm.get_model(), model_name=react_llm_name)
        self.react_name = react_llm_name
        self.json_log = []

        self.current_observation = ''
        self.current_data = None
        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        # print(self.retry_record)

        self.last_actions = []
        self.action_info = ''

        # self.log_path = logs_path + datetime.now().strftime('%Y%m%d%H%M%S') + '.out'
        # self.log_file = open(self.log_path, 'a+')

        # print("logs will be stored in " + self.log_path)

        # self.city_set = self.load_city(city_set_path=city_file_path)

        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.__reset_agent()

    def run(self, query, reset=True) -> None:

        self.query = query

        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:

        self.json_log.append({"step": self.step_n, "thought": "",
                              "action": "", "observation": "", "state": ""})

        # self.scratchpad += f'\nThought {self.step_n}:'
        generate_text = self.prompt_agent()
        try:
            thought = re.search(r'Thought[\s\d]*: ([^A。.]+)', generate_text).group(1)
            action = re.search(r"Action[\s\d]*: (.*?\[.*?\])", generate_text).group(1)
        except:
            self.retry_record['invalidAction'] += 1
            print(f'Invalid Thought or Action: {generate_text}')
            self.json_log[-1]["Thought"] = generate_text
            self.current_observation = '您生成了一条非法指令，请检查您的指令是否正确，需要重新以“Thought”、“Action”的指令格式生成接下来的规划与行动，已经搜集完足够信息后，请调用Finish工具返回最终的路线规划。'
            self.scratchpad += self.current_observation
            self.json_log[-1]['state'] = f'invalidAction'
            self.step_n += 1
            return
        self.scratchpad += f'\nThought {self.step_n}: {thought}. Action {self.step_n}: {action}'

        print(f"===============scratchpad {self.step_n}===================")


        print(self.scratchpad.split('\n')[-1])
        # self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'Thought {self.step_n}:', "")
        self.json_log[-1]['thought'] = thought
        # self.log_file.write(self.scratchpad.split('\n')[-1] + '\n')

        # Act
        # self.scratchpad += f'Action {self.step_n}:'
        # action = self.prompt_agent()
        print(self.step_n)

        if action == None or action == '' or action == '\n':
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        # else:
        #     self.scratchpad += ' ' + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        # self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:', "")
        self.json_log[-1]['action'] = action
        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            # self.log_file.write("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return

        # action_type, action_arg = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        # self.log_file.write(self.scratchpad.split('\n')[-1]+'\n')

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '

        if action == None or action == '' or action == '\n':
            action_type = None
            action_arg = None
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."

        else:
            action_type, action_arg = parse_action(action)

            if action_type != "Finish":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                else:
                    pending_action = 'invalidAction'

                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = "Finish"
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = "Finish"
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"invalidAction early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"invalidAction early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

                if action_type == 'RestaurantInfo':
                    try:
                        self.action_info = f"餐厅{action_arg}的信息"
                        self.current_data = self.tools[pending_action].run(action_arg)
                        self.current_observation = f"餐厅{action_arg}的信息为{self.current_data}"
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                    except Exception as e:
                        print(e)
                        self.retry_record[pending_action] += 1
                        self.current_observation = f'Illegal Restaurant info Search. Please try again.'
                        self.scratchpad += f'Illegal Restaurant info Search. Please try again.'
                        self.json_log[-1]['state'] = f'Illegal args. Other Error'

                elif action_type == 'RestaurantDistance':
                    try:
                        self.action_info = f"餐厅{action_arg.split(', ')[0]}到餐厅{action_arg.split(', ')[1]}的距离（千米）"
                        self.current_data = self.tools[pending_action].run_for_distance(action_arg.split(', ')[0], action_arg.split(', ')[1])
                        self.current_data = f"餐厅{action_arg.split(', ')[0]}到餐厅{action_arg.split(', ')[1]}的距离为{self.current_data}千米"
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                    except Exception as e:
                        print(e)
                        self.retry_record[pending_action] += 1
                        self.current_observation = f'Illegal Attraction Distance Search. Please try again.'
                        self.scratchpad += f'Illegal Attraction Distance Search. Please try again.'
                        self.json_log[-1]['state'] = f'Illegal args. Other Error'

                elif action_type == 'RestaurantSearch':
                    try:
                        self.action_info = f"在经纬度（{action_arg.split(', ')[0]}, {action_arg.split(', ')[1]}）附近的{action_arg.split(', ')[2]}家餐厅的信息"
                        self.current_data = self.tools[pending_action].get_nearest_restaurants(
                            float(action_arg.split(', ')[0]), float(action_arg.split(', ')[1]), int(action_arg.split(', ')[2]))
                        self.current_observation =f"在经纬度（{action_arg.split(', ')[0]}, {action_arg.split(', ')[1]}）附近的{action_arg.split(', ')[2]}家餐厅的信息为：{self.current_data}"
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                    except Exception as e:
                        print(e)
                        self.retry_record[pending_action] += 1
                        self.current_observation = f'Illegal Restaurant Search. Please try again.'
                        self.scratchpad += f'Illegal Restaurant Search. Please try again.'
                        self.json_log[-1]['state'] = f'Illegal args. Other Error'

                elif action_type == 'AttractionSearch':
                    try:
                        self.action_info = f"在经纬度（{action_arg.split(', ')[0]}, {action_arg.split(', ')[1]}）附近的{action_arg.split(', ')[2]}家景点的信息"
                        self.current_data = self.tools[pending_action].get_nearest_restaurants(
                            float(action_arg.split(', ')[0]), float(action_arg.split(', ')[1]), int(action_arg.split(', ')[2]))
                        self.current_observation = f"在经纬度（{action_arg.split(', ')[0]}, {action_arg.split(', ')[1]}）附近的{action_arg.split(', ')[2]}家景点的信息为{self.current_data}"
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                    except Exception as e:
                        print(e)
                        self.retry_record[pending_action] += 1
                        self.current_observation = f'Illegal Attraction Search. Please try again.'
                        self.scratchpad += f'Illegal Attraction Search. Please try again.'
                        self.json_log[-1]['state'] = f'Illegal args. Other Error'


                elif action_type == 'AttractionInfo':
                    try:
                        self.action_info = f"景点{action_arg}的信息"
                        self.current_data = self.tools[pending_action].run(action_arg)
                        self.current_observation = f"景点{action_arg}的信息为{self.current_data}"
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                    except Exception as e:
                        print(e)
                        self.retry_record[pending_action] += 1
                        self.current_observation = f'Illegal Attraction info Search. Please try again.'
                        self.scratchpad += f'Illegal Attraction info Search. Please try again.'
                        self.json_log[-1]['state'] = f'Illegal args. Other Error'

                elif action_type == 'AttractionDistance':
                    try:
                        self.action_info = f"景点{action_arg.split(', ')[0]}到景点{action_arg.split(', ')[1]}的距离（千米）"
                        self.current_data = self.tools[pending_action].run_for_distance(action_arg.split(', ')[0], action_arg.split(', ')[1])
                        self.current_data = f"景点{action_arg.split(', ')[0]}到景点{action_arg.split(', ')[1]}的距离为{self.current_data}千米"
                        self.current_observation = to_string(self.current_data).strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                    except Exception as e:
                        print(e)
                        self.retry_record[pending_action] += 1
                        self.current_observation = f'Illegal Attraction Distance Search. Please try again.'
                        self.scratchpad += f'Illegal Attraction Distance Search. Please try again.'
                        self.json_log[-1]['state'] = f'Illegal args. Other Error'

                elif action_type == 'AttractionRetrieval':
                    try:
                        self.action_info = f"查找到与{action_arg.split(', ')[0]}类型相似的景点信息"
                        self.current_data = self.tools[pending_action].run(action_arg.split(', ')[0], int(action_arg.split(', ')[1]))
                        self.current_data = f"查找到与{action_arg.split(', ')[0]}类型相似的景点信息为{self.current_data}"
                        self.current_observation = to_string(self.current_data).strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                    except Exception as e:
                        print(e)
                        self.retry_record[pending_action] += 1
                        self.current_observation = f'Illegal Attraction Retrieval Search. Please try again.'
                        self.scratchpad += f'Illegal Attraction Retrieval Search. Please try again.'
                        self.json_log[-1]['state'] = f'Illegal args. Other Error'

                try:
                    # store observation to notebook
                    self.tools['notebook'].write(self.current_data, self.action_info)
                    self.__reset_record()

                except Exception as e:
                    print(e)

            elif action_type == "Finish":
                # try:
                self.finished = True
                self.current_observation = action_arg
                self.answer = self.current_observation
                self.json_log[-1]['state'] = f'Successful'

            else:
                self.retry_record['invalidAction'] += 1
                print(f'Invalid Action: {action_type}---------------{action}')
                self.current_observation = 'Invalid Action.'
                self.scratchpad += self.current_observation
                self.json_log[-1]['state'] = f'invalidAction'

        if action == None or action == '' or action == '\n':
            print(f'Observation {self.step_n}: ' + "No feedback from the environment due to the null action.")
            # write(f'Observation {self.step_n}: ' + "Your action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again.")
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
        else:
            print(f'Observation {self.step_n}: ' + self.current_observation + '\n')
            # rite(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            self.json_log[-1]['observation'] = self.current_observation

        self.step_n += 1


        if action_type and action_type == "Finish":
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def prompt_agent(self) -> str:
        while True:
            try:
                # print(self._build_agent_prompt())
                if 'glm-4' in self.react_name:
                    request = format_step(self.llm(self._build_agent_prompt()))
                else:
                    # request = format_step(self.llm([HumanMessage(content=self._build_agent_prompt())]).content)
                    request = " "
                # print(request)
                return request
            except:

                print("Error !")
                return "Error !"

    def _build_agent_prompt(self) -> str:
        if self.mode == 'our':
            return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (self.step_n > self.max_steps) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad: str = ''
        self.__reset_record()
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []
        self.route_info = ''
        self.poi_info = ''
        self.action_info = ''

        if 'notebook' in self.tools:
            self.tools['notebook'].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0

    def load_tools(self, tools: List[str]) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            if tool_name == 'restaurant_info' or tool_name == 'restaurant_distance' or tool_name == 'restaurant_search':
                tools_map[tool_name] = apis.Restaurants()
            elif tool_name == 'attraction_info' or tool_name == 'attraction_distance' or tool_name == 'attraction_search':
                tools_map[tool_name] = apis.Attractions()
            elif tool_name == 'notebook':
                tools_map[tool_name] = Notebook()
            elif tool_name == 'attraction_retrieval':
                tools_map[tool_name] = self.vector_database
        return tools_map



def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    pattern = r'^([^\[]+)\[([^\]]+)\]'
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


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)


if __name__ == '__main__':

    tools_list = ['restaurant_info', 'attraction_info', 'restaurant_distance', 'attraction_distance',
                  'restaurant_search', 'attraction_search', 'notebook', 'planner', 'attraction_retrieval', 'finish']
    # model_name = ['gpt-3.5-turbo-1106','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral','ChatGLM3-6B-32K'][2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="glm-4-air")
    parser.add_argument("--output_dir", type=str, default="./logs")
    parser.add_argument("--dataset", type=str, default="fake")
    parser.add_argument("--mode", type=str, default='our')
    args = parser.parse_args()
    print(args)

    planner_agent = Planner(mode=args.mode, llm_name=args.model_name)
    solver_agent = Solver(mode=args.mode, tools=tools_list, max_steps=6, react_llm_name=args.model_name)
    trimmer_agent = Trimmer(mode=args.mode, llm_name=args.model_name, solver_agent=solver_agent, max_step=3)

    start_time = time.time()
    if True:
        evaluator = Evaluator()
        with open(f"data/base_request.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        step = 1
        for data in tqdm(all_data):
            if step > 1:
                break
            query = data["query"]
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}'))
            result = [{}]
            result[-1][f'{args.model_name}_query_{step}'] = query
            plan_info = planner_agent.run(query)
            print(plan_info)
            match = re.search(r'(\{.*\})', plan_info, re.DOTALL)
            tasks = []
            solver_results = []
            if match:
                extracted_json = match.group(1)
                json_data = json.loads(extracted_json)
                plan_number = json_data["子任务数量"]
                for i in range(plan_number):
                    task = json_data[f"子任务 {i + 1}"]
                    tasks.append(task)
                    solver_result, scratchpad, action_log = solver_agent.run(query=task)
                    solver_results.append(solver_result)

                    result[-1][f'{args.model_name}_subtask_{i + 1}'] = task
                    result[-1][f'{args.model_name}_results_logs_{i + 1}'] = scratchpad
                    result[-1][f'{args.model_name}_results_{i + 1}'] = solver_result
                    result[-1][f'{args.model_name}_action_logs_{i + 1}'] = action_log

            trimmer_result, trimmer_log = trimmer_agent.run(query=query, tasks=tasks, plans=solver_results)
            result[-1][f'{args.model_name}_trimmer_logs_{step}'] = trimmer_log
            result[-1][f'{args.model_name}_trimmer_results_{step}'] = trimmer_result

            # write to json file
            with open(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}/generated_plan_{step}.json'), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            step += 1

            evaluator.evaluate(agent_output=trimmer_result, externel_data=data)

        evaluator.print_result(args.mode)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
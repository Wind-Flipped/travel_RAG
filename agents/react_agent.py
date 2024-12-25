import re, string, os, sys

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools")))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/notebook")))
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
from pandas import DataFrame
from prompts import zeroshot_react_agent_prompt, zeroshot_react_agent_prompt_zh, zeroshot_react_agent_prompt_reformat_zh
import sys
import json
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
import os
from llms import LLMs, VectorDatabase
import tools.apis as apis
from tools.notebook.apis import Notebook
from tools.planner.apis import Planner
from evaluate import Evaluator
import time

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

actionMapping = {'RestaurantInfo':'restaurant_info', 'AttractionInfo':'attraction_info',
                 'RestaurantDistance':'restaurant_distance', 'AttractionDistance':'attraction_distance',
                 'RestaurantSearch': 'restaurant_search', 'AttractionSearch':'attraction_search',
                 'Notebook':'notebook','Planner':'planner', 'AttractionRetrieval': 'attraction_retrieval'}

class ReactAgent:
    def __init__(self,
                 mode: str = 'zero_shot_reformat_zh',
                 tools: List[str] = None,
                 max_steps: int = 10,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name='glm-4-air',
                 planner_llm_name='glm-4-air',
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

        if self.mode == 'zero_shot':
            self.agent_prompt = zeroshot_react_agent_prompt
        elif self.mode == 'zero_shot_zh':
            self.agent_prompt = zeroshot_react_agent_prompt_zh
        elif self.mode == 'zero_shot_reformat_zh' or self.mode == 'route_RAG_zh':
            self.agent_prompt = zeroshot_react_agent_prompt_reformat_zh

        self.vector_database = VectorDatabase(model=self.llm.get_model(), model_name=react_llm_name)
        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

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

    def run(self, query, place=None, index=-1, reset=True) -> None:

        self.query = query

        if reset:
            self.__reset_agent()
        if self.mode == 'zero_shot_reformat_zh':
            self.route_info, self.poi_info = self.vector_database.get_related_route_info(self.query, index=index)
        elif self.mode == 'route_RAG_zh':
            match = re.search(r"要求是：(.*?)。", query)
            if match:
                query = match.group(1)
            self.route_info, self.poi_info = self.vector_database.get_route_info_with_place(query, place, index=index)

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:

        self.json_log.append({"step": self.step_n, "thought": "",
                              "action": "", "observation": "", "state": ""})

        # self.scratchpad += f'\nThought {self.step_n}:'
        generate_text = self.prompt_agent()
        try:
            thought = re.search(r'Thought [\d+]: ([^A。.]+)', generate_text).group(1)
            action = re.search(r"Action [\d+]: (.*?\[.*?\])", generate_text).group(1)
        except:
            self.retry_record['invalidAction'] += 1
            print(f'Invalid Thought or Action: {generate_text}')
            self.json_log[-1]["Thought"] = generate_text
            self.current_observation = '您生成了一条非法指令，请检查您的指令是否正确，需要重新以“Thought”、“Action”的指令格式生成接下来的规划与行动，已经搜集完足够信息后，请调用Planner工具得到最终的路线规划。'
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

            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                else:
                    pending_action = 'invalidAction'

                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = 'Planner'
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

            elif action_type == "Planner":
                # try:
                if self.mode == 'zero_shot_zh':
                    self.current_observation = str(
                        self.tools["planner"].run(self.scratchpad, query=action_arg, route=None))
                elif self.mode == 'zero_shot_reformat_zh' or self.mode == 'route_RAG_zh':
                    self.current_observation = str(
                        self.tools["planner"].run(self.scratchpad, query=action_arg, route=self.route_info))
                self.scratchpad += self.current_observation
                self.answer = self.current_observation
                self.__reset_record()
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


        if action_type and action_type == 'Planner' and self.retry_record['planner'] == 0:
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
        if self.mode == 'zero_shot_zh':
            return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad)
        elif self.mode == 'zero_shot_reformat_zh' or self.mode == 'route_RAG_zh':
            return self.agent_prompt.format(
                query=self.query,
                route_info=self.route_info,
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
            elif tool_name == 'planner':
                tools_map[tool_name] = Planner(self.planner_name, mode=self.mode)
            elif tool_name == 'attraction_retrieval':
                tools_map[tool_name] = self.vector_database
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, 'r').read().strip().split('\n')
        for unit in lines:
            city_set.append(unit)
        return city_set


### String Stuff ###
# gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


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

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parse_args_string(s: str) -> dict:
    # Split the string by commas
    segments = s.split(",")

    # Initialize an empty dictionary to store the results
    result = {}

    for segment in segments:
        # Check for various operators
        if "contains" in segment:
            if "~contains" in segment:
                key, value = segment.split("~contains")
                operator = "~contains"
            else:
                key, value = segment.split("contains")
                operator = "contains"
        elif "<=" in segment:
            key, value = segment.split("<=")
            operator = "<="
        elif ">=" in segment:
            key, value = segment.split(">=")
            operator = ">="
        elif "=" in segment:
            key, value = segment.split("=")
            operator = "="
        else:
            continue  # If no recognized operator is found, skip to the next segment

        # Strip spaces and single quotes
        key = key.strip()
        value = value.strip().strip("'")

        # Store the result with the operator included
        result[key] = (operator, value)

    return result


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
                  'restaurant_search', 'attraction_search', 'notebook', 'planner', 'attraction_retrieval']
    # model_name = ['gpt-3.5-turbo-1106','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral','ChatGLM3-6B-32K'][2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="glm-4-air")
    parser.add_argument("--output_dir", type=str, default="./logs")
    parser.add_argument("--dataset", type=str, default="real")
    parser.add_argument("--mode", type=str, default='route_RAG_zh')
    args = parser.parse_args()
    agent = ReactAgent(mode=args.mode, tools=tools_list, max_steps=10, react_llm_name=args.model_name,
                       planner_llm_name=args.model_name)
    start_time = time.time()

    if args.mode == 'route_RAG_zh':
        evaluator = Evaluator(have_truth=True)
        with open(f"/home/wangb/cyo/graduation/rag/databases/hangzhou/key_place2_requests.json", 'r',
                  encoding='utf-8') as f:
            all_data = json.load(f)
        requires = all_data
        print("The total number: " + str(len(requires)))

        for index, item in tqdm(enumerate(requires)):
            query = item["input"]
            target_place = item["target_place"][0]
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}'))
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}/generated_plan_{index}.json')):
                result = [{}]
            else:
                result = json.load(
                    open(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}/generated_plan_{index}.json')))
            try:
                planner_results, scratchpad, action_log = agent.run(query=query, place=target_place, index=index)

                if planner_results == 'Max Token Length Exceeded.':
                    result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
                    result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
                    action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
                    result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
                else:
                    result[-1][f'{args.model_name}_two-stage_query'] = query
                    result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
                    result[-1][f'{args.model_name}_two-stage_results'] = planner_results
                    result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

                # write to json file
                with open(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}/generated_plan_{index}.json'), 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print("Error in results")
                print(e)
                planner_results = "Error in results"
                result[-1][f'{args.model_name}_two-stage_query'] = query
                result[-1][f'{args.model_name}_two-stage_results_logs'] = None
                result[-1][f'{args.model_name}_two-stage_results'] = planner_results
                result[-1][f'{args.model_name}_two-stage_action_logs'] = None
            evaluator.evaluate_real(agent_output=planner_results, target_place=item["target_place"],
                                    query=query, truth=item["route"])

        evaluator.print_real_result(args.mode)



    elif args.dataset == 'real':
        evaluator = Evaluator(have_truth=True)
        with open(f"/home/wangb/cyo/graduation/rag/databases/hangzhou/key_place2_requests.json", 'r',
                  encoding='utf-8') as f:
            all_data = json.load(f)
        requires = all_data
        print("The total number: " + str(len(requires)))

        for index, item in tqdm(enumerate(requires)):
            query = item["ai_input"]
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}'))
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}/generated_plan_{index}.json')):
                result = [{}]
            else:
                result = json.load(
                    open(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}/generated_plan_{index}.json')))
            try:
                planner_results, scratchpad, action_log = agent.run(query=query, index=index)

                if planner_results == 'Max Token Length Exceeded.':
                    result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
                    result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
                    action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
                    result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
                else:
                    result[-1][f'{args.model_name}_two-stage_query'] = query
                    result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
                    result[-1][f'{args.model_name}_two-stage_results'] = planner_results
                    result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

                # write to json file
                with open(os.path.join(f'{args.output_dir}/{args.dataset}/{args.mode}/generated_plan_{index}.json'), 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
            except:
                planner_results = "Error in results"
                result[-1][f'{args.model_name}_two-stage_query'] = query
                result[-1][f'{args.model_name}_two-stage_results_logs'] = None
                result[-1][f'{args.model_name}_two-stage_results'] = planner_results
                result[-1][f'{args.model_name}_two-stage_action_logs'] = None
            evaluator.evaluate_real(agent_output=planner_results, target_place=item["target_place"],
                                    query=query, truth=item["route"])

        evaluator.print_real_result(args.mode)

    else:
        number = 1
        evaluator = Evaluator()
        queries = evaluator.generate_request(number=number)
        step = 1
        for query in tqdm(queries):
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}'))
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}/generated_plan_{number}.json')):
                result = [{}]
            else:
                result = json.load(
                    open(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}/generated_plan_{number}.json')))


            planner_results, scratchpad, action_log = agent.run(query)


            if planner_results == 'Max Token Length Exceeded.':
                result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
                result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
                action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
                result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
            else:
                result[-1][f'{args.model_name}_two-stage_query'] = query
                result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
                result[-1][f'{args.model_name}_two-stage_results'] = planner_results
                result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

            # write to json file
            with open(os.path.join(f'{args.output_dir}/{args.set_type}/{args.mode}/generated_plan_{step}.json'), 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            step += 1

            evaluator.evaluate(agent_output=planner_results)

        evaluator.print_result()
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
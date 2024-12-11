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
from prompts import zeroshot_react_agent_prompt, zeroshot_react_agent_prompt_zh
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

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

actionMapping = {'restaurant_info':'restaurant_info', 'attraction_info':'attraction_info',
                 'restaurant_distance':'attraction_distance', 'attraction_distance':'attraction_distance',
                 'notebook':'notebook','planner':'planner'}


class CityError(Exception):
    pass


class DateError(Exception):
    pass


class ReactAgent:
    def __init__(self,
                 args,
                 mode: str = 'zero_shot',
                 tools: List[str] = None,
                 max_steps: int = 10,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name='glm-4-plus',
                 planner_llm_name='glm-4-plus',
                 #  logs_path = '../logs/',
                 city_file_path='../database/background/citySet.txt'
                 ) -> None:

        self.answer = ''
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == 'zero_shot':
            self.agent_prompt = zeroshot_react_agent_prompt
        elif self.mode == 'zero_shot_zh':
            self.agent_prompt = zeroshot_react_agent_prompt_zh

        self.json_log = []

        self.current_observation = ''
        self.current_data = None

        if 'glm-4' in react_llm_name:
            self.llm = LLMs(rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")
        else:
            print("LLM's name is getting wrong")
            self.llm = LLMs(rag_database="/home/wangb/cyo/graduation/rag/databases/hangzhou")

        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        # print(self.retry_record)

        self.last_actions = []

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

        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()


        print(self.scratchpad.split('\n')[-1])
        self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'\nThought {self.step_n}:', "")
        # self.log_file.write(self.scratchpad.split('\n')[-1] + '\n')

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        print(self.step_n)
        print("--------------action------------------")
        print(action)

        if action == None or action == '' or action == '\n':
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += ' ' + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:', "")

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
                elif action_type not in actionMapping:
                    pending_action = 'invalidAction'

                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1][
                            'state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
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

            if action_type == 'restaurant_info':
                try:
                    if True:
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),
                                                                  'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools[action_type].run(action_arg)
                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record['restaurant_info'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['restaurant_info'] += 1
                    self.current_observation = f'Illegal Restaurant info Search. Please try again.'
                    self.scratchpad += f'Illegal Flight Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'restaurant_distance':

                try:
                    if True:
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),
                                                                  'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools[action_type].run_distance(action_arg.split(', ')[0], action_arg.split(', ')[1])
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e:
                    self.retry_record[action_type] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record[action_type] += 1
                    self.current_observation = f'Illegal Attraction Search. Please try again.'
                    self.scratchpad += f'Illegal Attraction Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'attraction_info':

                try:
                    if True:
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),
                                                                  'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools[action_type].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record[action_type] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record[action_type] += 1
                    self.current_observation = f'Illegal Restaurant Search. Please try again.'
                    self.scratchpad += f'Illegal Restaurant Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'

            elif action_type == 'attraction_distance':
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),
                                                              'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools[action_type].run(action_arg)
                    self.current_observation = to_string(self.tools[action_type].run(action_arg.split(', ')[0], action_arg.split(', ')[1])).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record[action_type] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. State Error'

                except Exception as e:
                    print(e)
                    self.retry_record[action_type] += 1
                    self.current_observation = f'Illegal City Search. Please try again.'
                    self.scratchpad += f'Illegal City Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'

            elif action_type == 'notebook':
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),
                                                              'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_observation = str(self.tools[action_type].write(self.current_data, action_arg))
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except Exception as e:
                    print(e)
                    self.retry_record[action_type] += 1
                    self.current_observation = f'{e}'
                    self.scratchpad += f'{e}'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'


            elif action_type == "planner":
                # try:

                self.current_observation = str(
                    self.tools[action_type].run(str(self.tools['notebook'].list_all()), action_arg))
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

        #

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
                    print("Begin to generate----------")
                    request = format_step(self.llm(self._build_agent_prompt()))
                    print("------------scratchpad-----------")
                    print(self.scratchpad)
                    print("------------request--------------")
                    print(request)
                elif self.react_name == 'gemini':
                    request = format_step(self.llm.invoke(self._build_agent_prompt(), stop=['\n']).content)
                else:
                    # request = format_step(self.llm([HumanMessage(content=self._build_agent_prompt())]).content)
                    request = " "
                # print(request)
                return request
            except:

                print("Error !")
                return "Error !"

    def _build_agent_prompt(self) -> str:
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

        if 'notebook' in self.tools:
            self.tools['notebook'].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0

    def load_tools(self, tools: List[str]) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            if tool_name == 'restaurant_info' or tool_name == 'restaurant_distance':
                tools_map[tool_name] = apis.Restaurants()
            elif tool_name == 'attraction_info' or tool_name == 'attraction_distance':
                tools_map[tool_name] = apis.Attractions()
            elif tool_name == 'notebook':
                tools_map[tool_name] = Notebook()
            elif tool_name == 'planner':
                tools_map[tool_name] = Planner(self.planner_name)
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

    tools_list = ['restaurant_info', 'attraction_info', 'restaurant_distance', 'attraction_distance', 'notebook', 'planner']
    # model_name = ['gpt-3.5-turbo-1106','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral','ChatGLM3-6B-32K'][2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="glm-4-plus")
    parser.add_argument("--output_dir", type=str, default="./logs")
    args = parser.parse_args()
    agent = ReactAgent(None, mode="zero_shot_zh", tools=tools_list, max_steps=10, react_llm_name=args.model_name,
                       planner_llm_name=args.model_name)
    number = 1
    query = "请帮我推荐一个杭州的一日游，预算在200元以内，不要去人流量多的地方，想要去西湖。"
    # check if the directory exists
    if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
        os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
    if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
        result = [{}]
    else:
        result = json.load(
            open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))

    while True:
        planner_results, scratchpad, action_log = agent.run(query)
        if planner_results != None:
            break

    if planner_results == 'Max Token Length Exceeded.':
        result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
        result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
        action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
        result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
    else:
        result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
        result[-1][f'{args.model_name}_two-stage_results'] = planner_results
        result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

    # write to json file
    with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
        json.dump(result, f, indent=4)

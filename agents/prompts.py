from langchain.prompts import PromptTemplate


ZEROSHOT_REACT_INSTRUCTION = """Collect information for a query plan using interleaving 'Thought', 'Action', and 'Observation' steps. Ensure you gather valid information related to dining and attractions. All information should be written in Notebook, which will then be input into the Planner tool. Note that the nested use of tools is prohibited. 'Thought' can reason about the current situation, and 'Action' can have 6 different types:
(1) RestaurantInfo[Restaurant name]:
Description: A restaurant information retrieval tool.
Parameters:
Restaurant name: The name of the restaurant you would like to know more about.
Example: restaurant_info[KFC] would get the information of KFC, including latitude and longitude, cuisines, etc.

(2) RestaurantDistance[restaurant_1, restaurant_2]:
Description: Estimate the distance from two restaurants.
Parameters:
restaurant_1: The name of first restaurant.
restaurant_2: The name of second restaurant.
Example: restaurant_distance[Xiba tofu, Rest Hotpot] would provide distance between restaurant Xiba tofu and restaurant Rest Hotpot.

(3) AttractionInfo[attraction name]:
Description: An attraction information retrieval tool.
Parameters:
attraction name: The name of the attraction you would like to know more about.
Example: attraction_info[Forbidden City] would get the information of Forbidden City, including latitude and longitude.

(4) AttractionDistance[attraction_1, attraction_2]:
Description: Estimate the distance from two restaurants.
Parameters:
attraction_1: The name of first attraction.
attraction_2: The name of second attraction.
Example: restaurant_distance[Summer Palace, Forbidden City] would provide distance between attraction Forbidden City and attraction Summer Palace.

(5) NotebookWrite[Short Description]
Description: Writes a new data entry into the Notebook tool with a short description. This tool should be used immediately after FlightSearch, AccommodationSearch, AttractionSearch, RestaurantSearch or GoogleDistanceMatrix. Only the data stored in Notebook can be seen by Planner. So you should write all the information you need into Notebook.
Parameters: Short Description - A brief description or label for the stored data. You don't need to write all the information in the description. The data you've searched for will be automatically stored in the Notebook.
Example: NotebookWrite[Attraction infomation on Forbidden City] would store the attraction infomation on Forbidden City in the Notebook.

(6) Planner[Query]
Description: A smart planning tool that crafts detailed plans based on user input and the information stroed in Notebook.
Parameters: 
Query: The query from user.
Example: Planner[Give me a 1-day trip plan in Hangzhou city] would return a detailed 1-day trip plan in Hangzhou city.
You should use as many as possible steps to collect engough information to input to the Planner tool. 

Each action only calls one function once. Do not add any description in the action.

Query: {query}{scratchpad}"""

ZEROSHOT_REACT_INSTRUCTION_ZH = ""
# '''你是一个旅游线路规划专家，请结合用户的查询生成一条杭州的旅游线路推荐，用户查询想要去的景点必须包含在内，可以从必须要去的景点开始查找。使用交替进行的“思考”、“行动”和“观察”步骤收集查询计划所需的信息。确保收集到详细的与餐饮和景点相关的有效信息。请注意，不允许嵌套使用工具。“思考”可以推理当前的情况，“观察”会返回“行动”后查询到的内容，不需要你生成，“行动”可以有七种不同类型：
#
# (1) RestaurantInfo[餐厅名称]： 描述：一个餐厅信息检索工具。 参数： 餐厅名称：您想了解更多的餐厅名称。 示例：RestaurantInfo[海底捞火锅] 会获取海底捞火锅的相关信息，包括经纬度、菜系、人均价格等，需要注意，只有当餐厅文件内有这个餐厅时才会返回正确信息。
#
# (2) RestaurantDistance[餐厅1, 餐厅2]： 描述：估算两家餐厅之间的距离。 参数： 餐厅1：第一家餐厅的名称。 餐厅2：第二家餐厅的名称。 示例：RestaurantDistance[山舍茶屋·蟹料理, 近江海鲜城大排档] 会提供餐厅山舍茶屋·蟹料理和餐厅近江海鲜城大排档之间的距离。
#
# (3) RestaurantSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的餐厅信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的餐厅数量。 示例：RestaurantSearch[120.1470172, 30.26090127, 3]  会提供离经度 120.1470172 ，纬度 30.26090127 最近的3家餐厅的信息，其中'price'代表的数据为人均价格。
#
# (4) AttractionInfo[景点名称]： 描述：一个景点信息检索工具。 参数： 景点名称：您想了解更多的景点名称。 示例：AttractionInfo[紫禁城] 会获取紫禁城的相关信息，包括经纬度，需要注意，只有当景点文件内有这个景点时才会返回正确信息。
#
# (5) AttractionDistance[景点1, 景点2]： 描述：估算两个景点之间的距离。 参数： 景点1：第一个景点的名称。 景点2：第二个景点的名称。 示例：AttractionDistance[圆明园, 紫禁城] 会提供景点圆明园和景点紫禁城之间的距离。
#
# (6) AttractionSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的景点信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的景点数量。 示例：AttractionSearch[120.1454672, 30.244527, 4]  会提供离经度 120.1454672 ，纬度 30.244527 最近的4家景点的信息。
#
# (7) Planner[查询]： 描述：一个智能规划工具，根据用户输入和存储在笔记本中的信息制定详细的计划。 参数： 查询：用户的查询。 示例：Planner[给我一个在杭州的1日游计划] 会返回一个详细的杭州1日游计划。您应该尽可能多地使用步骤来收集足够的信息并输入到规划工具中，。
#
# 每次只进行一步思考和行动，最后要换行。每个行动仅调用一次某个功能。请不要在行动中添加任何描述，最多只能行动10次。你可以看到之前行动后的观察信息，这会辅助你进行下一步的思考，在收集查询完计划所需的所有信息后，最后请不要直接回答出规划好的旅游线路，应该要调用Planner，交给一个Planner工具，让这个工具规划出更加准确的旅游线路。
#
# *****示例*****
# 查询：我想要去西湖，请帮我规划一天的行程。
# Thought 1: 首先，我需要了解西湖的位置和相关信息，以便规划一日游行程。Action 1: AttractionInfo[西湖]。
# Thought 2: 知道了西湖的位置后，我需要知道KFC餐厅信息。Action 2: RestaurantInfo[KFC]。
# Thought 3: 知道了上述信息后，已经可以制定计划了。Action 3: Planner[我想要去西湖，请帮我规划一天的行程。]。
#
# *****示例结束*****
#
# 查询：{query}
# {scratchpad}
#
# '''

ZEROSHOT_REACT_INSTRUCTION_REFORMAT_ZH = '''你是一个旅游线路规划专家，请结合用户的查询生成一条杭州的旅游线路推荐，你需要制定全天的旅游线路以及午餐和晚餐的安排。使用交替进行的"Thought"、"Action"和"Observation"步骤收集查询计划所需的信息。确保收集到详细的与餐饮和景点相关的有效信息。请注意，不允许嵌套使用工具。"Thought"可以推理当前的情况，"Observation"会返回"Action"后查询到的内容，不需要你生成，"Action"可以有七种不同类型：

(1) RestaurantInfo[餐厅名称]： 描述：一个餐厅信息检索工具。 参数： 餐厅名称：您想了解更多的餐厅名称。 示例：RestaurantInfo[海底捞火锅] 会获取海底捞火锅的相关信息，包括经纬度、菜系、人均价格等。需要注意，只有当餐厅文件内有这个餐厅名称时才会返回正确信息，如果观察到的结果是这个城市没有当前所给餐厅，请使用RestaurantSearch工具查看给定经纬度附近已有餐厅。

(2) RestaurantSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的餐厅信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的餐厅数量。 示例：RestaurantSearch[120.1470172, 30.26090127, 3]  会提供离经度 120.1470172 ，纬度 30.26090127 最近的3家餐厅的信息，其中'price'代表的数据为人均价格。查找到信息后，不必再使用RestaurantInfo工具进一步查找具体的餐厅信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的餐厅。

(3) AttractionInfo[景点名称]： 描述：一个景点信息检索工具。 参数： 景点名称：您想了解更多的景点名称。 示例：AttractionInfo[紫禁城] 会获取紫禁城的相关信息，包括经纬度。需要注意，只有当景点文件内有这个景点时才会返回正确信息，如果观察到的结果是这个城市没有当前所给景点，请使用AttractionSearch工具查看给定经纬度附近已有景点。

(4) AttractionDistance[景点1, 景点2]： 描述：估算两个景点之间的距离。 参数： 景点1：第一个景点的名称。 景点2：第二个景点的名称。 示例：AttractionDistance[圆明园, 紫禁城] 会提供景点圆明园和景点紫禁城之间的距离，单位为千米。需要注意，在思考要先后去哪两个景点时，请调用该工具得到景点间的距离信息。

(5) AttractionSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的景点信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的景点数量。 示例：AttractionSearch[120.1454672, 30.244527, 4]  会提供离经度 120.1454672 ，纬度 30.244527 最近的4家景点的信息。查找到信息后，不必再使用AttractionInfo工具进一步查找具体的景点信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的景点。

(6) AttractionRetrieval[景点特点, 数量]  描述：获取给定景点特点的对应数量的景点名称与信息。  参数： 景点特点：想要搜索到的景点的特点。  数量：需要搜索的景点数量。  数量：需要搜索的景点数量。 示例：AttractionRetrieval[寺庙巡礼, 3]  会提供与寺庙巡礼相关的3个景点的信息，该工具适合用来查找满足用户要求的景点，在查找到这些景点后，请进一步使用AttractionInfo工具进一步查找具体的景点信息，安排的两个景点之间的距离不能过远。

(7) Planner[查询]： 描述：一个智能规划工具，会根据当前的"Thought"，"Action"和"Observation"信息制定详细的计划，所以请在搜集完所有必要的信息之后，最后再调用这个工具，调用完这个工具后不能再进行任何"Thought"与"Action"步骤了。 参数： 查询：用户想要的轨迹查询信息。 示例：Planner[给我一个在杭州的1日游计划] 会返回一个详细的杭州1日游计划。您应该尽可能多地使用"Thought"和"Action"步骤来收集足够的信息，在搜集完所有必要的信息后，才调用Planner工具，将用户原本的查询作为参数。

每次生成文本时，只进行一步"Thought"和一步"Action"，最后要换行，不需要再输出其他内容。每一步"Action"仅调用一次某个工具。请不要在"Action"中添加任何描述，最多只能行动10次。你可以看到之前"Action"后的"Observation"信息，这会辅助你进行下一步的思考，请不要进行参数完全一样的重复的"Action"操作，在收集查询完计划所需的所有信息后，最后请不要直接回答出规划好的旅游线路，应该要调用Planner，交给一个Planner工具，让这个工具规划出更加准确的旅游线路。

*****示例输入*****
查询：我想要去西湖，请帮我规划一天的行程。
可参考的已有相关路线：无
Thought 1: 首先，我需要了解西湖的位置和相关信息，以便规划一日游行程。Action 1: AttractionInfo[西湖]。
Observation 1: 景点西湖的信息为{{'shopname': '西湖', 'star': 0, 'longitude': 120.1470172, 'latitude': 30.26090127}}
Thought 2: 知道了西湖的位置后，我需要知道KFC餐厅信息。Action 2: RestaurantInfo[KFC]。
Observation 2: 餐厅的信息为{{'shopname': 'KFC', 'star': 0, 'longitude': 120.121312, 'latitude': 30.223907}}

*****示例输出*****
Thought 3: 知道了上述信息后，已经可以制定计划了。Action 3: Planner[我想要去西湖，请帮我规划一天的行程。]。

*****示例结束*****
查询：{query}
可参考的已有相关路线：{route_info}
{scratchpad}
'''


zeroshot_react_agent_prompt = PromptTemplate(
                        input_variables=["query", "scratchpad"],
                        template=ZEROSHOT_REACT_INSTRUCTION,
                        )
zeroshot_react_agent_prompt_zh = PromptTemplate(
                        input_variables=["query", "scratchpad"],
                        template=ZEROSHOT_REACT_INSTRUCTION_ZH,
                        )
zeroshot_react_agent_prompt_reformat_zh = PromptTemplate(
                        input_variables=["query", "route_info", "scratchpad"],
                        template=ZEROSHOT_REACT_INSTRUCTION_REFORMAT_ZH,
                        )

PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as restaurant names and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary.

***** Example *****
Query: Could you create a travel plan for 4 people in Hangzhou spanning 1 day,with a budget of 1,200 yuan?
Travel Plan:
Day 1:
Transportation: Subway
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: Xihu
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte

***** Example Ends *****

Given information: {text}
Query: {query}
Travel Plan:"""

PLANNER_INSTRUCTION_ZH = """您是一位熟练的规划师。根据提供的信息和查询，请为我提供一份详细的计划，包括餐厅名称和景点名称等具体内容。查询里想要去的景点必须要包含在计划内。
给定信息中有"Thought"与"Action"和"Observation"语句，其中"Thought"语句为每个步骤用户的想法，"Action"语句为每个步骤用户调用的工具，"Observation"语句为调用工具后返回给用户的内容。
请注意，您计划中的所有信息应来自提供的数据。您必须遵循示例中给出的格式。此外，所有细节应符合常识。符号 '-' 表示该信息不必要。
返回的结果以json格式输出，包括"交通"，"早餐"，"上午景点"，"午餐"，"下午景点"，"晚餐"，"晚上景点"属性。若在一个时间段内要去多个景点，请按照顺序依次写出，并用符号 ',' 分隔。
- 返回一个json格式的数据，具有以下键值对：
    - **交通**：推荐的出行方式。
    - **早餐**: 选择早上吃什么，可以不吃早餐。
    - **上午景点**：推荐上午去游玩的景点，必须要有。这里只能填写查询到的景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
    - **午餐**: 选择中午吃什么，必须要有。这里只能填写餐厅的具体名称，不能填写景点。
    - **下午景点**：推荐下午去游玩的景点，必须要有。这里只能填写查询到的景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
    - **晚餐**: 选择晚上吃什么，必须要有。这里只能填写餐厅的具体名称，不能填写景点。
    - **晚上景点**：推荐晚上去游玩的景点，可以没有。如果有的话，这里只能填写景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
***** 示例 *****
给定信息：麦当劳大约10元一位，景点有西湖与杭州博物馆，灵隐寺，良渚公园，净慈寺和雷峰塔。餐厅有绿茶餐厅，价格大约50元，麦坡咖啡，价格大约70元，海底捞火锅，价格大约95元。
查询：您能为4人制定一份在杭州的1日游计划，预算为1200元吗？ 
旅行计划： 
{{
    "交通": "地铁" ,
    "早餐": "麦当劳 ",
    "上午景点": "西湖",
    "午餐": "麦坡咖啡",
    "下午景点":  "杭州博物馆, 净慈寺, 雷峰塔",
    "晚餐": "海底捞火锅",
    "晚上景点": "灵隐寺, 良渚公园"
}}

***** 示例结束 *****

查询：{query}
给定信息：{text}
可参考的已有路线：{route}
旅行计划：

"""

EVALUATE_REQUEST_ROUTE_ZH = """你好，针对用户需求"{request}"和一条路线{route}，请你根据参考信息帮我判断这条路线是否和用户需求有冲突,路线能否满足用户的需求，请根据参考信息判断，并输出一个0-100之间的分数，分数越高，表明路线越符合用户的需求。只用输出一个0-100的数字，不要输出其他内容。
参考信息为{poi_info}
该路线的分数为
"""
evaluate_request_route_zh = PromptTemplate(
                        input_variables=["request", "route", "poi_info"],
                        template=EVALUATE_REQUEST_ROUTE_ZH,
                        )


COT_PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and hotel names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. Attraction visits and meals are expected to be diverse. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). 

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

Given information: {text}
Query: {query}
Travel Plan: Let's think step by step. First, """

REACT_PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and hotel names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. Attraction visits and meals are expected to be diverse. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). Solve this task by alternating between Thought, Action, and Observation steps. The 'Thought' phase involves reasoning about the current situation. The 'Action' phase can be of two types:
(1) CostEnquiry[Sub Plan]: This function calculates the cost of a detailed sub plan, which you need to input the people number and plan in JSON format. The sub plan should encompass a complete one-day plan. An example will be provided for reference.
(2) Finish[Final Plan]: Use this function to indicate the completion of the task. You must submit a final, complete plan as an argument.
***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
You can call CostEnquiry like CostEnquiry[{{"people_number": 7,"day": 1,"current_city": "from Ithaca to Charlotte","transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46","breakfast": "Nagaland's Kitchen, Charlotte","attraction": "The Charlotte Museum of History, Charlotte","lunch": "Cafe Maple Street, Charlotte","dinner": "Bombay Vada Pav, Charlotte","accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}}]
You can call Finish like Finish[Day: 1
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -]
***** Example Ends *****

You must use Finish to indict you have finished the task. And each action only calls one function once.
Given information: {text}
Query: {query}{scratchpad} """

REFLECTION_HEADER = 'You have attempted to give a sub plan before and failed. The following reflection(s) give a suggestion to avoid failing to answer the query in the same way you did previously. Use them to improve your strategy of correctly planning.\n'

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an automatic cost calculation environment, a travel query to give plan and relevant information. Only the selection whose name and city match the given information will be calculated correctly. You were unsuccessful in creating a plan because you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  

Given information: {text}

Previous trial:
Query: {query}{scratchpad}

Reflection:"""

REACT_REFLECT_PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and hotel names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. Attraction visits and meals are expected to be diverse. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). Solve this task by alternating between Thought, Action, and Observation steps. The 'Thought' phase involves reasoning about the current situation. The 'Action' phase can be of two types:
(1) CostEnquiry[Sub Plan]: This function calculates the cost of a detailed sub plan, which you need to input the people number and plan in JSON format. The sub plan should encompass a complete one-day plan. An example will be provided for reference.
(2) Finish[Final Plan]: Use this function to indicate the completion of the task. You must submit a final, complete plan as an argument.
***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
You can call CostEnquiry like CostEnquiry[{{"people_number": 7,"day": 1,"current_city": "from Ithaca to Charlotte","transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46","breakfast": "Nagaland's Kitchen, Charlotte","attraction": "The Charlotte Museum of History, Charlotte","lunch": "Cafe Maple Street, Charlotte","dinner": "Bombay Vada Pav, Charlotte","accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}}]
You can call Finish like Finish[Day: 1
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -]
***** Example Ends *****

{reflections}

You must use Finish to indict you have finished the task. And each action only calls one function once.
Given information: {text}
Query: {query}{scratchpad} """

planner_agent_prompt = PromptTemplate(
                        input_variables=["text","query"],
                        template = PLANNER_INSTRUCTION,
                        )

planner_agent_prompt_zh = PromptTemplate(
                        input_variables=["text","query", "route"],
                        template = PLANNER_INSTRUCTION_ZH,
                        )

cot_planner_agent_prompt = PromptTemplate(
                        input_variables=["text","query"],
                        template = COT_PLANNER_INSTRUCTION,
                        )

react_planner_agent_prompt = PromptTemplate(
                        input_variables=["text","query", "scratchpad"],
                        template = REACT_PLANNER_INSTRUCTION,
                        )

reflect_prompt = PromptTemplate(
                        input_variables=["text", "query", "scratchpad"],
                        template = REFLECT_INSTRUCTION,
                        )

react_reflect_planner_agent_prompt = PromptTemplate(
                        input_variables=["text", "query", "reflections", "scratchpad"],
                        template = REACT_REFLECT_PLANNER_INSTRUCTION,
                        )

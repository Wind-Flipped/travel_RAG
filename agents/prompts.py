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

ZEROSHOT_REACT_INSTRUCTION_ZH = '''你是一个旅游线路规划专家，请结合用户的查询生成一条杭州的旅游线路推荐，你需要制定全天的旅游线路以及午餐和晚餐的安排。使用交替进行的"Thought"、"Action"和"Observation"步骤收集查询计划所需的信息。确保收集到详细的与餐饮和景点相关的有效信息。请注意，不允许嵌套使用工具。"Thought"可以推理当前的情况，"Observation"会返回"Action"后查询到的内容，不需要你生成，"Action"可以有六种不同类型：

(1) RestaurantInfo[餐厅名称]： 描述：一个餐厅信息检索工具。 参数： 餐厅名称：您想了解更多的餐厅名称。 示例：RestaurantInfo[海底捞火锅] 会获取海底捞火锅的相关信息，包括经纬度、菜系、人均价格等。需要注意，只有当餐厅文件内有这个餐厅名称时才会返回正确信息，如果观察到的结果是这个城市没有当前所给餐厅，请使用RestaurantSearch工具查看给定经纬度附近已有餐厅。

(2) RestaurantSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的餐厅信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的餐厅数量。 示例：RestaurantSearch[120.1470172, 30.26090127, 3]  会提供离经度 120.1470172 ，纬度 30.26090127 最近的3家餐厅的信息，其中'price'代表的数据为人均价格。查找到信息后，不必再使用RestaurantInfo工具进一步查找具体的餐厅信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的餐厅。

(3) AttractionInfo[景点名称]： 描述：一个景点信息检索工具。 参数： 景点名称：您想了解更多的景点名称。 示例：AttractionInfo[紫禁城] 会获取紫禁城的相关信息，包括经纬度。推荐在第一步"Action"中调用该工具找到用户必去的景点的信息，需要注意，只有当景点文件内有这个景点时才会返回正确信息，如果观察到的结果是这个城市没有当前所给景点，请使用AttractionSearch工具查看给定经纬度附近已有景点。

(4) AttractionDistance[景点1, 景点2]： 描述：估算两个景点之间的距离。 参数： 景点1：第一个景点的名称。 景点2：第二个景点的名称。 示例：AttractionDistance[圆明园, 紫禁城] 会提供景点圆明园和景点紫禁城之间的距离，单位为千米。需要注意，在思考要先后去哪两个景点时，请调用该工具得到景点间的距离信息。

(5) AttractionSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的景点信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的景点数量。 示例：AttractionSearch[120.1454672, 30.244527, 4]  会提供离经度 120.1454672 ，纬度 30.244527 最近的4家景点的信息。查找到信息后，不必再使用AttractionInfo工具进一步查找具体的景点信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的景点。

(6) Planner[查询]： 描述：一个智能规划工具，会根据当前的"Thought"，"Action"和"Observation"信息制定详细的计划，所以请在搜集完所有必要的信息之后，最后再调用这个工具，调用完这个工具后不能再进行任何"Thought"与"Action"步骤了。 参数： 查询：用户想要的轨迹查询信息。 示例：Planner[给我一个在杭州的1日游计划] 会返回一个详细的杭州1日游计划。您应该尽可能多地使用"Thought"和"Action"步骤来收集足够的信息，在搜集完所有必要的信息后，才调用Planner工具，将用户原本的查询作为参数。

每次生成文本时，只进行一步"Thought"和一步"Action"，最后要换行，不需要再输出其他内容。每一步"Action"仅调用一次某个工具。请不要在"Action"中添加任何描述，最多只能行动10次。你可以看到之前"Action"后的"Observation"信息，这会辅助你进行下一步的思考，请不要进行参数完全一样的重复的"Action"操作，在收集查询完计划所需的所有信息后，最后请不要直接回答出规划好的旅游线路，应该要调用Planner，交给一个Planner工具，让这个工具规划出更加准确的旅游线路。

*****示例输入*****
查询：能否帮我规划一下在杭州的一日游行程？我想要一条包括文化古迹和自然美景的路线,并且特别想去断桥游览。
Thought 1: 用户想要在杭州规划一日游行程，特别想去断桥游览。Action 1: AttractionInfo[断桥]。
Observation 1: 景点断桥的信息为{{'shopname': '断桥', 'star': 0, 'longitude': 120.1472297, 'latitude': 30.26120261}}
Thought 2: 知道了断桥的位置后，我需要知道断桥附近景点信息。Action 2: AttractionSearch[120.1472297, 30.26120261, 2]。
Observation 2: 经纬度（120.1472297, 30.26120261）附近的2家景点的信息为[{{"shopname":"杭州上天竺法喜讲寺","star":50,"place":"灵隐\/白乐桥","type":"宗教","longitude":120.0911312,"latitude":30.22921186,"distance":0.67}},{{"shopname":"法喜寺","star":0,"place":null,"type":null,"longitude":120.0911312,"latitude":30.22921186,"distance":1.54}}]
Thought 3: 西湖是杭州的标志性景点，可以与断桥结合游览。Action 3: AttractionInfo[西湖]。
Observation 3: 景点西湖的信息为{{'shopname': '西湖', 'star': 0, 'longitude': 120.1470172, 'latitude': 30.26090127}}
Thought 4: 现在需要知道西湖附近的景点信息。Action 4: AttractionSearch[120.1470172, 30.26090127, 2]
Observation 4: 在经纬度（120.1470172, 30.26090127）附近的景点信息为[{{"shopname":"小河历史文化街区","star":50,"place":"","type":"","longitude":120.0911312,"latitude":30.22921186,"distance":0.52}}]
Thought 5: 确定了断桥和西湖，再加上一个文化古迹，行程较为丰富，现在需要确定西湖附近餐厅信息。Action 5: RestaurantSearch[120.1470172, 30.26090127, 2]
Observation 5: 在经纬度（120.1470172, 30.26090127）附近的2家餐厅的信息为：[{{"shopname":"湖畔·私房菜·杭帮菜(湖滨店)","price":"￥65","star":40,"type":"浙菜","longitude":120.154905,"latitude":30.26008035,"distance":0.7645047646}},{{"shopname":"木屋烧烤(武林路店)","price":"￥96","star":40,"type":"烤串","longitude":120.1550593,"latitude":30.26186047,"distance":0.7811816704}}]

*****示例输出*****
Thought 6: 所有信息已经收集完毕，可以制定计划了。Action 6: Planner[能否帮我规划一下在杭州的一日游行程？我想要一条包括文化古迹和自然美景的路线,并且特别想去断桥游览。]。

*****示例结束*****
查询：{query}
{scratchpad}
'''

ZEROSHOT_REACT_INSTRUCTION_REFORMAT_ZH = '''你是一个旅游线路规划专家，请结合用户的查询生成一条杭州的旅游线路推荐，你需要制定全天的旅游线路以及午餐和晚餐的安排。使用交替进行的"Thought"、"Action"和"Observation"步骤收集查询计划所需的信息。确保收集到详细的与餐饮和景点相关的有效信息。请注意，不允许嵌套使用工具。"Thought"可以推理当前的情况，"Observation"会返回"Action"后查询到的内容，不需要你生成，"Action"可以有六种不同类型：

(1) RestaurantInfo[餐厅名称]： 描述：一个餐厅信息检索工具。 参数： 餐厅名称：您想了解更多的餐厅名称。 示例：RestaurantInfo[海底捞火锅] 会获取海底捞火锅的相关信息，包括经纬度、菜系、人均价格等。需要注意，只有当餐厅文件内有这个餐厅名称时才会返回正确信息，如果观察到的结果是这个城市没有当前所给餐厅，请使用RestaurantSearch工具查看给定经纬度附近已有餐厅。

(2) RestaurantSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的餐厅信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的餐厅数量。 示例：RestaurantSearch[120.1470172, 30.26090127, 3]  会提供离经度 120.1470172 ，纬度 30.26090127 最近的3家餐厅的信息，其中'price'代表的数据为人均价格。查找到信息后，不必再使用RestaurantInfo工具进一步查找具体的餐厅信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的餐厅。

(3) AttractionInfo[景点名称]： 描述：一个景点信息检索工具。 参数： 景点名称：您想了解更多的景点名称。 示例：AttractionInfo[紫禁城] 会获取紫禁城的相关信息，包括经纬度。推荐在第一步"Action"中调用该工具找到用户必去的景点的信息，需要注意，只有当景点文件内有这个景点时才会返回正确信息，如果观察到的结果是这个城市没有当前所给景点，请使用AttractionSearch工具查看给定经纬度附近已有景点。

(4) AttractionDistance[景点1, 景点2]： 描述：估算两个景点之间的距离。 参数： 景点1：第一个景点的名称。 景点2：第二个景点的名称。 示例：AttractionDistance[圆明园, 紫禁城] 会提供景点圆明园和景点紫禁城之间的距离，单位为千米。需要注意，在思考要先后去哪两个景点时，请调用该工具得到景点间的距离信息。

(5) AttractionSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的景点信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的景点数量。 示例：AttractionSearch[120.1454672, 30.244527, 4]  会提供离经度 120.1454672 ，纬度 30.244527 最近的4家景点的信息。查找到信息后，不必再使用AttractionInfo工具进一步查找具体的景点信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的景点。

(6) Planner[查询]： 描述：一个智能规划工具，会根据当前的"Thought"，"Action"和"Observation"信息制定详细的计划，所以请在搜集完所有必要的信息之后，最后再调用这个工具，调用完这个工具后不能再进行任何"Thought"与"Action"步骤了。 参数： 查询：用户想要的轨迹查询信息。 示例：Planner[给我一个在杭州的1日游计划] 会返回一个详细的杭州1日游计划。您应该尽可能多地使用"Thought"和"Action"步骤来收集足够的信息，在搜集完所有必要的信息后，才调用Planner工具，将用户原本的查询作为参数。

每次生成文本时，只进行一步"Thought"和一步"Action"，最后要换行，不需要再输出其他内容。每一步"Action"仅调用一次某个工具。请不要在"Action"中添加任何描述，最多只能行动10次。你可以看到之前"Action"后的"Observation"信息，这会辅助你进行下一步的思考，请不要进行参数完全一样的重复的"Action"操作，在收集查询完计划所需的所有信息后，最后请不要直接回答出规划好的旅游线路，应该要调用Planner，交给一个Planner工具，让这个工具规划出更加准确的旅游线路。

*****示例输入*****
查询：能否帮我规划一下在杭州的一日游行程？我想要一条包括文化古迹和自然美景的路线,并且特别想去断桥游览。
可参考的已有相关路线：无
Thought 1: 用户想要在杭州规划一日游行程，特别想去断桥游览。Action 1: AttractionInfo[断桥]。
Observation 1: 景点断桥的信息为{{'shopname': '断桥', 'star': 0, 'longitude': 120.1472297, 'latitude': 30.26120261}}
Thought 2: 知道了断桥的位置后，我需要知道断桥附近景点信息。Action 2: AttractionSearch[120.1472297, 30.26120261, 2]。
Observation 2: 经纬度（120.1472297, 30.26120261）附近的2家景点的信息为[{{"shopname":"杭州上天竺法喜讲寺","star":50,"place":"灵隐\/白乐桥","type":"宗教","longitude":120.0911312,"latitude":30.22921186,"distance":0.67}},{{"shopname":"法喜寺","star":0,"place":null,"type":null,"longitude":120.0911312,"latitude":30.22921186,"distance":1.54}}]
Thought 3: 西湖是杭州的标志性景点，可以与断桥结合游览。Action 3: AttractionInfo[西湖]。
Observation 3: 景点西湖的信息为{{'shopname': '西湖', 'star': 0, 'longitude': 120.1470172, 'latitude': 30.26090127}}
Thought 4: 现在需要知道西湖附近的景点信息。Action 4: AttractionSearch[120.1470172, 30.26090127, 2]
Observation 4: 在经纬度（120.1470172, 30.26090127）附近的景点信息为[{{"shopname":"小河历史文化街区","star":50,"place":"","type":"","longitude":120.0911312,"latitude":30.22921186,"distance":0.52}}]
Thought 5: 确定了断桥和西湖，再加上一个文化古迹，行程较为丰富，现在需要确定西湖附近餐厅信息。Action 5: RestaurantSearch[120.1470172, 30.26090127, 2]
Observation 5: 在经纬度（120.1470172, 30.26090127）附近的2家餐厅的信息为：[{{"shopname":"湖畔·私房菜·杭帮菜(湖滨店)","price":"￥65","star":40,"type":"浙菜","longitude":120.154905,"latitude":30.26008035,"distance":0.7645047646}},{{"shopname":"木屋烧烤(武林路店)","price":"￥96","star":40,"type":"烤串","longitude":120.1550593,"latitude":30.26186047,"distance":0.7811816704}}]

*****示例输出*****
Thought 6: 所有信息已经收集完毕，可以制定计划了。Action 6: Planner[能否帮我规划一下在杭州的一日游行程？我想要一条包括文化古迹和自然美景的路线,并且特别想去断桥游览。]。

*****示例结束*****
查询：{query}
可参考的已有相关路线：{route_info}
{scratchpad}
'''

REFLECTION_ZH = '''你是一个旅游线路规划专家，请结合用户的查询生成一条杭州的旅游线路推荐，你需要制定全天的旅游线路以及午餐和晚餐的安排。使用交替进行的"Thought"、"Action"和"Observation"步骤收集查询计划所需的信息。确保收集到详细的与餐饮和景点相关的有效信息。请注意，不允许嵌套使用工具。"Thought"可以推理当前的情况，"Observation"会返回"Action"后查询到的内容，不需要你生成，"Action"可以有六种不同类型：

(1) RestaurantInfo[餐厅名称]： 描述：一个餐厅信息检索工具。 参数： 餐厅名称：您想了解更多的餐厅名称。 示例：RestaurantInfo[海底捞火锅] 会获取海底捞火锅的相关信息，包括经纬度、菜系、人均价格等。需要注意，只有当餐厅文件内有这个餐厅名称时才会返回正确信息，如果观察到的结果是这个城市没有当前所给餐厅，请使用RestaurantSearch工具查看给定经纬度附近已有餐厅。

(2) RestaurantSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的餐厅信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的餐厅数量。 示例：RestaurantSearch[120.1470172, 30.26090127, 3]  会提供离经度 120.1470172 ，纬度 30.26090127 最近的3家餐厅的信息，其中'price'代表的数据为人均价格。查找到信息后，不必再使用RestaurantInfo工具进一步查找具体的餐厅信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的餐厅。

(3) AttractionInfo[景点名称]： 描述：一个景点信息检索工具。 参数： 景点名称：您想了解更多的景点名称。 示例：AttractionInfo[紫禁城] 会获取紫禁城的相关信息，包括经纬度。推荐在第一步"Action"中调用该工具找到用户必去的景点的信息，需要注意，只有当景点文件内有这个景点时才会返回正确信息，如果观察到的结果是这个城市没有当前所给景点，请使用AttractionSearch工具查看给定经纬度附近已有景点。

(4) AttractionDistance[景点1, 景点2]： 描述：估算两个景点之间的距离。 参数： 景点1：第一个景点的名称。 景点2：第二个景点的名称。 示例：AttractionDistance[圆明园, 紫禁城] 会提供景点圆明园和景点紫禁城之间的距离，单位为千米。需要注意，在思考要先后去哪两个景点时，请调用该工具得到景点间的距离信息。

(5) AttractionSearch[经度, 纬度, 数量]： 描述：获取离给定经纬度最近的景点信息。  参数： 经度：给定的经度。 纬度：给定的纬度。  数量：需要搜索的景点数量。 示例：AttractionSearch[120.1454672, 30.244527, 4]  会提供离经度 120.1454672 ，纬度 30.244527 最近的4家景点的信息。查找到信息后，不必再使用AttractionInfo工具进一步查找具体的景点信息，必须要在知道某个景点的具体经纬度坐标后再使用该工具查询附近的景点。

(6) Planner[查询]： 描述：一个智能规划工具，会根据当前的"Thought"，"Action"和"Observation"信息制定详细的计划，所以请在搜集完所有必要的信息之后，最后再调用这个工具，若该工具已经可以做出计划，则任务结束，若该工具不通过当前的方案，认为需要查找额外的信息，则会返回新的观察结果，则请根据观察结果继续进行"Thought"与"Action"步骤，直到获取了足够的信息后，再调用该工具。 参数： 查询：用户想要的轨迹查询信息。 示例：Planner[给我一个在杭州的1日游计划] 会返回一个详细的杭州1日游计划。您应该尽可能多地使用"Thought"和"Action"步骤来收集足够的信息，在搜集完所有必要的信息后，才调用Planner工具，将用户原本的查询作为参数。

每次生成文本时，只进行一步"Thought"和一步"Action"，最后要换行，不需要再输出其他内容。每一步"Action"仅调用一次某个工具。请不要在"Action"中添加任何描述，最多只能行动15次。你可以看到之前"Action"后的"Observation"信息，这会辅助你进行下一步的思考，请不要进行参数完全一样的重复的"Action"操作，在收集查询完计划所需的所有信息后，最后请不要直接回答出规划好的旅游线路，应该要调用Planner，交给一个Planner工具，让这个工具规划出更加准确的旅游线路。如果调用Planner后返回了新的观察结果，请根据观察结果继续进行"Thought"与"Action"步骤，直到获取了足够的信息后，再次调用Planner工具。由于调用Planner工具可能会不通过当前的方案，请合理安排每个步骤的"Action"，不要超过15次。

*****示例输入*****
查询：能否帮我规划一下在杭州的一日游行程？我想要一条包括文化古迹和自然美景的路线,并且特别想去断桥游览，一共2个人，预算为300元。
Thought 1: 用户想要在杭州规划一日游行程，特别想去断桥游览。Action 1: AttractionInfo[断桥]。
Observation 1: 景点断桥的信息为{{'shopname': '断桥', 'star': 0, 'longitude': 120.1472297, 'latitude': 30.26120261}}
Thought 2: 知道了断桥的位置后，我需要知道断桥附近景点信息。Action 2: AttractionSearch[120.1472297, 30.26120261, 2]。
Observation 2: 经纬度（120.1472297, 30.26120261）附近的2家景点的信息为[{{"shopname":"杭州上天竺法喜讲寺","star":50,"place":"灵隐\/白乐桥","type":"宗教","longitude":120.0911312,"latitude":30.22921186,"distance":0.67}},{{"shopname":"法喜寺","star":0,"place":null,"type":null,"longitude":120.0911312,"latitude":30.22921186,"distance":1.54}}]
Thought 3: 西湖是杭州的标志性景点，可以与断桥结合游览。Action 3: AttractionInfo[西湖]。
Observation 3: 景点西湖的信息为{{'shopname': '西湖', 'star': 0, 'longitude': 120.1470172, 'latitude': 30.26090127}}
Thought 4: 现在需要知道西湖附近的景点信息。Action 4: AttractionSearch[120.1470172, 30.26090127, 2]
Observation 4: 在经纬度（120.1470172, 30.26090127）附近的景点信息为[{{"shopname":"小河历史文化街区","star":50,"place":"","type":"","longitude":120.0911312,"latitude":30.22921186,"distance":0.52}}]
Thought 5: 确定了断桥和西湖，再加上一个文化古迹，行程较为丰富，现在需要确定西湖附近餐厅信息。Action 5: RestaurantSearch[120.1470172, 30.26090127, 2]
Observation 5: 在经纬度（120.1470172, 30.26090127）附近的2家餐厅的信息为：[{{"shopname":"湖畔·私房菜·杭帮菜(湖滨店)","price":"￥65","star":40,"type":"浙菜","longitude":120.154905,"latitude":30.26008035,"distance":0.7645047646}},{{"shopname":"木屋烧烤(武林路店)","price":"￥96","star":40,"type":"烤串","longitude":120.1550593,"latitude":30.26186047,"distance":0.7811816704}}]
Thought 6: 所有信息已经收集完毕，可以制定计划了。Action 6: Planner[能否帮我规划一下在杭州的一日游行程？我想要一条包括文化古迹和自然美景的路线,并且特别想去断桥游览。]。
Observation 6: 一共2个人，预算为300元，平均每人预算为300/2=150元，湖畔·私房菜·杭帮菜(湖滨店)的价格为人均65元，木屋烧烤(武林路店)的价格为人均96元，用户吃饭的总预算为65+96=161元，超过了平均每人预算150元，需要重新选择餐厅。

*****示例输出*****
Thought 7: 预算不够，需要重新查找断桥附近的更多餐厅，找到更加便宜的餐厅作为午餐。Action 7: RestaurantSearch[20.1472297, 30.26120261, 2]

*****示例结束*****
查询：{query}
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
reflection_zh = PromptTemplate(
                        input_variables=["query", "scratchpad"],
                        template=REFLECTION_ZH,
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
返回的结果以json格式输出，包括"交通"，"早餐"，"上午景点"，"午餐"，"下午景点"，"晚餐"，"晚上景点"属性。若在一个时间段内要去多个景点，请按照顺序依次写出，并用符号 ',' 分隔。请注意，填写景点时应该参照工具查找后的精确值，不要对景点名称进行任何修改，例如"西湖"不要写成"西湖景区"或"西湖（游船）"，应当只写成"西湖"。
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

PLANNER_ZERO_SHOT_ZH = """您是一位熟练的规划师。根据提供的信息和查询，请为我提供一份详细的计划，包括餐厅名称和景点名称等具体内容。用户查询里想要去的景点必须要包含在计划内。
给定信息中有"Thought"与"Action"和"Observation"语句，其中"Thought"语句为每个步骤用户的想法，"Action"语句为每个步骤用户调用的工具，"Observation"语句为调用工具后返回给用户的内容。
请注意，您计划中的所有信息应来自提供的数据。您必须遵循示例中给出的格式。此外，所有细节应符合常识。符号 '-' 表示该信息不必要。
返回的结果以json格式输出，包括"交通"，"早餐"，"上午景点"，"午餐"，"下午景点"，"晚餐"，"晚上景点"属性。若在一个时间段内要去多个景点，请按照顺序依次写出，并用符号 ',' 分隔。请注意，用户查询中提到的想去的景点，那么这个景点必须要在计划中出现，即该景点必须要在"上午景点"，"下午景点"或"晚上景点"中。
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
旅行计划：

"""

PLANNER_REFLECTION_ZH = """您是一位熟练的规划师。根据提供的信息和查询，请为我提供一份详细的计划，包括餐厅名称和景点名称等具体内容。用户查询里想要去的景点必须要包含在计划内。
给定信息中有"Thought"与"Action"和"Observation"语句，其中"Thought"语句为每个步骤用户的想法，"Action"语句为每个步骤用户调用的工具，"Observation"语句为调用工具后返回给用户的内容。
请注意，您计划中的所有信息应来自提供的数据。您必须遵循示例中给出的格式。此外，所有细节应符合常识。符号 '-' 表示没有安排。
首先分析当前信息是否能规划出满足用户需求的行程，如果用户有要求想去的景点或餐厅，预算以及距离等信息，则最后的行程需要尽量全部满足。预算只用于餐厅的花费，不考虑景点的门票以及交通等额外花费，距离只考虑景点之间的距离，不考虑到餐厅的距离。如果用户没有预算或距离的要求，则表明预算充足或能接受很长的旅游距离，不需要再获取额外的用户信息。若已知的信息能满足用户全部需求，则在评价中给出"Success"，并以如下json格式输出线路，若已知的信息不能满足用户全部需求，需要额外的信息，则在评价中给出"Fail"，并在理由中给出不满足要求的理由以及需要额外查找的信息。请注意，必须要规划出午餐和晚餐，如果餐厅数量不够或价格超过预算，则需要重新查找餐厅信息，应在评价中给出"Fail"，需要知道额外的信息。
如果给定信息中用户已经调用过Planner工具，则表明您已经给出过一次"Fail"评价，这次需要给出"Success"评价，就利用已知信息为用户规划线路。
返回的路线结果以json格式输出，包括"分析"，"评价"，"理由"，"早餐"，"上午景点"，"午餐"，"下午景点"，"晚餐"，"晚上景点"属性。若在一个时间段内要去多个景点，请按照顺序依次写出，并用符号 ',' 分隔。请注意，用户查询中提到的想去的景点，那么这个景点必须要在计划中出现，即该景点必须要在"上午景点"，"下午景点"或"晚上景点"中。只用输出json格式的数据，不需要再输出其他内容。
- 返回一个json格式的数据，具有以下键值对：
    - **分析**：分析过程，包括是否满足用户需求，预算等信息。
    - **评价**：是否成功制定行程，可以为"Success"或"Fail"。
    - **理由**：若评价为"Fail"，需要给出不满足要求的理由以及需要额外查找的信息。
    - **早餐**: 选择早上吃什么，可以不吃早餐。
    - **上午景点**：推荐上午去游玩的景点，必须要有。这里只能填写查询到的景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
    - **午餐**: 选择中午吃什么，必须要有。这里只能填写餐厅的具体名称，不能填写景点。
    - **下午景点**：推荐下午去游玩的景点，必须要有。这里只能填写查询到的景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
    - **晚餐**: 选择晚上吃什么，必须要有。这里只能填写餐厅的具体名称，不能填写景点。
    - **晚上景点**：推荐晚上去游玩的景点，可以没有。如果有的话，这里只能填写景点的具体名称，不要加任何其他修饰或随意改动，不能填写餐厅。
***** 示例 1 *****
给定信息：麦当劳大约10元一位，景点有西湖与杭州博物馆，灵隐寺，良渚公园，净慈寺和雷峰塔。餐厅有绿茶餐厅，价格大约50元，麦坡咖啡，价格大约70元，海底捞火锅，价格大约95元。
查询：为4个人制定一份在杭州的1日游计划，想要去西湖，预算为1200元。 
旅行计划： 
{{
    "分析": "让我们一步一步的思考，用户想要去的景点有西湖，包含在给定信息内，预算为1200元，一共有4人，人均1200/4=300元，需要包括午餐，晚餐。根据给定信息，午餐选择绿茶餐厅，晚餐选择海底捞火锅，一共花费50+95=145元，小于人均300元的预算，可以制定一份行程。" ,
    "评价": "Success",
    "早餐": "-",
    "上午景点": "西湖",
    "午餐": "绿茶餐厅",
    "下午景点":  "杭州博物馆, 净慈寺, 雷峰塔",
    "晚餐": "海底捞火锅",
    "晚上景点": "灵隐寺, 良渚公园"
}}

***** 示例 2 *****
给定信息：景点有法喜寺，灵隐寺，良渚公园，净慈寺和雷峰塔。餐厅有北京烤鸭，价格大约150元，花间火锅，价格大约125元。
查询：为3个人制定一份在杭州的1日游计划，想要去灵隐寺，按照顺序参观不同景点之间的交通距离之和不要超过15km，预算为600元。 
旅行计划： 
{{
    "分析": "让我们一步一步的思考，用户想要去的景点有西湖，包含在给定信息内，预算为600元，一共有3人，人均600/3=200元，需要包括午餐，晚餐。根据给定信息，只有两个餐厅可供选择，一共花费150+125=275元，大于人均200元的预算，无法制定行程，需要再搜索到景点附近更便宜的餐厅。" ,
    "评价": "Fail",
    "理由": "午餐和晚餐的总花费为150+125=275元，大于人均200元的预算，无法制定行程，需要再搜索到景点附近更便宜的餐厅。"
}}

***** 示例结束 *****

查询：{query}
给定信息：{text}
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
planner_zero_shot_zh = PromptTemplate(
                        input_variables=["text","query"],
                        template = PLANNER_ZERO_SHOT_ZH,
                        )
planner_reflection_zh = PromptTemplate(
                        input_variables=["text","query"],
                        template = PLANNER_REFLECTION_ZH,
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



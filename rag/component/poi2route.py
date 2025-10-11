import json
from tool import Attractions
import re


class Poi2Route:
    def __init__(self):
        self.cache = {}
        self.tool = Attractions()
        with open("../databases/popularity.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.popularity = {eval(key): value for key, value in data.items()}

    def distance(self, poi1, poi2):
        answer = self.tool.run_for_distance(poi1, poi2)
        return answer

    def time(self, poi1, poi2):
        return self.tool.run_for_distance(poi1, poi2) / 0.5

    def calculate_segment_popularity(self, start, end, prefix_sum):
        if start == 0:
            return prefix_sum[end - 1]
        return prefix_sum[end - 1] - prefix_sum[start - 1]

    def select_most_popularity(self, route):
        n = len(route)
        if n < 2:
            return [], 0

        popularity_values = [self.popularity.get((route[i], route[i + 1]), -1) for i in range(n - 1)]

        prefix_sum = [0] * len(popularity_values)
        prefix_sum[0] = popularity_values[0]
        for i in range(1, len(popularity_values)):
            prefix_sum[i] = prefix_sum[i - 1] + popularity_values[i]

        best_segment = None
        max_popularity = -float('inf')

        for start in range(n - 1):
            for end in range(start + 2, n):  # 片段长度必须大于2
                segment_popularity = self.calculate_segment_popularity(start, end, prefix_sum) / (end - start)

                # 如果当前片段的受欢迎度高于之前的片段，则更新
                if segment_popularity > max_popularity:
                    max_popularity = segment_popularity
                    best_segment = route[start:end]

        return best_segment, max_popularity

    def calculate_cost(self, p1, p2, goal):

        key = tuple(sorted((p1, p2))) + (goal,)  # 生成三元组键 (p1, p2, goal)

        if key in self.cache:
            return self.cache[key]
        # 根据目标选择调用函数
        if goal == 'distance':
            return self.distance(p1, p2)
        elif goal == 'time':
            return self.time(p1, p2)
        else:
            raise ValueError("Invalid goal, must be 'distance' or 'time'")

    def cost(self, route, target):
        route_text = []
        cost = 0
        for i in range(len(route) - 1):
            start_station = route[i]
            end_station = route[i + 1]

            cur_cost = self.calculate_cost(start_station, end_station, target)
            # 计算当前两个站点之间的花费（时间/距离），这里假设 calculate_cost 方法返回的是分钟
            cost += cur_cost
            if target == "time":
                # 格式化输出为"第一站：A - n min - 第二站：B"
                route_text.append(f"第{i + 1}站：{start_station} - {cur_cost} min -> 第{i + 2}站：{end_station}")
            elif target == "distance":
                route_text.append(f"第{i + 1}站：{start_station} - {cur_cost} km -> 第{i + 2}站：{end_station}")

            # 将所有的描述连接起来，生成最终文本
        return " / ".join(route_text), cost

    def insert_pois_min_cost(self, route, pois, target):
        # 将 POI 列表按最小花费依次插入
        for poi in pois:
            min_cost = float('inf')  # 初始化最小花费为无穷大
            best_position = -1  # 最佳插入位置

            for i in range(len(route) + 1):
                if i == 0:  # 插入在第一个位置
                    cost = self.calculate_cost(poi, route[0], target) + sum(
                        self.calculate_cost(route[j], route[j + 1], target) for j in range(len(route) - 1))
                elif i == len(route):  # 插入在最后位置
                    cost = self.calculate_cost(poi, route[-1], target) + sum(
                        self.calculate_cost(route[j], route[j + 1], target) for j in range(len(route) - 1))
                else:  # 插入在中间
                    cost = sum(
                        self.calculate_cost(route[j], route[j + 1], target) for j in
                        range(i - 1)) + self.calculate_cost(
                        poi, route[i - 1], target) + self.calculate_cost(poi, route[i], target) + sum(
                        self.calculate_cost(route[j], route[j + 1], target) for j in range(i, len(route) - 1))

                if cost < min_cost:
                    min_cost = cost
                    best_position = i

            # 在计算出最佳位置后插入该 poi
            route.insert(best_position, poi)  # 在最佳位置插入 poi

        return route

    def input2answer(self, input_json):
        match = re.search(r'(\{.*\})', input_json, re.DOTALL)
        if match:
            extracted_json = match.group(1)
            data = json.loads(extracted_json)
        else:
            print("wrong json")
            return None

        route_array = data["route"].split(",")
        poi_array = data.get("poi", "").split(",") if "poi" in data else []

        if poi_array and '' not in poi_array :
            route = self.insert_pois_min_cost(route_array, poi_array, "distance")
        else:
            route = route_array
        time_route, time_cost = self.cost(route, "time")
        distance_route, distance_cost = self.cost(route, "distance")
        best_segment, _ = self.select_most_popularity(route)
        template = f'''
通过时间维度，距离维度，景区浏览模式三个维度，对你生成的路线{route}提供以下更多信息。
--
###维度说明：
时间维度: 给出浏览完该线路所需要的交通时间。
空间维度: 给出浏览完该线路所需要的交通距离。
景区浏览模式: 给出该路线中最受大众欢迎的子部分。

###时间维度
-按照该路线前进，总计交通（驾车）时间是{round(time_cost, 0)}min,具体时间如下：
{time_route}

###空间维度
-按照该路线前进，总计交通距离是{round(distance_cost, 0)}km,具体距离如下：
{distance_route}

###按照该线路前进，其中最受欢迎的子部分是：
{best_segment}
'''
        return template


if __name__ == '__main__':
    routeTool = Poi2Route()
    routeTool.input2answer('''
```json
{
  "route": "南宋御街,清河坊历史文化特色街区,大马弄,中山南路,河坊街,杭州博物馆,小河直街",
  "poi": "",
  "reason": "用户希望推荐一条亲子线路，多吃好吃的，且交通时间在10分钟内，不要去西湖。第2条路线攻略符合这些要求，包含多个美食打卡点和亲子友好的文化体验活动，且集中在市中心，交通便利。",
  "advantage": "这条路线适合家庭亲子游，涵盖了多个美食打卡点，如南宋御街的榨菜鲜肉月饼、大马弄的诸暨特色藤羹等，同时还有杭州博物馆这样的文化体验场所，适合小朋友学习和娱乐。路线集中在市中心，交通便利，符合用户对交通时间的要求。"
}
```
''')

import numpy as np
import json
from tool import Attractions
import math
from embedding import Zhipuembedding
from zhipuai import ZhipuAI
from tqdm import tqdm
import re
import csv
from poi2route import Poi2Route

tool = Poi2Route()
embedding_model = Zhipuembedding()

llm = ZhipuAI(api_key="a3a531fa24a2da257c09b9b8fd85d826.P6MoZZTHWTOPpkaQ")
tool = Attractions()
with open("../databases/popularity.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    popularity = {eval(key): value for key, value in data.items()}

with open('../databases/hangzhou_poi_coo2.json', 'r', encoding='utf-8') as file:
    poi_file = json.load(file)
poi_dict = {poi['name']: poi for poi in poi_file}


# # 打印所有 POI 的名称
# for poi_name in poi_dict:
#     print(poi_name)


def jaccard_similarity(route, truth):
    set1, set2 = set(route), set(truth)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def precision(route, truth):
    set1, set2 = set(route), set(truth)
    intersection = len(set1 & set2)
    return intersection / len(set1) if set1 else 0.0


def recall(route, truth):
    set1, set2 = set(route), set(truth)
    intersection = len(set1 & set2)
    return intersection / len(set2) if set2 else 0.0


def f1_score(route, truth):
    rec = recall(route, truth)
    prec = precision(route, truth)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球半径，单位：公里
    R = 6371
    distance = R * c
    return distance


def get_poi_description(poi):
    poi = poi_dict.get(poi)
    if poi:
        return poi['summarized_description']
    else:
        print(poi)
        print("description")
        raise ValueError("ERROR: Wrong poi, can not get description")


def get_poi_location(input):
    poi = poi_dict.get(input)
    if poi:
        return poi['longitude'], poi['latitude']
    else:
        print(input, poi)
        print("location")
        raise ValueError("ERROR: Wrong poi, can not get location")


def aggregate_poi_center(poi_list):
    total_lat = 0
    total_lon = 0
    valid_num = 0
    for poi in poi_list:
        try:
            lon, lat = get_poi_location(poi)
            total_lat += lon
            total_lon += lat
            valid_num += 1
        except ValueError as e:
            print(e)

    # 重心的经纬度
    center_lat = total_lat / valid_num
    center_lon = total_lon / valid_num
    return center_lon, center_lat


def aggregate_poi_distances(poi_list):
    total_distance = 0
    for i in range(len(poi_list) - 1):
        try:
            poi = poi_list[i]
            poi1 = poi_list[i + 1]
            lon, lat = get_poi_location(poi)
            lon1, lat1 = get_poi_location(poi1)
            distance = haversine(lat1, lon1, lat, lon)
            total_distance += distance
        except ValueError as e:
            print(e)
    return total_distance


def aggregate_poi_popularity(poi_list):
    total_popularity = 0
    for i in range(len(poi_list) - 1):
        poi = poi_list[i]
        poi1 = poi_list[i + 1]
        cur_popularity = popularity.get((poi, poi1), 0)
        total_popularity += cur_popularity
    return total_popularity


def calculate_center_distance(route, truth):
    center1_lon, center1_lat = aggregate_poi_center(route)
    center2_lon, center2_lat = aggregate_poi_center(truth)
    return haversine(center1_lat, center1_lon, center2_lat, center2_lon)


def calculate_distance_similarity(route, truth):
    distance1 = aggregate_poi_distances(route)
    distance2 = aggregate_poi_distances(truth)
    distance_diff = abs(distance1 - distance2)
    if distance2 == 0:
        print(truth)
    distance_similarity = distance_diff
    return distance_similarity


def calculate_time_similarity(route, truth):
    distance1 = aggregate_poi_distances(route)
    distance2 = aggregate_poi_distances(truth)
    distance_diff = abs(distance1 - distance2)
    if distance2 == 0:
        print(truth)
    distance_similarity = distance_diff
    return distance_similarity


def calculate_valid_td(route, d, t):
    distance = aggregate_poi_distances(route)
    if d * 0.8 < distance <= d * 1 and distance < t * 1 * 10:
        return 1
    else:
        return 0


def calculate_popularity_similarity(route, truth):
    popularity1 = aggregate_poi_popularity(route)
    popularity2 = aggregate_poi_popularity(truth)
    popularity_diff = abs(popularity1 - popularity2)
    popularity_similarity = popularity_diff
    return popularity_similarity


def calculate_ai_similarity(route, request):
    pois_description = [get_poi_description(poi) for poi in route]
    format_poi_info = "\n".join(f"-第{index + 1}条景点信息: {item}" for index, item in enumerate(pois_description))
    template = f'''
你好，针对用户需求"{request}"和一条路线{route}，请你根据参考信息帮我判断这条路线匹配这条路线的程度，程度的范围从0~10,10代表完全匹配，0代表完全不匹配。

###输出规范：
-只输出一个数字，即匹配程度，不要输出多余内容。

###可参考的信息
{format_poi_info}
'''
    response = llm.chat.completions.create(
        model="glm-4-air",
        messages=[
            {
                "role": "user",
                "content": template
            }
        ],
    )
    response = response.choices[0].message.content
    print(response)
    if 'Yes' in response:
        return 1
    return 0


def longest_common_subsequence(route, truth):
    # 获取 route 和 truth 的长度
    m = len(route)
    n = len(truth)

    # 创建一个二维 DP 数组，初始化为 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充 DP 数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if route[i - 1] == truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1  # 如果相等，公共子序列长度加 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # 否则，取最大值

    # dp[m][n] 存储的是最大公共子序列的长度
    return dp[m][n]


def process_file(file_path):
    jaccard = 0
    valid_num = 0
    subsequence_sum = 0
    dt = 0
    distance = 0
    popularity = 0
    recall_sum = 0
    precision_sum = 0
    f1_sum = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 循环处理数据
    for item in tqdm(all_data):
        truth = item['route']
        response = item['response']
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        if match:
            try:
                extracted_json = match.group(1)
                data = json.loads(extracted_json)
                route = data['route']
                route = [poi.strip() for poi in route.split(',')]
                valid_num += 1

                # 计算指标
                jaccard_val = jaccard_similarity(route, truth)
                dt_val = calculate_valid_td(route, item['distance'], item['time'])
                subsequence = longest_common_subsequence(route, truth)
                distance_val = calculate_distance_similarity(route, truth)
                pop_val = calculate_popularity_similarity(route, truth)
                recall_val = recall(route, truth)
                precision_val = precision(route, truth)
                f1_val = f1_score(route, truth)

                # 更新累加器
                jaccard += jaccard_val
                dt += dt_val
                distance += distance_val
                subsequence_sum += subsequence
                popularity += pop_val
                recall_sum += recall_val
                precision_sum += precision_val
                f1_sum += f1_val

            except (ZeroDivisionError, json.decoder.JSONDecodeError) as e:
                print(e)
        else:
            print("Output wrong JSON")

    # 统计平均值
    if valid_num > 0:
        avg_jaccard = jaccard / valid_num
        avg_subsequence = subsequence_sum / valid_num
        avg_dt = dt / valid_num
        avg_dis = distance / valid_num
        avg_pop = popularity / valid_num
        avg_recall = recall_sum / valid_num
        avg_precision = precision_sum / valid_num
        avg_f1 = f1_sum / valid_num

        # 打印最终统计
        print(f"Valid Data Points: {valid_num}")
        print(f"Average Jaccard Similarity (large better): {avg_jaccard:.4f}")
        # print(f"Average dt (large better): {avg_dt:.4f}")
        print(f"Average Recall (large better): {avg_recall:.4f}")
        print(f"Average Precision (large better): {avg_precision:.4f}")
        print(f"Average F1 Score (large better): {avg_f1:.4f}")
        print(f"Average Subsequence (large better): {avg_subsequence:.4f}")
        print(f"Average Distance (small better): {avg_dis:.4f}")
        print(f"Average Popularity (small better): {avg_pop:.4f}")

    else:
        print("No valid data points found.")


# 分别处理两个文件并计算结果
file_list = ["../output/output_requests_place2_td_k2_n2.json"]
for file in file_list:
    print(f"Processing file: {file}")
    process_file(file)

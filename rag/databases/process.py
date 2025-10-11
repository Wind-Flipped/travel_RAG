import json
import re
import pandas as pd

# 加载 JSON 文件
with open("/home/wangb/cyo/graduation/rag/databases/xihu/document.json", "r", encoding="utf-8") as f1, \
        open("/home/wangb/cyo/graduation/rag/databases/xihu/entire_vectors.json", "r", encoding="utf-8") as f2, \
        open("/home/wangb/cyo/graduation/rag/databases/xihu/topic_vectors.json", "r", encoding="utf-8") as f3:
    data1 = json.load(f1)
    data2 = json.load(f2)
    data3 = json.load(f3)

# 读取 CSV 文件并提取景点名
csv_file = "hangzhou_pois.csv"
csv_data = pd.read_csv(csv_file)
valid_pois = set(csv_data["name"])  # 将景点名存储为集合


# def extract_text(item1):
#     # 匹配“景点名字”后面直到换行符的文本
#     match = re.search(r"景点名字[:：](.+?)\n", item1)
#     if match:
#         return match.group(1).strip()  # 提取并去掉首尾空格
#     return None  # 如果未找到，返回 None

def extract_route(item):
    # 匹配“路线:”后面的地点，多个地点以逗号分隔
    match = re.search(r"路线[:：](.+)帖子", item)
    if match:
        # 提取并按逗号分隔，过滤掉空结果
        return [place.strip() for place in match.group(1).split(",") if place.strip()]
    return []


print(len(data1))
# 遍历三个 JSON 数据
filtered_data1, filtered_data2, filtered_data3 = [], [], []
for item1, item2, item3 in zip(data1, data2, data3):
    # print(extract_text(item1))
    poi_names = extract_route(item1)
    print(poi_names)
    if all(place in valid_pois for place in poi_names):  # 检查景点是否有效
        filtered_data1.append(item1)
        filtered_data2.append(item2)
        filtered_data3.append(item3)

print(len(filtered_data1))

with open("/home/wangb/cyo/graduation/rag/databases/xihu_1/document.json", "w", encoding="utf-8") as f1, \
     open("/home/wangb/cyo/graduation/rag/databases/xihu_1/entire_vectors.json", "w", encoding="utf-8") as f2, \
     open("/home/wangb/cyo/graduation/rag/databases/xihu_1/topic_vectors.json", "w", encoding="utf-8") as f3:
    json.dump(filtered_data1, f1, ensure_ascii=False, indent=4)
    json.dump(filtered_data2, f2, ensure_ascii=False, indent=4)
    json.dump(filtered_data3, f3, ensure_ascii=False, indent=4)

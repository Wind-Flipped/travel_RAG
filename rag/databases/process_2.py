import json
from zhipuai import ZhipuAI
from tqdm import tqdm
# 读取 JSON 文件
with open('hangzhou_poi_coo2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

client = ZhipuAI(api_key="8cf93821658b7df312645b6dc443b871.aY8DFiq17G0NrVmx")
# 修改数据：例如为每个地点增加一个新的字段
for location in tqdm(data):
    description = location['summarized_description']
    response = client.embeddings.create(
        model="embedding-2",
        input=description
    )
    location['embedding'] = response.data[0].embedding
# 保存回原文件
with open('hangzhou_poi_coo2.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("文件更新成功！")

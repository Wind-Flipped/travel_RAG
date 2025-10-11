import json
from zhipuai import ZhipuAI

with open("/home/wangb/cyo/graduation/rag/databases/xihu_1/document.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)
topic = []
entire = []
client = ZhipuAI(api_key="ff9747c0ac6262a47b8c9a80b253624b.pPgwRsgvGv5sjuph")
for data in data1:
    post_index = data.find("帖子")
    content_before_post = data[post_index:]
    response_route = client.chat.completions.create(
        model="glm-4-air",
        messages=[
            {"role": "user",
             "content": "请精炼下面的内容，删去不重要的部分，删去表情包等。帖子的内容是：" + content_before_post},
        ],
    )
    print(response_route.choices[0].message.content)
    response = client.embeddings.create(
        model="embedding-2",
        input=response_route.choices[0].message.content
    )
    entire.append(response.data[0].embedding)

with open('xihu_2/entire_vectors.json', 'w', encoding='utf-8') as file:
    json.dump(entire, file, ensure_ascii=False, indent=4)

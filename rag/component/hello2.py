# encoding: utf-8

import gradio as gr
import plotly.graph_objects as go

import numpy as np
import sys
from llms import LLMs

llm = LLMs(rag_database="/home/wangb/cyo/graduation/rag/databases/xihu_1")


def load_database():
    sys.stdout.reconfigure(encoding='utf-8')

    import json

    path = "../../trajGradio/data/seprate.json"

    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    return dataset


def extract_keywords(user_input, boroughs):
    keywords = []
    keywords.extend(boroughs)
    temp = user_input.split(",")
    temp = [item.strip() for item in temp]
    keywords.extend(temp)
    return keywords


def get_location_dic():
    location = np.load("../../trajGradio/data/coordinates.npy")
    location_dic = {}
    for item in location:
        location_dic[item[0]] = (item[1], item[2])
    return location_dic


location_dic = get_location_dic()


def get_similarity(keywords, dataset):
    best = -1
    best_traj = -1

    for item in dataset:
        traj = item["轨迹"].split(",")
        character = item["关键词"].split(",")

        traj = [item.strip() for item in traj]
        character = [item.strip() for item in character]

        now_list = []
        now_list.extend(traj)
        now_list.extend(character)
        cnt = 0
        for keyword in keywords:
            for cha in now_list:
                # 如果cha是keyword的字串
                if cha in keyword or keyword in cha:
                    cnt += 1
                    break
        if cnt > best:
            best = cnt
            best_traj = traj
    return best_traj


# 画图得做好，相似度计算，定位（这个做麻烦的）
def filter_map(user_input, boroughs):
    print(user_input, boroughs)
    response = llm.chat(user_input + ",".join(boroughs), "")
    print('-' * 100)
    print(response)
    # dataset = load_database()  # list: 关键词，轨迹
    # keywords = extract_keywords(user_input, boroughs)
    # best_traj = get_similarity(keywords, dataset)  # 计算每一条轨迹和keywords的相似度 lat是30 前面那个
    best_traj = response.split(",")
    print(best_traj)
    answer_traj = ""

    latitudes = []
    longitudes = []
    names = []
    steps = []
    for location in best_traj:
        if location in location_dic.keys():
            latitudes.append(location_dic[location][0])
            longitudes.append(location_dic[location][1])
            names.append(location)
            steps.append(f"第{len(latitudes)}站")
            answer_traj = answer_traj + f"第{len(latitudes)}站：{location}，"

    text_list = [(names[i], steps[i]) for i in range(0, len(names))]

    fig = go.Figure(go.Scattermapbox(
        customdata=text_list,
        lat=latitudes,
        lon=longitudes,
        mode='lines+markers',
        marker=go.scattermapbox.Marker(
            size=6
        ),

        line=dict(
            color='blue',
            width=2
        ),
        hoverinfo="text",
        hovertemplate='<b>景点</b>: %{customdata[0]}<br><b>次序</b>: %{customdata[1]}'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=30.2741,
                lon=120.1552
            ),
            pitch=0,
            zoom=11
        ),
    )

    return fig, answer_traj


with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            user_input = gr.Textbox(lines=2, placeholder="描述你的喜好", label="你的喜好")
        boroughs = gr.CheckboxGroup(choices=["小众", "大众", "晴天", "雨天", "网红", "citywalk"],
                                    value=["citywalk"], label="选择标签")
        btn = gr.Button(value="线路推荐")
        recommendations = gr.Textbox(label="推荐的路线")
        map = gr.Plot()
    demo.load(filter_map, [user_input, boroughs], [map, recommendations])
    btn.click(filter_map, [user_input, boroughs], [map, recommendations])

if __name__ == "__main__":
    import os

    # os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    demo.launch(server_name="0.0.0.0", server_port=32101)

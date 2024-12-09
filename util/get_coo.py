import json

# -*- coding: utf-8 -*-
import requests
import re
import pandas as pd
import math

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率


review_header = {
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9"}

# 腾讯获得经纬度
def get_location(keyword):
    url = "https://apis.map.qq.com/ws/place/v1/search?"
    params = {'keyword': keyword,
              'boundary': 'region(杭州 ,0)',
              'key': 'LQQBZ-JAOWZ-RWTX6-ZXLKJ-WGYX7-JTFA4',
              'page_size': 1}
    html = requests.get(url=url, headers=review_header, params=params).json()
    # print(html)

    # 火星坐标系转wgs1984坐标
    def gcj02towgs84(lng, lat):
        """
        GCJ02(火星坐标系)转GPS84
        :param lng:火星坐标系经度
        :param lat:火星坐标系纬度
        :return:
        """
        if out_of_china(lng, lat):
            return lng, lat
        dlat = transformlat(lng - 105.0, lat - 35.0)
        dlng = transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return [lng * 2 - mglng, lat * 2 - mglat]

    def transformlat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                math.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * pi) + 40.0 *
                math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
                math.sin(lat * pi / 30.0)) * 2.0 / 3.0
        return ret

    def transformlng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                math.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * pi) + 40.0 *
                math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
                math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
        return ret

    def out_of_china(lng, lat):
        """
        判断是否在国内，不在国内不进行纠偏
        :param lng:
        :param lat:
        :return:
        """
        if lng < 72.004 or lng > 137.8347:
            return True
        if lat < 0.8293 or lat > 55.8271:
            return True
        return False

    return gcj02towgs84(html['data'][0]['location']['lng'], html['data'][0]['location']['lat'])

# 更换访问头直到返回正确数据

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 保存JSON文件
def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def main():
    input_file = 'output/hangzhou_poi_coo1.json'  # 输入文件名
    output_file = 'output/hangzhou_poi_coo2.json'  # 输出文件名

    # 读取JSON文件
    data = read_json_file(input_file)

    # 遍历每一项，获取"name"字段，并搜索经纬度
    for item in data:
        name = item.get('name')
        if (('latitude' not in item or item['latitude'] == 0)
                or ('longitude' not in item or item['longitude'] == 0)):
            try:
                lon, lat = get_location(name)
                item['latitude'] = lat
                item['longitude'] = lon
            except Exception as e:
                print("Error in fetching coordinates", e)
                item['latitude'] = 0
                item['longitude'] = 0

    # 保存更新后的数据
    save_json_file(data, output_file)


# 3. 主程序
def main2():
    input_file = "output/unique_places1.json"
    output_file = "output/unique_places2.json"
    # 读取输入的 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        locations = json.load(f)

    # 查询每个地点的经纬度
    results = {}
    for location in locations:
        if location["longitude"] == 0 and location["latitude"] == 0:
            try:
                print(f"查询 {location} 的经纬度...")
                lon, lat = get_location(location)
                results[location] = {"longitude": float(lon), "latitude": float(lat)}
            except Exception as e:
                print("Error in fetching coordinates", e)
                results[location] = {"longitude": 0, "latitude": 0}

    # 保存结果到新的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至 {output_file}")

def main3():
    """
    检查每个地点的经纬度是否为零，如果为零则重新查询。
    """

    input_file = "output/unique_places2.json"
    output_file = "output/unique_places3.json"
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检查并更新经纬度

    for location, coords in data.items():
        if coords["longitude"] == 0 and coords["latitude"] == 0:
            try:
                print(f"查询 {location} 的经纬度...")
                lon, lat = get_location(location)
                data[location] = {"longitude": float(lon), "latitude": float(lat)}
            except Exception as e:
                print("Error in fetching coordinates", e)
                data[location] = {"longitude": 0, "latitude": 0}

    # 保存更新后的数据回原文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"已更新文件中的经纬度，结果保存至 {output_file}")

if __name__ == '__main__':
    # SOSBZ-G22KZ-ZTMXF-TSE3M-MLURJ-V3BPK (cyo)
    # MAEBZ-PCVCD-CUX4Z-HJ42R-L57SK-4CFBP (example)
    # LQQBZ-JAOWZ-RWTX6-ZXLKJ-WGYX7-JTFA4 (lyq)
    main3()
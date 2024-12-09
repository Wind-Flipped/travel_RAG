from geopy.distance import geodesic


def calculate_distance(coord1, coord2):
    """
    distance

    :param coord1: 第一个坐标，格式为 (纬度, 经度)
    :param coord2: 第二个坐标，格式为 (纬度, 经度)
    :return: 距离（单位：千米）
    """
    distance = geodesic(coord1, coord2).kilometers
    return distance


if __name__ == '__main__':
    print(calculate_distance((39.909, 116.43), (39.919, 116.38)))
import numpy as np
from math import sqrt


# 定义一个函数用于计算轮廓的长宽，确定最后的输出结果
def calculateWH(PolygonPoints):
    if len(PolygonPoints) != 4:
        raise ValueError("your polygon is error!")
    Points = np.array(PolygonPoints)

    # 计算两点之间的距离
    def DistancePoints(p1, p2):
        Dis = sqrt((p2[0][0] - p1[0][0]) ** 2 + (p2[0][1] - p1[0][1]) ** 2)
        return Dis

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0)
    ]
    # 计算四边长
    edge_length = []
    for edge in edges:
        p1, p2 = edge
        length = DistancePoints(Points[p1], Points[p2])
        edge_length.append({"edge": edge, "length": length,
                            "points": (Points[p1].tolist(), Points[p2].tolist())})

    longest_edge = max(edge_length, key=lambda x: x["length"])
    maxedge = longest_edge["length"]

    # 找出最长边的两个顶点索引
    vertex1, vertex2 = longest_edge["edge"]

    # 找出与vertex1相连的其他边（不包括最长边）
    vertex1_edges = []
    for edge in edges:
        if edge != longest_edge["edge"] and edge != (longest_edge["edge"][1], longest_edge["edge"][0]):  # 排除反向边
            if vertex1 in edge:
                other_vertex = edge[0] if edge[1] == vertex1 else edge[1]
                vertex1_edges.append({
                    "edge": edge,
                    "length": DistancePoints(Points[vertex1], Points[other_vertex]),
                    "points": (Points[vertex1].tolist(), Points[other_vertex].tolist())
                })

    vertex2_edges = []
    for edge in edges:
        if edge != longest_edge["edge"] and edge != (longest_edge["edge"][1], longest_edge["edge"][0]):  # 排除反向边
            if vertex2 in edge:
                other_vertex = edge[0] if edge[1] == vertex2 else edge[1]
                vertex2_edges.append({
                    "edge": edge,
                    "length": DistancePoints(Points[vertex2], Points[other_vertex]),
                    "points": (Points[vertex2].tolist(), Points[other_vertex].tolist())
                })
    # 结果反馈
    print(maxedge)
    print(vertex1_edges[0]["length"], vertex2_edges[0]["length"])
    if vertex1_edges is None or vertex2_edges is None:
        raise ValueError("Error!")
    else:
        Secondedge = max(vertex1_edges[0]["length"], vertex2_edges[0]["length"])

    return Secondedge, maxedge

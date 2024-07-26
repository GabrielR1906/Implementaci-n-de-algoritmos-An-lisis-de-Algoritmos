import heapq
from collections import defaultdict


class Graph:
    def __init__(self, is_directed=False):
        self.graph = defaultdict(list)
        self.is_directed = is_directed

    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))
        if not self.is_directed:
            self.graph[v].append((u, w))

    def load_from_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                u, w, v = map(int, line.strip().split(','))
                self.add_edge(u, v, w)

    def save_to_file(self, edges, file_path):
        with open(file_path, 'w') as file:
            for u, v, w in edges:
                file.write(f"{u},{w},{v}\n")


def kruskal(graph):
    parent = {}
    rank = {}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    edges = [(w, u, v) for u in graph.graph for v, w in graph.graph[u]]
    edges = sorted(set(edges))

    for node in graph.graph:
        parent[node] = node
        rank[node] = 0

    mst = []
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, weight))

    return mst


def prim(graph, start_node=0):
    visited = set()
    min_heap = [(0, start_node, -1)]
    mst = []
    while min_heap:
        weight, u, prev = heapq.heappop(min_heap)
        if u not in visited:
            visited.add(u)
            if prev != -1:
                mst.append((prev, u, weight))
            for v, w in graph.graph[u]:
                if v not in visited:
                    heapq.heappush(min_heap, (w, v, u))
    return mst


def dijkstra(graph, start_node=0):
    distances = {node: float('infinity') for node in graph.graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    predecessors = {node: None for node in graph.graph}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph.graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return predecessors, distances


def save_dijkstra_paths(predecessors, distances, start_node, file_path):
    with open(file_path, 'w') as file:
        for node in distances:
            if node != start_node:
                path = []
                current = node
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path = path[::-1]
                for i in range(len(path) - 1):
                    file.write(f"{path[i]},{distances[path[i + 1]] - distances[path[i]]},{path[i + 1]}\n")


# Función para ejecutar los algoritmos y guardar resultados
def execute_algorithms(graph_file, prefix):
    graph = Graph()
    graph.load_from_file(graph_file)

    # Ejecutar y guardar resultados de Kruskal
    mst_kruskal = kruskal(graph)
    graph.save_to_file(mst_kruskal, f'{prefix}_kruskal.txt')

    # Ejecutar y guardar resultados de Prim
    mst_prim = prim(graph, start_node=0)
    graph.save_to_file(mst_prim, f'{prefix}_prim.txt')

    # Ejecutar y guardar resultados de Dijkstra
    predecessors, distances = dijkstra(graph, start_node=0)
    save_dijkstra_paths(predecessors, distances, start_node=0, file_path=f'{prefix}_dijkstra.txt')


# Ejecutar los algoritmos para ambos grafos
execute_algorithms('C:/Users/renat/Desktop/Escritorio/Universidad/Cuarto Ciclo/Análisis de Algoritmos/Grafo30.txt', 'resultado_grafo30')
execute_algorithms('C:/Users/renat/Desktop/Escritorio/Universidad/Cuarto Ciclo/Análisis de Algoritmos/Grafo50.txt', 'resultado_grafo50')

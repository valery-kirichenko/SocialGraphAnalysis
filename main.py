from math import log
import csv
import copy
import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self, nodes=None, adjacency=None):
        self._nodes = [] if nodes is None else nodes
        self._adjacency = [] if adjacency is None else adjacency

    @classmethod
    def read_csv(cls, file_name):
        with open(file_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            nodes = list(filter(None, reader.__next__()))
            adjacency = []
            for line in reader:
                adjacency.append(list(map(int, line[1:])))
            return cls(nodes, adjacency)
 
    def neighbors(self, node):
        for node_index, is_adjacent in enumerate(self._adjacency[self._nodes.index(node)]):
            if is_adjacent:
                yield self._nodes[node_index]

    def common_neighbors(self, node1, node2):
        return list(set(self.neighbors(node1)) & set(self.neighbors(node2)))

    def nodes(self):
        return self._nodes.copy()

    def adjacency_matrix(self):
        return copy.deepcopy(self._adjacency)

    def add_nodes_from(self, nodes_list):
        self._nodes.extend(nodes_list)

        if len(self._adjacency) > 0:
            for row in self._adjacency:
                row.extend([0] * len(nodes_list))
            self._adjacency.extend([[0] * len(self._adjacency[0]) for _ in range(len(nodes_list))])
        else:
            self._adjacency = [[0] * len(nodes_list) for _ in range(len(nodes_list))]

    def add_edges_from(self, edges_list):
        for edge in edges_list:
            self._adjacency[self._nodes.index(edge[0])][self._nodes.index(edge[1])] = 1

    def subgraph(self, nodes):
        nodes = list(nodes)
        indices = [self._nodes.index(node) for node in nodes]
        adjacency = [[0] * len(nodes) for _ in range(len(nodes))]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                adjacency[i][j] = self._adjacency[indices[i]][indices[j]]
        return self.__class__(nodes, adjacency)

    def __str__(self):
        return '\n'.join(map(str, self._adjacency))


class DiGraph(Graph):
    def to_undirected(self):
        adjacency = self.adjacency_matrix()
        for y in range(len(adjacency)):
            for x in range(len(adjacency[y])):
                adjacency[y][x] = adjacency[x][y] or adjacency[y][x]
        return Graph(self.nodes(), adjacency)

    def reverse(self):
        adjacency = [
            [self._adjacency[j][i] for j in range(len(self._adjacency))]
            for i in range(len(self._adjacency[0]))
        ]
        return DiGraph(self.nodes(), adjacency)


def dfs(graph, start_node, time=0, visited=None):
    if visited is None:
        visited = set()
    stack = []
    entry = {}
    leave = {}
    stack.append(start_node)
    while stack:
        v = stack.pop()
        if v is None:  # Next node was already visited, need to rememeber it's exit time
            time += 1
            leave[stack.pop()] = time
            continue
        if v not in visited:  # Node wasn't visited
            time += 1
            entry[v] = time
            visited.add(v)
            stack.extend([v, None])  # Add parent node with exit marker
            stack.extend(graph.neighbors(v))  # Add its neighbors to traverse
    return visited, entry, leave


def dfs_full(graph):
    time = 0
    visited = set()
    entry = {}
    leave = {}
    for node in graph.nodes():
        if node not in visited:
            visited_comp, entry_comp, leave_comp = dfs(graph, node, time=time, visited=visited)
            visited.update(visited_comp)
            entry.update(entry_comp)
            leave.update(leave_comp)
            time = max(leave_comp.values())
    return visited, entry, leave


def strongly_connected_components(digraph):
    reversed_graph = digraph.reverse()
    dfs_list, entry, post = dfs_full(reversed_graph)
    sorted_post = sorted(post, key=post.get, reverse=True)
    visited = set()
    sccs = []
    for v in sorted_post:
        scc, _, _ = dfs(digraph, v, visited=visited.copy())
        if scc - visited:
            sccs.append(scc - visited)
        visited.update(scc)
    return sccs


def weakly_connected_components(digraph):
    graph = digraph.to_undirected()
    visited = set()
    wccs = []
    for node in graph.nodes():
        if node not in visited:
            wcc, _, _ = dfs(graph, node)
            wccs.append(wcc - visited)
            visited.update(wcc)
    return wccs


def floyd_warshall(graph):
    distances = graph.adjacency_matrix()
    n = len(graph.nodes())

    # Prepare distances matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = distances[i][j] if distances[i][j] else n

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
    return distances


def adjacency_matrix(graph):
    def adjacency_row(node):
        neighbors = list(graph.neighbors(node))
        row = list(map(lambda x: str(int(x in neighbors)), graph.nodes()))
        return ';'.join(row)

    return ';' + ';'.join(graph.nodes()) + '\n' + \
           '\n'.join(map(lambda node: f'{node};{adjacency_row(node)}', graph.nodes()))


def adjacency_list(graph):
    def adjacency_record(node):
        neighbors = list(graph.neighbors(node))
        return node + (';' if len(neighbors) > 0 else '') + ';'.join(neighbors)

    return '\n'.join(map(lambda node: adjacency_record(node), graph.nodes()))


def similarity_measures(graph, mode):
    def measure(x, y):
        if x == y:
            return ''
        if mode == 0:  # Common neighbors
            return len(graph.common_neighbors(x, y))
        if mode == 1:  # Jaccard's Coefficient
            return len(graph.common_neighbors(x, y)) / len(set(graph.neighbors(x)) | set(graph.neighbors(y)))
        if mode == 2:  # Adamic/Adar
            return sum([1 / log(len(list(graph.neighbors(z)))) for z in graph.common_neighbors(x, y)])
        if mode == 3:  # Preferential Attachment
            return len(list(graph.neighbors(x))) * len(list(graph.neighbors(y)))

    nodes = graph.nodes()
    measures = [list(nodes)]
    measures[0].insert(0, '')
    for x in nodes:
        node_measures = [measure(x, y) for y in nodes]
        node_measures.insert(0, x)
        measures.append(node_measures)

    return measures


def task1(graph):
    wccs = weakly_connected_components(graph)
    sccs = strongly_connected_components(graph)

    if len(wccs) == 1:
        print("Граф является слабосвязным")
    else:
        print("Граф не является слабосвязным")

    if len(sccs) == 1:
        print("Граф является сильносвязным")
    else:
        print("Граф не является сильносвязным")

    print("Число компонент слабой связности:", len(wccs))
    print("Число компонент сильной связности:", len(sccs))
    print("Количество вершин в компонентах слабой связности:", ", ".join(map(lambda x: str(len(x)), wccs)))
    print("Количество вершин в компонентах сильной связности:", ", ".join(map(lambda x: str(len(x)), sccs)))
    print(f"Наибольшей компоненте слабой связности принадлежит "
          f"{len(max(wccs, key=len)) / len(graph.nodes()) * 100:.2f}% узлов")


def task2(graph):
    wccs = weakly_connected_components(graph)
    max_wcc_nodes = max(wccs, key=len)
    max_wcc_subgraph = graph.subgraph(max_wcc_nodes).to_undirected()

    average_node_degree = sum(map(lambda node: len(list(max_wcc_subgraph.neighbors(node))), max_wcc_nodes))\
                          / len(max_wcc_nodes)
    print("Средняя степень вершины:", average_node_degree)

    distances = floyd_warshall(max_wcc_subgraph)
    eccentricities = list(map(max, distances))

    diameter = max(eccentricities)
    radius = min(eccentricities)
    print("Диаметр графа:", diameter)
    print("Радиус графа:", radius)

    central_nodes = [max_wcc_subgraph.nodes()[node_index] for node_index, eccentricity in enumerate(eccentricities) if
                     eccentricity == radius]
    peripheral_nodes = [max_wcc_subgraph.nodes()[node_index] for node_index, eccentricity in enumerate(eccentricities)
                        if eccentricity == diameter]
    print("Центральные вершины:", ", ".join(central_nodes))
    print("Периферийные вершины:", ", ".join(peripheral_nodes))

    average_distance = sum(map(sum, distances)) / (len(max_wcc_nodes) ** 2 - len(max_wcc_nodes))
    print("Средняя длина пути:", average_distance)

    node_degrees = list(map(lambda node: len(list(max_wcc_subgraph.neighbors(node))), max_wcc_nodes))
    degree_prob = {}
    for degree in node_degrees:
        degree_prob[degree] = len([1 for x in node_degrees if x == degree]) / len(max_wcc_nodes)
    plt.bar(degree_prob.keys(), degree_prob.values())
    plt.savefig("degree_prob.png")


def task3(graph):
    wccs = weakly_connected_components(graph)
    max_wcc_nodes = max(wccs, key=len)
    max_wcc_subgraph = graph.subgraph(max_wcc_nodes).to_undirected()
    measures = {0: "Common neighbors", 1: "Jaccard's Coefficient", 2: "Adamic_Adar", 3: "Preferential Attachment"}
    for mode in measures:
        measure = similarity_measures(max_wcc_subgraph, mode)
        with open(f"{measures[mode]}.csv", "w") as output:
            output.write('\n'.join(map(lambda row: ';'.join(map(str, row)), measure)))


dg = DiGraph.read_csv('graph.csv')

task1(dg)
task2(dg)
task3(dg)

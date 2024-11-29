graph = {
    '5' : ['3','7'],
    '3' : ['2','4'],
    '7' : ['8'],
    '2' : [],
    '4' : ['8'],
    '8' : []
}

def ids(graph, target, d):
    for i in range(d):
        print(f'Iteration no. {i+1} , Depth Limit: {i+1}')
        visited = set()
        if depth_limit(visited, graph, target, '5', 0, i):
            print(f'Goal Node found at Depth {i+1}')
            return
    print('Goal Node not found')


def depth_limit(visited, graph, target, node, l, d):
    if node not in visited and l <= d:
        print(node)
        if node == target:
            return True
        visited.add(node)
        for neighbour in graph[node]:
            if depth_limit(visited, graph, target, neighbour, l + 1, d):
                return True

print('Following is the Iterative Deepening Search')
ids(graph,'4',4)
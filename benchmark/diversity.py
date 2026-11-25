

def tsp_edge_distance(route_a, route_b):
    edges_a = set()
    edges_b = set()
    for i in range (len(route_a)):
        city = route_a[i]
        next_city = route_a[(i+1) % len(route_a)]
        edge = frozenset([city, next_city])
        edges_a.add(edge)

    for i in range (len(route_b)):
        city = route_b[i]
        next_city = route_b[(i+1) % len(route_b)]
        edge = frozenset([city, next_city])
        edges_b.add(edge)

    common_edges = edges_a.intersection(edges_b)
    return 1 - len(common_edges) / len(route_a)

def tsp_population_diversity(population):
    total_distance = 0
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            total_distance += tsp_edge_distance(population[i], population[j])

    n = len(population)
    num_pairs = n * (n - 1) / 2
    return total_distance / num_pairs
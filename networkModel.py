import networkx as nx
import random
import math

G = nx.MultiDiGraph()

supply_nodes = ["Retailer1", "Retailer2"]
G.add_nodes_from(supply_nodes, layer="Retail Chain")

consumer_nodes = ["Consumer1", "Consumer2", "Consumer3", "Consumer4", "Consumer5", "Consumer6"]
G.add_nodes_from(consumer_nodes, layer="Consumers")

social_edges = []
for count, item in enumerate(consumer_nodes):
    if count != len(consumer_nodes) -1:
        social_edges.append((item, consumer_nodes[count+1]))
G.add_edges_from(social_edges, weight=0.3, layer="Social")

edges = [
    ("Supplier", "Processor1", 0.8),
    ("Processor1", "Retailer1", 0.7),
    ("Retailer1", "Consumer1", 0.9),
    ("Retailer1", "Consumer2", 0.4),
]
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2], layer="Supply Chain")

contamination_status = {node: 0 for node in G.nodes}
contamination_status["Supplier"] = 1

def simulate_contamination(G, contamination_status, steps=5):
    for _ in range(steps):
        new_status = contamination_status.copy()
        for node in G.nodes:
            if contamination_status[node] > 0:
                new_status[node] *= math.exp(-0.1)

                for neighbor in G.successors(node):
                    transmission_prob = G[node][neighbor].get('weight', 1)
                    if random.random() < transmission_prob:
                        new_status[neighbor] += contamination_status[node] * transmission_prob
        contamination_status = new_status
    return contamination_status

final_contamination = simulate_contamination(G, contamination_status)

print("Final Contamination Levels:")
for node, status in final_contamination.items():
    print(f"{node}: {status:.2f}")

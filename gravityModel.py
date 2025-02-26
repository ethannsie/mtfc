import pandas as pd
import numpy as np
from math import radians, sin, cos, atan2, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

data = {
    'zone_id': range(1, 11),
    'population': np.random.randint(1000, 10000, size=10),
    'retail_revenue': np.random.randint(50000, 500000, size=10),
    'latitude': np.random.uniform(48.7, 48.8, size=10),
    'longitude': np.random.uniform(9.0, 9.1, size=10)
}

df = pd.DataFrame(data)
df.set_index('zone_id', inplace=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def create_distance_matrix(df):
    n = len(df)
    distance_matrix = pd.DataFrame(index=df.index, columns=df.index)
    for i in df.index:
        for j in df.index:
            distance_matrix.loc[i, j] = haversine(df.loc[i, 'latitude'], df.loc[i, 'longitude'],
                                                 df.loc[j, 'latitude'], df.loc[j, 'longitude'])
    return distance_matrix

distance_matrix = create_distance_matrix(df)

def intra_zonal_distance(area, num_retailers):
    if num_retailers == 0:
        return np.sqrt(area / np.pi)
    entity_density = num_retailers / area
    return 0.5 / np.sqrt(entity_density)

df['area'] = np.random.uniform(1, 10, size=len(df))
df['num_retailers'] = np.random.randint(1, 10, size=len(df))

for zone_id in df.index:
    area = df.loc[zone_id, 'area']
    num_retailers = df.loc[zone_id, 'num_retailers']
    distance_matrix.loc[zone_id, zone_id] = intra_zonal_distance(area, num_retailers)

def gravity_model(population, retail_revenue, distance_matrix, beta=0.1):
    n = len(population)
    flows = pd.DataFrame(index=population.index, columns=population.index)
    A_i = pd.Series(index=population.index, dtype='float64')
    B_j = pd.Series(index=population.index, dtype='float64')
    tolerance = 1e-6
    max_iterations = 100
    for iteration in range(max_iterations):
        for j in population.index:
            B_j[j] = 1.0 / sum([A_i[k] * retail_revenue[k] * np.exp(-beta * distance_matrix.loc[k, j]) for k in population.index])
        for i in population.index:
            A_i[i] = 1.0 / sum([B_j[l] * population[l] * np.exp(-beta * distance_matrix.loc[i, l]) for l in population.index])
        if iteration > 0 and np.max(np.abs(A_i.diff())) < tolerance:
            breakf
    for i in population.index:
        for j in population.index:
            flows.loc[i, j] = A_i[i] * retail_revenue[i] * B_j[j] * population[j] * np.exp(-beta * distance_matrix.loc[i, j])
    return flows

flows = gravity_model(df['population'], df['retail_revenue'], distance_matrix, beta=0.1)
flows = flows.apply(pd.to_numeric, errors='coerce')
plt.figure(figsize=(10, 8))
sns.heatmap(flows, annot=False, cmap="YlGnBu", fmt=".1f")
plt.title("Estimated Flows Between Zones")
plt.xlabel("Destination Zone")
plt.ylabel("Origin Zone")
plt.show()

G = nx.DiGraph()
threshold = flows.mean().mean()
for i in flows.index:
    for j in flows.index:
        if flows.loc[i, j] > threshold:
            G.add_edge(i, j, weight=flows.loc[i, j])

pos = nx.spring_layout(G)
plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10, alpha=0.7,
        arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("Network Graph of Flows (Above Threshold)")
plt.show()

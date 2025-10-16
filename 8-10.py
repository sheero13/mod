import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("my_graph1.tsv", sep="\t")
print(" Dataset loaded successfully!")
print(df.head())

G = nx.from_pandas_edgelist(df, source='source', target='target')
print("\nGraph created successfully!")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

print("\n Degree Centrality:")
for node, val in degree_centrality.items():
    print(f"{node}: {val:.3f}")

print("\n Betweenness Centrality:")
for node, val in betweenness_centrality.items():
    print(f"{node}: {val:.3f}")

print("\n Closeness Centrality:")
for node, val in closeness_centrality.items():
    print(f"{node}: {val:.3f}")

print("\n Shortest Paths Between Every Node Pair:")
shortest_paths = dict(nx.all_pairs_shortest_path(G))
for source, paths in shortest_paths.items():
    for target, path in paths.items():
        print(f"{source} → {target}: {path}")

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)  # Spring layout

# Node sizes proportional to degree centrality
node_sizes = [3000 * degree_centrality[node] for node in G.nodes()]
#de colors proportional to betweenness centrality No
node_colors = [betweenness_centrality[node] for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis,
                       node_size=node_sizes, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=9, font_color='black')

plt.title("Graph Visualization using Spring Layout\n(Size ~ Degree, Color ~ Betweenness)")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Betweenness Centrality")
plt.axis("off")
plt.tight_layout()
plt.show()

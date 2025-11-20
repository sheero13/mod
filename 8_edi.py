import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns # Imported but not used for analysis/plot here, kept for robustness

# --- 1. DATA LOADING ---

# Change the file name to the correct dataset
filepath = "my_graph3.tsv" 
sep = "\t"

# Assuming the file has column headers 'source' and 'target'
try:
    df = pd.read_csv(filepath, sep=sep)
    print("Dataset loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found. Please check the file path.")
    exit()

# --- 2. GRAPH CREATION ---
# Map the 'source' and 'target' columns to create the graph edges.
G = nx.from_pandas_edgelist(df, source='source', target='target')

print("\nGraph created successfully!")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# --- 3. CENTRALITY MEASURES AND SHORTEST PATHS ---

# a. Degree Centrality: Measures local importance based on number of connections.
degree_centrality = nx.degree_centrality(G)
# b. Betweenness Centrality: Measures global influence based on how often a node lies on the shortest path between other node pairs.
betweenness_centrality = nx.betweenness_centrality(G)
# c. Closeness Centrality: Measures efficiency/speed of information flow based on average distance to all other nodes.
closeness_centrality = nx.closeness_centrality(G)

print("\n--- Statistical Values ---")

print("\nDegree Centrality (Local Importance):")
for node, val in list(degree_centrality.items())[:5]: # Showing top 5 for brevity
    print(f"{node}: {val:.4f}")

print("\nBetweenness Centrality (Global Influence):")
for node, val in list(betweenness_centrality.items())[:5]:
    print(f"{node}: {val:.4f}")

print("\nCloseness Centrality (Efficiency):")
for node, val in list(closeness_centrality.items())[:5]:
    print(f"{node}: {val:.4f}")

# d. Shortest Path between Every Node Pair
print("\nShortest Paths Between Sample Node Pairs:")
shortest_paths = dict(nx.all_pairs_shortest_path(G))
# Print a few sample paths (you can iterate through all if needed)
sample_nodes = list(G.nodes())[:3]
for source in sample_nodes:
    for target in sample_nodes:
        if source != target:
            try:
                path = shortest_paths[source][target]
                print(f"{source} → {target}: {path}")
            except KeyError:
                 print(f"{source} → {target}: No path found (disconnected)")


# --- 4. VISUALIZATION (Spring Layout) ---
# 
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=0.1) # k=0.1 reduces node overlap

# Node sizes are proportional to degree centrality (scaled up for visibility)
node_sizes = [5000 * degree_centrality[node] for node in G.nodes()]
# Node colors represent betweenness centrality
node_colors = [betweenness_centrality[node] for node in G.nodes()]

# Draw nodes, scaling size and color based on centrality
nx.draw_networkx_nodes(G, pos, 
                       node_color=node_colors, 
                       cmap=plt.cm.viridis,
                       node_size=node_sizes, 
                       alpha=0.9,
                       linewidths=1.5, edgecolors='gray')
                       
# Draw edges
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.6)
# Draw labels only for the most central nodes (optional, helps readability for large graphs)
labels = {node: node for node, val in degree_centrality.items() if val > 0.1} 
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')

# Add a color bar for betweenness centrality
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
sm.set_array(node_colors)
plt.colorbar(sm, label="Betweenness Centrality")

plt.title("Graph Visualization using Spring Layout\n(Size ~ Degree, Color ~ Betweenness Centrality)")
plt.axis("off")
plt.tight_layout()
plt.show()

"""Create an interactive visualization of the knowledge graph."""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Path to output results
output_dir = Path("output")

print("🔨 Creating graph visualization...")

# Load data
entities = pd.read_parquet(output_dir / "create_final_entities.parquet")
relationships = pd.read_parquet(output_dir / "create_final_relationships.parquet")

# Create NetworkX graph
G = nx.Graph()

# Add nodes (entities)
for _, entity in entities.iterrows():
    G.add_node(
        entity["name"],
        type=entity.get("type", "unknown"),
        description=entity.get("description", "")[:100]  # Truncate description
    )

# Add edges (relationships)
for _, rel in relationships.iterrows():
    if rel["source"] in G.nodes and rel["target"] in G.nodes:
        G.add_edge(
            rel["source"],
            rel["target"],
            description=rel.get("description", "")
        )

print(f"✅ Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Limit to subgraph for readability (top nodes by degree)
if G.number_of_nodes() > 50:
    print("⚠️  Graph too large, displaying top 50 most connected nodes")
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
    G = G.subgraph([node for node, _ in top_nodes])

# Create visualization
plt.figure(figsize=(20, 15))

# Node positions with spring layout
pos = nx.spring_layout(G, k=2, iterations=50)

# Color by entity type
entity_types = nx.get_node_attributes(G, "type")
unique_types = list(set(entity_types.values()))
color_map = {t: plt.cm.Set3(i) for i, t in enumerate(unique_types)}
node_colors = [color_map.get(entity_types.get(node, "unknown"), "gray") for node in G.nodes()]

# Draw graph
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

plt.title("GraphRAG Knowledge Graph", fontsize=16, fontweight="bold")
plt.axis("off")
plt.tight_layout()

# Save
output_file = "knowledge_graph.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"✅ Graph saved to: {output_file}")

# Display
plt.show()

print("\n📊 Graph statistics:")
print(f"  - Nodes: {G.number_of_nodes()}")
print(f"  - Edges: {G.number_of_edges()}")
print(f"  - Density: {nx.density(G):.4f}")
print(f"  - Connected components: {nx.number_connected_components(G)}")


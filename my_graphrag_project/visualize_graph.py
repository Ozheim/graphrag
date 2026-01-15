"""
Visualisation du graphe de connaissances GraphRAG
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
output_dir = Path("output")

print("Chargement des données...")

# Charger les entités et relations
entities = pd.read_parquet(output_dir / "entities.parquet")
relationships = pd.read_parquet(output_dir / "relationships.parquet")

print(f"{len(entities)} entités chargées")
print(f"{len(relationships)} relations chargées")

# Afficher les premières entités
print("\nPremières entités extraites :")
print(entities[['title', 'type', 'description']].head(10).to_string())

# Afficher les premières relations
print("\nPremières relations extraites :")
print(relationships[['source', 'target', 'description']].head(10).to_string())

# Créer le graphe NetworkX
print("\nCréation du graphe...")
G = nx.Graph()

# Ajouter les nœuds (entités)
for _, entity in entities.iterrows():
    G.add_node(
        entity['title'],
        type=entity.get('type', 'unknown'),
        description=entity.get('description', '')[:100]  # Tronquer la description
    )

# Ajouter les arêtes (relations)
for _, rel in relationships.iterrows():
    G.add_edge(
        rel['source'],
        rel['target'],
        description=rel.get('description', '')[:100]
    )

print(f"Graphe créé : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

# Statistiques
print("\nStatistiques du graphe :")
print(f"  - Densité : {nx.density(G):.3f}")
print(f"  - Composantes connexes : {nx.number_connected_components(G)}")

# Top 10 des nœuds les plus connectés
degree_dict = dict(G.degree())
sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 des entités les plus connectées :")
for node, degree in sorted_nodes:
    print(f"  - {node}: {degree} connexions")

# Visualisation
print("\nCréation de la visualisation...")
plt.figure(figsize=(20, 16))

# Layout : spring_layout pour une disposition automatique
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Taille des nœuds proportionnelle au degré
node_sizes = [300 + degree_dict[node] * 200 for node in G.nodes()]

# Couleur par type
node_colors = []
for node in G.nodes():
    node_type = G.nodes[node].get('type', 'unknown')
    if node_type == 'organization':
        node_colors.append('#FF6B6B')  # Rouge
    elif node_type == 'person':
        node_colors.append('#4ECDC4')  # Cyan
    elif node_type == 'concept':
        node_colors.append('#95E1D3')  # Vert
    else:
        node_colors.append('#FFBE0B')  # Jaune

# Dessiner le graphe
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.7,
    edgecolors='black',
    linewidths=1.5
)

nx.draw_networkx_edges(
    G, pos,
    width=1.5,
    alpha=0.3,
    edge_color='gray'
)

nx.draw_networkx_labels(
    G, pos,
    font_size=9,
    font_weight='bold',
    font_color='black'
)

plt.title("Graphe de Connaissances - BAABI Documentation", fontsize=20, fontweight='bold')
plt.axis('off')
plt.tight_layout()

# Sauvegarder
output_file = "knowledge_graph.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nGraphe sauvegardé : {output_file}")

# Afficher
plt.show()

print("\nTerminé !")

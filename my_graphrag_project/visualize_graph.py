"""Visualize the GraphRAG knowledge graph."""

import pandas as pd
from pathlib import Path

# Path to output results
output_dir = Path("output")

print("=" * 60)
print("📊 GraphRAG KNOWLEDGE GRAPH")
print("=" * 60)

# 1. Entities (graph nodes)
entities_file = output_dir / "create_final_entities.parquet"
if entities_file.exists():
    entities = pd.read_parquet(entities_file)
    print(f"\n✅ ENTITIES: {len(entities)} entities found")
    print("\nFirst entities:")
    print(entities[["name", "type", "description"]].head(10))
    
    # Stats by type
    print("\n📈 Distribution by type:")
    print(entities["type"].value_counts())

# 2. Relationships (graph edges)
relationships_file = output_dir / "create_final_relationships.parquet"
if relationships_file.exists():
    relationships = pd.read_parquet(relationships_file)
    print(f"\n✅ RELATIONSHIPS: {len(relationships)} relationships found")
    print("\nFirst relationships:")
    print(relationships[["source", "target", "description"]].head(10))

# 3. Communities
communities_file = output_dir / "create_final_communities.parquet"
if communities_file.exists():
    communities = pd.read_parquet(communities_file)
    print(f"\n✅ COMMUNITIES: {len(communities)} communities found")
    print("\nFirst communities:")
    print(communities[["title"]].head(5))

# 4. Community reports
reports_file = output_dir / "create_final_community_reports.parquet"
if reports_file.exists():
    reports = pd.read_parquet(reports_file)
    print(f"\n✅ REPORTS: {len(reports)} reports generated")

print("\n" + "=" * 60)
print("To visualize graphically, use visualize_network.py")
print("=" * 60)


import os
import pandas as pd
import webbrowser
from pathlib import Path
from pyvis.network import Network

# --- CONFIGURATION & CONSTANTS ---
ARTIFACTS_DIR = Path("output") 
OUTPUT_HTML = "governance_graph.html"

STYLE_MAP = {
    "GOVERNANCE_ROLE":    {"color": "#3498db", "shape": "dot",      "label": "Rôle"},
    "DATA_CONCEPT":       {"color": "#2ecc71", "shape": "dot",      "label": "Concept"},
    "TECHNICAL_ASSET":    {"color": "#e67e22", "shape": "square",   "label": "Asset Technique"},
    "PROCESS_ACTIVITY":   {"color": "#9b59b6", "shape": "box",      "label": "Activité"},
    "DECISION_GATE":      {"color": "#e74c3c", "shape": "diamond",  "label": "Décision/Gate"},
    "REGULATION_POLICY":  {"color": "#f1c40f", "shape": "triangle", "label": "Règle/Politique"},
    "DOCUMENT_METHOD":    {"color": "#95a5a6", "shape": "text",     "label": "Document"},
    "UNKNOWN":            {"color": "#34495e", "shape": "dot",      "label": "Inconnu"}
}

def load_parquet_data(base_path: Path):
    """Charge les entités et relations depuis les fichiers Parquet GraphRAG."""
    ent_path = base_path / "create_final_entities.parquet"
    rel_path = base_path / "create_final_relationships.parquet"
    
    # Fallback pour structure de fichiers alternative
    if not ent_path.exists():
        ent_path, rel_path = base_path / "entities.parquet", base_path / "relationships.parquet"

    if not ent_path.exists():
        raise FileNotFoundError(f"Parquet files not found in {base_path}")

    ents = pd.read_parquet(ent_path)
    rels = pd.read_parquet(rel_path)
    
    # Nettoyage
    ents["type"] = ents["type"].fillna("UNKNOWN").str.upper().str.strip()
    ents["description"] = ents["description"].fillna("").astype(str)
    return ents, rels

def build_network(entities, relationships):
    """Construit l'objet PyVis Network avec la physique et les styles."""
    net = Network(height="95vh", width="100%", bgcolor="#1e1e1e", font_color="white", select_menu=True, filter_menu=True)
    net.force_atlas_2based(gravity=-100, central_gravity=0.01, spring_length=200, spring_strength=0.05, damping=0.4, overlap=0)

    # 1. Ajout des Nœuds
    for _, row in entities.iterrows():
        e_type = row["type"]
        style = STYLE_MAP.get(e_type, STYLE_MAP["UNKNOWN"])
        desc = row["description"].replace("\n", " ")
        tooltip = f"<b>{row['title']}</b><br><i>{e_type}</i><hr>{desc[:300]}..."

        net.add_node(
            row["title"],
            label=row["title"],
            title=tooltip,
            color=style["color"],
            shape=style["shape"],
            size=30 if style["shape"] in ["diamond", "text"] else 25,
            font={'size': 16, 'face': 'arial', 'color': 'white'}
        )

    # 2. Ajout des Relations
    valid_nodes = set(net.get_nodes())
    for _, row in relationships.iterrows():
        if row["source"] in valid_nodes and row["target"] in valid_nodes:
            net.add_edge(
                row["source"],
                row["target"],
                title=str(row.get("description", "")),
                color="#555555",
                width=1,
                arrows="to"
            )
    return net

def inject_custom_legend(html_path):
    """Injecte une légende CSS/HTML propre directement dans le fichier généré."""
    legend_items = ""
    for k, v in STYLE_MAP.items():
        legend_items += f"""
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display:inline-block; width:15px; height:15px; background-color:{v['color']}; margin-right:10px; border-radius:3px;"></span>
            <span style="font-size: 12px; color: #eee;">{v['label']}</span>
        </div>"""

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; background-color: rgba(30, 30, 30, 0.9); padding: 15px; border: 1px solid #555; border-radius: 8px; z-index: 1000; font-family: Arial, sans-serif;">
        <h4 style="margin-top: 0; margin-bottom: 10px; color: white; border-bottom: 1px solid #555; padding-bottom: 5px;">Légende Gouvernance</h4>
        {legend_items}
    </div></body>"""

    with open(html_path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0)
        f.write(content.replace('</body>', legend_html))
        f.truncate()

if __name__ == "__main__":
    try:
        print(f"--- Loading Data from {ARTIFACTS_DIR} ---")
        ents_df, rels_df = load_parquet_data(ARTIFACTS_DIR)
        
        print(f"--- Building Graph ({len(ents_df)} nodes, {len(rels_df)} edges) ---")
        network = build_network(ents_df, rels_df)
        
        print(f"--- Saving & Injecting UI to {OUTPUT_HTML} ---")
        network.save_graph(OUTPUT_HTML)
        inject_custom_legend(OUTPUT_HTML)
        
        webbrowser.open('file://' + os.path.realpath(OUTPUT_HTML))
        print("Done.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
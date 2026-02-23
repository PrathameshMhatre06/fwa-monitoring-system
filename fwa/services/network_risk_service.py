import networkx as nx
import pandas as pd
import sqlite3
from datetime import datetime

ALPHA = 0.6
BETA = 0.25
GAMMA = 0.15

def compute_network_risk(df, db_path):
    G = nx.Graph()

    # Build graph
    for _, row in df.iterrows():
        hospital = f"H_{row['hospital_id']}"
        doctor = f"D_{row['doctor_id']}"

        G.add_node(hospital, type="hospital")
        G.add_node(doctor, type="doctor")

        G.add_edge(hospital, doctor)

    # Base risk = fraud rate
    hospital_risk = df.groupby("hospital_id")["is_fraud"].mean().to_dict()
    doctor_risk = df.groupby("doctor_id")["is_fraud"].mean().to_dict()

    network_results = []

    for node in G.nodes():
        if node.startswith("H_"):
            entity_id = node
            base = hospital_risk.get(int(node.split("_")[1]), 0)
        else:
            entity_id = node
            base = doctor_risk.get(int(node.split("_")[1]), 0)

        neighbors = list(G.neighbors(node))

        # 1-hop risk
        one_hop_values = []
        for n in neighbors:
            if n.startswith("H_"):
                val = hospital_risk.get(int(n.split("_")[1]), 0)
            else:
                val = doctor_risk.get(int(n.split("_")[1]), 0)
            one_hop_values.append(val)

        one_hop = sum(one_hop_values)/len(one_hop_values) if one_hop_values else 0

        # 2-hop risk
        two_hop_nodes = set()
        for n in neighbors:
            two_hop_nodes.update(G.neighbors(n))
        two_hop_nodes.discard(node)

        two_hop_values = []
        for n in two_hop_nodes:
            if n.startswith("H_"):
                val = hospital_risk.get(int(n.split("_")[1]), 0)
            else:
                val = doctor_risk.get(int(n.split("_")[1]), 0)
            two_hop_values.append(val)

        two_hop = sum(two_hop_values)/len(two_hop_values) if two_hop_values else 0

        amplified = (
            ALPHA * base +
            BETA * one_hop +
            GAMMA * two_hop
        ) * 100  # scale to %

        network_results.append((
            entity_id,
            "hospital" if node.startswith("H_") else "doctor",
            base * 100,
            one_hop * 100,
            two_hop * 100,
            amplified,
            str(datetime.now())
        ))

    # Save to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for row in network_results:
        cursor.execute("""
    INSERT OR REPLACE INTO entity_network_metrics
    (entity_id, entity_type, base_risk, one_hop_risk,
     two_hop_risk, network_amplified_risk, last_updated)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", row)


    conn.commit()
    conn.close()

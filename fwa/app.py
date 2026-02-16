import streamlit as st
import pandas as pd
import joblib
import os
import shap
import sqlite3
import plotly.express as px
from datetime import datetime

from services.aggregation_service import get_aggregation_metrics
from services.velocity_service import get_velocity_metrics
from services.network_service import get_doctor_network_metrics
from services.entity_risk_service import calculate_entity_risk
from services.behavioral_service import get_behavioral_metrics
from services.benchmark_service import get_disease_deviation
from rules.rule_engine import evaluate_rules


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Enterprise Fraud Intelligence Console",
    layout="wide"
)

# ==========================================================
# PROJECT PATHS
# ==========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "fwa", "models", "xgb_fraud_model.pkl")
ANOMALY_PATH = os.path.join(PROJECT_ROOT, "fwa", "models", "isolation_forest.pkl")

# ---------------------------------------------
# DATABASE PATH (LOCAL + SERVER SAFE)
# ---------------------------------------------
if os.getenv("RENDER"):
    DB_PATH = "/var/data/fwa_claims.db"   # Render persistent disk
else:
    DB_PATH = os.path.join(PROJECT_ROOT, "fwa", "data", "fwa_claims.db")  # Local machine


ml_model = joblib.load(MODEL_PATH)
anomaly_model = joblib.load(ANOMALY_PATH)

RULE_WEIGHT = 0.5
ML_WEIGHT = 0.35
ANOMALY_WEIGHT = 0.15

# ==========================================================
# ENSURE CLAIMS TABLE HAS REQUIRED COLUMNS (SAFE MIGRATION)
# ==========================================================

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS claims (
    claim_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    hospital_id INTEGER,
    doctor_id INTEGER,
    claim_amount REAL,
    disease_code INTEGER,
    length_of_stay INTEGER,
    policy_age_days INTEGER,
    previous_claims_count INTEGER,
    claim_date TEXT,
    is_fraud INTEGER DEFAULT 0
)
""")

required_columns = {
    "final_score": "REAL",
    "risk_tier": "TEXT",
    "recommended_action": "TEXT",
    "is_fraud_flag": "INTEGER",
    "case_status": "TEXT DEFAULT 'OPEN'",
    "assigned_analyst": "TEXT",
    "investigation_notes": "TEXT"
}

cursor.execute("PRAGMA table_info(claims)")
existing_columns = [col[1] for col in cursor.fetchall()]

for column, col_type in required_columns.items():
    if column not in existing_columns:
        cursor.execute(f"ALTER TABLE claims ADD COLUMN {column} {col_type}")

conn.commit()
conn.close()

# ==========================================================
# SCORE FUNCTION (UNCHANGED LOGIC)
# ==========================================================
def score_claim(claim_dict):

    claim_date = claim_dict["claim_date"]

    aggregation = get_aggregation_metrics(
        claim_dict["patient_id"],
        claim_dict["hospital_id"],
        claim_date
    )

    velocity = get_velocity_metrics(
        claim_dict["patient_id"],
        claim_date
    )

    network = get_doctor_network_metrics(
        claim_dict["doctor_id"],
        claim_date
    )

    behavioral = get_behavioral_metrics(
        claim_dict["patient_id"],
        claim_dict["hospital_id"],
        claim_dict["claim_amount"],
        claim_date
    )

    disease_deviation_ratio = get_disease_deviation(
        claim_dict["disease_code"],
        claim_dict["claim_amount"]
    )

    behavioral["disease_deviation_ratio"] = disease_deviation_ratio

    rule_score, triggered = evaluate_rules(
        claim_dict,
        aggregation,
        velocity,
        network,
        behavioral,
        disease_deviation_ratio
    )

    ml_input = pd.DataFrame([{
        "claim_amount": claim_dict["claim_amount"],
        "disease_code": claim_dict["disease_code"],
        "length_of_stay": claim_dict["length_of_stay"],
        "policy_age_days": claim_dict["policy_age_days"],
        "previous_claims_count": claim_dict["previous_claims_count"],
        "cost_per_day": claim_dict["claim_amount"] / max(claim_dict["length_of_stay"], 1),
        "claim_policy_ratio": claim_dict["claim_amount"] / max(claim_dict["policy_age_days"], 1),
        "hospital_30day_total": aggregation["hospital_30day_total"],
        "patient_60day_count": aggregation["patient_60day_count"],
        "patient_7day_count": velocity["patient_7day_count"],
        "doctor_30day_total": network["doctor_30day_total"],
        "patient_deviation_ratio": behavioral["patient_deviation_ratio"],
        "hospital_deviation_ratio": behavioral["hospital_deviation_ratio"],
        "disease_deviation_ratio": disease_deviation_ratio
    }])

    ml_input = ml_input[ml_model.feature_names_in_]

    ml_proba = ml_model.predict_proba(ml_input)[0][1]
    ml_score = ml_proba * 100

    anomaly_input = ml_input[anomaly_model.feature_names_in_]
    anomaly_raw = anomaly_model.decision_function(anomaly_input)[0]
    anomaly_score = max((1 - anomaly_raw) * 50, 0)

    final_score = (
        RULE_WEIGHT * rule_score +
        ML_WEIGHT * ml_score +
        ANOMALY_WEIGHT * anomaly_score
    )
    # ---------------------------------
    # Component Contributions
    # ---------------------------------
    rule_component = RULE_WEIGHT * rule_score
    ml_component = ML_WEIGHT * ml_score
    anomaly_component = ANOMALY_WEIGHT * anomaly_score

    final_score = min(final_score, 100)

    if final_score < 30:
        risk = "LOW"
        action = "Auto Approve"
    elif final_score < 60:
        risk = "REVIEW"
        action = "Manual Review"
    elif final_score < 80:
        risk = "INVESTIGATE"
        action = "Fraud Analyst"
    else:
        risk = "CRITICAL"
        action = "Payment Hold"

    base_model = ml_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(ml_input)

    shap_df = pd.DataFrame({
        "Feature": ml_input.columns,
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    return (
        final_score,
        risk,
        action,
        triggered,
        shap_df,
        rule_component,
        ml_component,
        anomaly_component
    )



# ==========================================================
# MAIN VIEW SELECTOR
# ==========================================================
st.title("🛡 Enterprise Fraud Intelligence Console")

view = st.sidebar.radio(
    "Select View",
    [
        "Claim Scoring",
        "Risk Heatmap Dashboard",
        "Model Drift Monitoring",
        "Fraud Network Graph",
        "Fraud Contagion Simulation",
        "Case Management Console" # ✅ Added
    ]
)

# ==========================================================
# CLAIM SCORING VIEW
# ==========================================================
if view == "Claim Scoring":

    st.sidebar.header("Enter Claim Details")

    claim_id = st.sidebar.number_input("Claim ID", value=1001)
    patient_id = st.sidebar.number_input("Patient ID", value=1500)
    hospital_id = st.sidebar.number_input("Hospital ID", value=10)
    doctor_id = st.sidebar.number_input("Doctor ID", value=20)
    claim_amount = st.sidebar.number_input("Claim Amount", value=50000)
    disease_code = st.sidebar.number_input("Disease Code", value=3)
    length_of_stay = st.sidebar.number_input("Length of Stay", value=5)
    policy_age_days = st.sidebar.number_input("Policy Age (days)", value=300)
    previous_claims_count = st.sidebar.number_input("Previous Claims Count", value=2)
    claim_date = st.sidebar.date_input("Claim Date")

if st.sidebar.button("Score Claim"):

    claim_data = {
        "claim_id": claim_id,
        "patient_id": patient_id,
        "hospital_id": hospital_id,
        "doctor_id": doctor_id,
        "claim_amount": claim_amount,
        "disease_code": disease_code,
        "length_of_stay": length_of_stay,
        "policy_age_days": policy_age_days,
        "previous_claims_count": previous_claims_count,
        "claim_date": claim_date
    }

    (
        final_score,
        risk,
        action,
        triggered,
        shap_df,
        rule_component,
        ml_component,
        anomaly_component
    ) = score_claim(claim_data)

    # ==========================================================
    # AUTO SAVE CLAIM AFTER SCORING
    # ==========================================================

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO claims (
            claim_id,
            patient_id,
            hospital_id,
            doctor_id,
            claim_amount,
            disease_code,
            length_of_stay,
            policy_age_days,
            previous_claims_count,
            claim_date,
            final_score,
            risk_tier,
            recommended_action,
            is_fraud_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        claim_id,
        patient_id,
        hospital_id,
        doctor_id,
        claim_amount,
        disease_code,
        length_of_stay,
        policy_age_days,
        previous_claims_count,
        str(claim_date),
        float(final_score),
        risk,
        action,
        1 if final_score >= 60 else 0
    ))

    conn.commit()
    conn.close()

    st.success("✅ Claim scored and saved successfully.")
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Risk Score", f"{final_score:.2f}")
    col2.metric("Risk Tier", risk)
    col3.metric("Recommended Action", action)

    st.write("### Triggered Rules")
    st.write(triggered if triggered else "None")

    # ==========================================================
    # 📊 FRAUD SCORE DECOMPOSITION PANEL (NEW ADDITION)
    # ==========================================================

    st.markdown("---")
    st.subheader("📊 Fraud Score Decomposition")

    comp_col1, comp_col2, comp_col3 = st.columns(3)

    comp_col1.metric("Rule Engine Contribution", f"{rule_component:.2f}")
    comp_col2.metric("ML Model Contribution", f"{ml_component:.2f}")
    comp_col3.metric("Anomaly Model Contribution", f"{anomaly_component:.2f}")

    decomposition_df = pd.DataFrame({
        "Component": ["Rule Engine", "ML Model", "Anomaly Model"],
        "Contribution": [
            rule_component,
            ml_component,
            anomaly_component
        ]
    })

    st.bar_chart(decomposition_df.set_index("Component"))

    total_components = rule_component + ml_component + anomaly_component

    if total_components > 0:
        rule_pct = (rule_component / total_components) * 100
        ml_pct = (ml_component / total_components) * 100
        anomaly_pct = (anomaly_component / total_components) * 100

        st.markdown("### 🔎 Contribution Breakdown (%)")
        st.write(f"• Rule Engine: {rule_pct:.1f}%")
        st.write(f"• ML Model: {ml_pct:.1f}%")
        st.write(f"• Anomaly Model: {anomaly_pct:.1f}%")

    st.write("### 🔍 ML Explainability (SHAP)")
    st.dataframe(shap_df.head(10))
    st.bar_chart(shap_df.set_index("Feature")["SHAP Value"].head(10))

# ==========================================================
# RISK HEATMAP DASHBOARD
# ==========================================================
if view == "Risk Heatmap Dashboard":

    st.header("📊 Enterprise Risk Heatmap Intelligence")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()

    if df.empty:
        st.warning("No claims data available.")
    else:

        df["claim_date"] = pd.to_datetime(df["claim_date"])

        st.subheader("🏥 Hospital Fraud Concentration")

        hospital_matrix = (
            df.groupby(["hospital_id", "doctor_id"])
            .agg(
                total_claims=("claim_id", "count"),
                fraud_rate=("is_fraud", "mean"),
                exposure=("claim_amount", "sum")
            )
            .reset_index()
        )

        hospital_matrix["fraud_rate"] *= 100

        heatmap_fig = px.density_heatmap(
            hospital_matrix,
            x="hospital_id",
            y="doctor_id",
            z="fraud_rate",
            color_continuous_scale="reds",
            title="Fraud Rate Heatmap (Hospital vs Doctor)"
        )

        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("💰 Exposure Heatmap")

        exposure_fig = px.density_heatmap(
            hospital_matrix,
            x="hospital_id",
            y="doctor_id",
            z="exposure",
            color_continuous_scale="blues",
            title="Financial Exposure Heatmap"
        )

        st.plotly_chart(exposure_fig, use_container_width=True)

# ==========================================================
# MODEL DRIFT MONITORING
# ==========================================================

if view == "Model Drift Monitoring":

    st.header("🧠 Model Drift Monitoring Dashboard")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()

    if df.empty:
        st.warning("No historical claims available.")
    else:

        df["claim_date"] = pd.to_datetime(df["claim_date"])

        latest_date = df["claim_date"].max()

        last_30 = df[df["claim_date"] >= latest_date - pd.Timedelta(days=30)]
        prev_30 = df[
            (df["claim_date"] < latest_date - pd.Timedelta(days=30)) &
            (df["claim_date"] >= latest_date - pd.Timedelta(days=60))
        ]

        st.subheader("🚨 Fraud Rate Drift")

        fraud_last = last_30["is_fraud"].mean() * 100 if not last_30.empty else 0
        fraud_prev = prev_30["is_fraud"].mean() * 100 if not prev_30.empty else 0

        col1, col2 = st.columns(2)
        col1.metric("Last 30 Days Fraud %", f"{fraud_last:.2f}%")
        col2.metric("Previous 30 Days Fraud %", f"{fraud_prev:.2f}%")

        st.subheader("⚠ Drift Alert System")

        drift_flag = abs(fraud_last - fraud_prev)

        if drift_flag > 10:
            st.error("⚠ Significant Fraud Rate Drift Detected!")
        elif drift_flag > 5:
            st.warning("⚠ Moderate Drift Observed.")
        else:
            st.success("✅ Model Behavior Stable.")


# ==========================================================
# FRAUD NETWORK GRAPH (RISK AWARE VERSION)
# ==========================================================

if view == "Fraud Network Graph":

    import networkx as nx
    import plotly.graph_objects as go

    st.header("🕸 Risk-Aware Fraud Network Intelligence")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()

    if df.empty:
        st.warning("No claims data available.")
    else:

        # ----------------------------
        # Aggregate metrics
        # ----------------------------
        hospital_metrics = df.groupby("hospital_id").agg(
            exposure=("claim_amount", "sum"),
            fraud_rate=("is_fraud", "mean"),
            total_claims=("claim_id", "count")
        )

        doctor_metrics = df.groupby("doctor_id").agg(
            exposure=("claim_amount", "sum"),
            fraud_rate=("is_fraud", "mean"),
            total_claims=("claim_id", "count")
        )

        # ----------------------------
        # Build Graph
        # ----------------------------
        G = nx.Graph()

        for _, row in df.iterrows():
            hospital_node = f"H_{row['hospital_id']}"
            doctor_node = f"D_{row['doctor_id']}"

            if not G.has_node(hospital_node):
                G.add_node(
                    hospital_node,
                    node_type="hospital",
                    exposure=hospital_metrics.loc[row["hospital_id"], "exposure"],
                    fraud_rate=hospital_metrics.loc[row["hospital_id"], "fraud_rate"]
                )

            if not G.has_node(doctor_node):
                G.add_node(
                    doctor_node,
                    node_type="doctor",
                    exposure=doctor_metrics.loc[row["doctor_id"], "exposure"],
                    fraud_rate=doctor_metrics.loc[row["doctor_id"], "fraud_rate"]
                )

            if G.has_edge(hospital_node, doctor_node):
                G[hospital_node][doctor_node]["weight"] += 1
            else:
                G.add_edge(hospital_node, doctor_node, weight=1)

        pos = nx.spring_layout(G, k=0.5, seed=42)

        edge_x = []
        edge_y = []
        edge_width = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_width.append(edge[2]["weight"])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)

            fraud_rate = node[1]["fraud_rate"] * 100
            exposure = node[1]["exposure"]

            node_color.append(fraud_rate)
            node_size.append(max(exposure / 1000000, 10))

            node_text.append(
                f"{node[0]}<br>"
                f"Fraud Rate: {fraud_rate:.2f}%<br>"
                f"Exposure: ₹{exposure:,.0f}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale="Reds",
                color=node_color,
                size=node_size,
                colorbar=dict(
                    title="Fraud Rate %"
                ),
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Fraud Network (Node Size = Exposure | Color = Fraud Rate)",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            plot_bgcolor="#0E1117",
                            paper_bgcolor="#0E1117",
                            font=dict(color="white")
                        ))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📌 Interpretation Guide")
        st.markdown("""
        - 🔴 Dark Red Nodes → High Fraud Rate Entities  
        - 🟠 Medium Shade → Moderate Risk  
        - 🟢 Light Shade → Low Risk  
        - Bigger Node → Higher Financial Exposure  
        - Dense Clusters → Possible Fraud Rings  
        """)

        # ==========================================================
        # 🔥 CENTRALITY INTELLIGENCE (NEW ADDITION)
        # ==========================================================

        st.markdown("---")
        st.header("🧠 Centrality Intelligence")

        # Degree Centrality (connection importance)
        degree_centrality = nx.degree_centrality(G)

        # Betweenness Centrality (bridge detection)
        betweenness_centrality = nx.betweenness_centrality(G)

        centrality_df = pd.DataFrame({
            "Entity": list(degree_centrality.keys()),
            "Degree Centrality": list(degree_centrality.values()),
            "Betweenness Centrality": [
                betweenness_centrality[node] for node in degree_centrality.keys()
            ]
        })

        centrality_df = centrality_df.sort_values(
            by="Degree Centrality",
            ascending=False
        )

        st.subheader("📊 Top Influential Entities (Network Hubs)")
        st.dataframe(centrality_df.head(10))


        # ==========================================================
        # 🚨 NETWORK RISK SCORE
        # ==========================================================

        avg_fraud_rate = df["is_fraud"].mean() * 100
        avg_exposure = df["claim_amount"].mean()

        network_risk_score = (
            (avg_fraud_rate * 0.6) +
            (avg_exposure / 100000 * 0.4)
        )

        st.subheader("🌐 Network Risk Score")

        col1, col2, col3 = st.columns(3)

        col1.metric("Average Fraud Rate %", f"{avg_fraud_rate:.2f}%")
        col2.metric("Average Claim Exposure", f"₹{avg_exposure:,.0f}")
        col3.metric("Network Risk Score", f"{network_risk_score:.2f}")


        # ==========================================================
        # 🔎 HIGH-RISK BRIDGE DETECTION
        # ==========================================================

        st.subheader("🔎 Critical Bridge Entities (High Betweenness)")

        bridge_entities = centrality_df.sort_values(
            by="Betweenness Centrality",
            ascending=False
        ).head(5)

        st.dataframe(bridge_entities)


        # ==========================================================
        # 📌 ADVANCED ANALYTICS SUMMARY
        # ==========================================================

        st.markdown("### 📌 Network Interpretation Guide (Advanced)")
        st.markdown("""
        - 🔴 High Degree Centrality → Entity connected to many others (possible hub)
        - 🔥 High Betweenness → Entity acting as fraud bridge
        - 📈 High Network Risk Score → System-wide fraud escalation
        - 🕸 Dense sub-clusters → Potential coordinated fraud rings
        """)

# ==========================================================
# FRAUD CONTAGION SIMULATION (1-Hop Propagation)
# ==========================================================

if view == "Fraud Contagion Simulation":

    st.header("🧨 Fraud Risk Contagion Simulation")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()

    if df.empty:
        st.warning("No claims data available.")
    else:

        df["fraud_rate"] = df.groupby("hospital_id")["is_fraud"].transform("mean")

        hospital_list = df["hospital_id"].unique()

        selected_hospital = st.selectbox(
            "Select High-Risk Hospital to Simulate Contagion",
            hospital_list
        )

        # -----------------------------
        # Base Hospital Metrics
        # -----------------------------
        base_df = df[df["hospital_id"] == selected_hospital]

        base_exposure = base_df["claim_amount"].sum()
        base_fraud_rate = base_df["is_fraud"].mean() * 100

        # -----------------------------
        # Connected Doctors
        # -----------------------------
        connected_doctors = base_df["doctor_id"].unique()

        doctor_df = df[df["doctor_id"].isin(connected_doctors)]

        propagated_exposure = doctor_df["claim_amount"].sum()
        propagated_fraud_rate = doctor_df["is_fraud"].mean() * 100

        # -----------------------------
        # Metrics Panel
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Base Hospital Exposure", f"₹{base_exposure:,.0f}")
        col2.metric("Base Hospital Fraud %", f"{base_fraud_rate:.2f}%")
        col3.metric("Connected Doctors", len(connected_doctors))

        st.markdown("---")

        col4, col5 = st.columns(2)

        col4.metric("Propagated Exposure", f"₹{propagated_exposure:,.0f}")
        col5.metric("Network Fraud %", f"{propagated_fraud_rate:.2f}%")

        st.markdown("---")

        # -----------------------------
        # Simple Risk Interpretation
        # -----------------------------
        if propagated_fraud_rate > 50:
            st.error("⚠ Severe Contagion Risk: High systemic fraud exposure detected.")
        elif propagated_fraud_rate > 30:
            st.warning("⚠ Moderate Contagion Risk: Network-level exposure rising.")
        else:
            st.success("✅ Low systemic contagion risk.")

        st.markdown("### 📌 Interpretation")
        st.markdown("""
        - Base Exposure → Financial impact of selected hospital  
        - Propagated Exposure → Total exposure across connected doctors  
        - Network Fraud % → Systemic risk indicator  
        - Higher propagation = Possible organized fraud behavior  
        """)

# ==========================================================
# CASE MANAGEMENT CONSOLE
# ==========================================================

if view == "Case Management Console":

    st.header("🗂 Fraud Case Management Console")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM claims", conn)

    if df.empty:
        st.warning("No claims available.")
        conn.close()
    else:
        # Show only risky cases (score >= 60)
        case_df = df[df["final_score"] >= 60].copy()

        # CHECK: If there are no high-risk cases after filtering
        if case_df.empty:
            st.success("No high-risk cases at the moment.")
            conn.close()
        else:
            # ----------------------------------------
            # Priority Scoring Logic
            # ----------------------------------------
            case_df["priority_score"] = (
                case_df["final_score"] * 0.6 +
                case_df["claim_amount"] / 100000 * 0.4
            )

            def assign_priority(row):
                if row["priority_score"] > 80:
                    return "🔴 CRITICAL"
                elif row["priority_score"] > 60:
                    return "🟠 HIGH"
                else:
                    return "🟡 MEDIUM"

            case_df["priority_level"] = case_df.apply(assign_priority, axis=1)

            # Sort by latest claim_date so the UI is consistent
            case_df = case_df.sort_values("claim_date", ascending=False)

            # ----------------------------------------
            # UI Display Logic
            # ----------------------------------------
            st.subheader("📋 High Risk Cases")

            st.dataframe(case_df[[
                "claim_id",
                "claim_date",  # Added this so users see the date being sorted
                "priority_level",
                "priority_score",
                "final_score",
                "risk_tier",
                "recommended_action",
                "case_status",
                "assigned_analyst"
            ]])

            selected_claim = st.selectbox(
                "Select Claim ID to Manage",
                case_df["claim_id"],
                index=0  # Auto-selects the first item (most recent date)
            )

            # Locate the specific row for the selected claim
            selected_row = case_df[case_df["claim_id"] == selected_claim].iloc[0]

            st.markdown("---")
            st.subheader(f"🛠 Update Case: {selected_claim}")

            # Form Layout for better organization
            status_options = ["OPEN", "UNDER_REVIEW", "CLOSED_FRAUD", "CLOSED_CLEAN"]
            current_status = selected_row.get("case_status", "OPEN")
            
            try:
                status_index = status_options.index(current_status)
            except ValueError:
                status_index = 0

            new_status = st.selectbox(
                "Update Case Status",
                status_options,
                index=status_index
            )

            analyst_name = st.text_input(
                "Assign Analyst",
                value=selected_row.get("assigned_analyst", "")
            )

            notes = st.text_area(
                "Investigation Notes",
                value=selected_row.get("investigation_notes", "")
            )

            if st.button("💾 Save Case Update"):
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE claims
                    SET case_status = ?,
                        assigned_analyst = ?,
                        investigation_notes = ?
                    WHERE claim_id = ?
                """, (
                    new_status,
                    analyst_name,
                    notes,
                    selected_claim
                ))

                conn.commit()
                st.success(f"Case {selected_claim} updated successfully!")
                st.rerun()

            conn.close()
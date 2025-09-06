import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import cvxpy as cp
import numpy as np
import pandas as pd
import joblib
import json

# Save the trained model
model = joblib.load("Model/xgb_model.pkl")

with open("Model/model_metrics.json", "r") as f:
    metrics = json.load(f)

with open("Model/model_params.json", "r") as f:
    params = json.load(f)

mse = metrics["mse"]
r2 = metrics["r2"]
accuracy = metrics["accuracy"]
pred_df = pd.read_csv("Model/predictions.csv")
feature_importance = pd.read_csv("Model/feature_importance.csv")

# Extract key params only (to keep it minimal in UI)
key_params = {
    "n_estimators": params["n_estimators"],
    "max_depth": params["max_depth"],
    "learning_rate": params["learning_rate"],
    "subsample": params["subsample"],
    "colsample_bytree": params["colsample_bytree"],
    "objective": params["objective"]
}


df = pd.read_csv("Data/Processed_Resource_utilization.csv")

# Page Title
st.set_page_config(page_title="IEEE Dashboard", layout="wide")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🤖 Prediction", "⚙️ Optimization", "💡 Insights"])

# --- Visualization Tab ---
with tab1:
    st.header("Visualization")

    # --- Time-Series Trends ---
    st.subheader("Resource Trends Over Time")

    fig = px.line(df, x='timestamp', y='Resource Allocation',
                  title='Resource Allocation Over Time', width=800, height=400)
    st.plotly_chart(fig)

    fig = px.line(df, x='timestamp', y='cpu_utilization',
                  title='CPU Utilization Over Time', width=800, height=400)
    st.plotly_chart(fig)

    fig = px.line(df, x='timestamp', y='memory_usage',
                  title='Memory Usage Over Time', width=800, height=400)
    st.plotly_chart(fig)

    fig = px.line(df, x='timestamp', y='storage_usage',
                  title='Storage Usage Over Time', width=800, height=400)
    st.plotly_chart(fig)

    # --- Correlation Matrix ---
    st.subheader("Correlation Analysis")
    corr_matrix = df[['workload', 'Resource Allocation', 'cpu_utilization', 'memory_usage', 'storage_usage']].corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale='RdBu',
        showscale=True
    )
    fig.update_layout(width=500, height=400, title_text="Correlation Matrix", title_x=0.5)
    st.plotly_chart(fig)

    # --- Scatterplots ---
    st.subheader("Resource Allocation vs Metrics")
    metrics = ['cpu_utilization', 'memory_usage', 'storage_usage', 'workload']
    for metric in metrics:
        fig = px.scatter(
            df, x=metric, y='Resource Allocation', trendline="ols",
            labels={'Resource Allocation': 'Resource Allocation', metric: metric},
            title=f"Resource Allocation vs {metric}", width=500, height=400
        )
        st.plotly_chart(fig)

    # --- Storage Usage Distribution ---
    st.subheader("Distribution Analysis")
    fig = px.histogram(
        df, x="storage_usage", nbins=30, color_discrete_sequence=['skyblue'],
        title="Distribution of Storage Usage", width=500, height=400
    )
    fig.add_vline(x=df["storage_usage"].max(), line_color="green",
                annotation_text=f"Max: {df['storage_usage'].max()}", annotation_position="top right")
    st.plotly_chart(fig)
    st.write("Max Storage Usage:", df["storage_usage"].max())
    st.write("95th percentile:", df["storage_usage"].quantile(0.95))

    # --- CPU Utilization Distribution ---
    fig = px.histogram(
        df, x="cpu_utilization", nbins=30, color_discrete_sequence=['skyblue'],
        title="Distribution of CPU Utilization", width=500, height=400
    )
    fig.add_vline(x=df["cpu_utilization"].max(), line_color="green",
                annotation_text=f"Max: {df['cpu_utilization'].max()}", annotation_position="top right")
    st.plotly_chart(fig)
    st.write("Max CPU Utilization:", df["cpu_utilization"].max())
    st.write("95th percentile:", df["cpu_utilization"].quantile(0.95))

    # --- Memory Usage Distribution ---
    fig = px.histogram(
        df, x="memory_usage", nbins=30, color_discrete_sequence=['skyblue'],
        title="Distribution of Memory Usage", width=500, height=400
    )
    fig.add_vline(x=85, line_color="red", annotation_text="85 (cap)", annotation_position="top left")
    fig.add_vline(x=df["memory_usage"].max(), line_color="green",
                annotation_text=f"Max: {df['memory_usage'].max()}", annotation_position="top right")
    st.plotly_chart(fig)
    st.write("Max memory observed:", df["memory_usage"].max())
    st.write("95th percentile:", df["memory_usage"].quantile(0.95))
    st.write("Values above 85:", (df["memory_usage"] > 85).sum())

    # --- Workload Distribution ---
    fig = px.histogram(df, x='workload', nbins=50, color_discrete_sequence=['skyblue'],
                    title="Workload Distribution", width=500, height=400)
    st.plotly_chart(fig)
    st.write(df['workload'].describe())


# --- Prediction Tab ---
with tab2:
    st.header("Prediction Results")

    st.markdown(f"""
    **Model Used:** XGBoost Regressor  

    **Key Parameters:**  
    - n_estimators = {key_params['n_estimators']}  
    - max_depth = {key_params['max_depth']}  
    - learning_rate = {key_params['learning_rate']}  
    - subsample = {key_params['subsample']}  
    - colsample_bytree = {key_params['colsample_bytree']}  
    - objective = {key_params['objective']}  

    **Preprocessing:**  
    - StandardScaler applied to features  
    - Train-test split: 80% train, 20% test  
    - Feature engineering: lag features, rolling mean/std, expanding mean
    """)

    st.subheader("Model Performance")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**Accuracy (model.score):** {accuracy:.4f}")
    st.subheader("Predicted vs Actual Workload")
    fig = px.line(pred_df, x="timestamp", y=["Actual", "Predicted"],
                labels={"value": "Workload", "timestamp": "Time"},
                title="Workload Prediction Over Time", width=800, height=400)
    st.plotly_chart(fig)

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h",
                title="Feature Importance", width=600, height=400)
    st.plotly_chart(fig)

# --- Optimization Tab ---
with tab3:
    st.header("Resource Allocation Optimization")

    # Load dataset
    df = pd.read_csv("Predicted_Resource_utilization.csv")

    y_pred = df['predicted_workload'].values[:50]  
    cpu = df["cpu_utilization"].values[:50]
    memory = df["memory_usage"].values[:50]
    storage = df["storage_usage"].values[:50]

    T = len(y_pred)
    resources = ["small", "medium", "large"]
    R = len(resources)

    x = cp.Variable((T, R), integer=True)

    alloc = cp.sum(x, axis=1)

    capacity_limit = 0.7 * cpu + 0.5 * memory + 0.7 * storage

    objective = cp.Maximize(cp.sum(alloc))

    # Constraints
    constraints = [
        alloc <= y_pred,         # cannot allocate more than predicted workload
        alloc <= capacity_limit, # cannot exceed resource capacities
        x >= 0                   # non-negative allocation
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIP, verbose=True)

    # Display results
    if x.value is not None:
        x_opt = np.rint(x.value).astype(int)
        alloc_df = pd.DataFrame(x_opt, columns=resources)
        alloc_df["total_allocation"] = alloc_df[resources].sum(axis=1)
        alloc_df["predicted_demand"] = y_pred
        alloc_df["served_demand"] = np.minimum(alloc_df["total_allocation"], y_pred)

        st.subheader("Optimized Allocation (first 10 rows)")
        st.dataframe(alloc_df.head(10))

        st.write("Total Served Demand:", alloc_df["served_demand"].sum())

        # Optional: Plot allocations
        import plotly.express as px
        fig = px.line(alloc_df, y=resources + ["served_demand", "predicted_demand"], 
                      labels={"value": "Units", "index": "Time Step"},
                      title="Resource Allocation vs Predicted Workload")
        st.plotly_chart(fig)

    else:
        st.error(f"Solver failed: {prob.status}")

# --- Insights Tab ---
with tab4:
    st.header("Operational Insights")

    st.markdown("""
    **Key Observations from Data:**
    - Workload frequently reaches 100, causing **very high CPU utilization**, while resource allocation remains below actual demand.
    - **Memory usage stays mostly under 85%**, showing it is not the limiting factor.
    - Storage usage patterns indicate that sustained performance depends on efficient data handling.

    **Implications:**
    - The service relies **heavily on CPU and storage capacity** rather than memory.
    - CPU is the primary bottleneck during peak demand, directly affecting workload handling.
    - Advanced and optimized CPUs significantly improve the system’s ability to process workloads.
    - Underutilized memory suggests potential for rebalancing resources.

    **Recommendations for Cloud Environment:**
    1. Implement **CPU-focused auto-scaling** to dynamically add processing power during workload spikes.
    2. Invest in **newer, higher-performance CPUs** to boost workload throughput and efficiency.
    3. Optimize **storage allocation and I/O performance** to complement CPU improvements.
    4. Reallocate workloads or select **CPU- and storage-optimized instance types** rather than memory-heavy ones.
    5. Use **predictive workload forecasting** for proactive provisioning of CPU and storage resources.
    6. Set up **intelligent monitoring and alerts** to detect CPU saturation early and trigger scaling.
    7. Control costs by combining **reserved instances for baseline CPU/storage demand** with auto-scaling for peaks.
    """)
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

# load model
model = joblib.load("workload_model.pkl")

df = pd.read_csv("Processed_Resource_utilization.csv")

# Page Title
st.set_page_config(page_title="IEEE Dashboard", layout="wide")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🤖 Prediction", "⚙️ Optimization", "💡 Insights"])

# --- Visualization Tab ---
with tab1:
    st.header("Visualization")

   # --- Correlation Matrix ---
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
    metrics = ['cpu_utilization', 'memory_usage', 'storage_usage', 'workload']
    for metric in metrics:
        fig = px.scatter(
            df, x=metric, y='Resource Allocation', trendline="ols",
            labels={'Resource Allocation': 'Resource Allocation', metric: metric},
            title=f"Resource Allocation vs {metric}", width=500, height=400
        )
        st.plotly_chart(fig)

    # --- Storage Usage Distribution ---
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
    st.header("Predict Workload")


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

    capacity_limit = 0.7 * cpu + 0.5 * memory + 0.3 * storage

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
    **Observations from data:**
    - CPU utilization is often very high when workload reaches 100, but resource allocation is still less than demand.
    - Memory usage is mostly below 85%, indicating spare capacity.

    **Implications:**
    - CPU is a bottleneck during peak demand.
    - Memory is underutilized, leading to inefficient resource usage.

    **Recommendations for Cloud Environment:**
    1. Implement **CPU auto-scaling** to dynamically allocate resources based on predicted workload or CPU thresholds.
    2. Utilize underused memory by adjusting workloads or selecting better-balanced instance types.
    3. Use **predictive allocation** leveraging workload forecasts to provision resources ahead of peak demand.
    4. Introduce **resource prioritization** to ensure critical workloads receive CPU first.
    5. Set up **monitoring and alerts** to scale resources proactively.
    6. Optimize costs by combining reserved instances with autoscaling for peak loads.
    """)
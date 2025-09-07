# IEEE Project
# Resource Utilization Prediction, Optimization, and Visualization For Cloud Services (Supply Chain Problem)

This repository contains a comprehensive pipeline for analyzing, predicting, optimizing, and visualizing resource utilization in a computing environment. The project processes historical data on CPU, memory, storage, and workload, uses machine learning to predict future workload, applies optimization techniques for efficient resource allocation, and provides an interactive dashboard for data visualization and insights. The project is built using Python, with Jupyter notebooks for data processing, modeling, optimization, and visualization, and a Streamlit app for the user interface.

## Project Overview

- **Data Preprocessing**: Cleans and enriches data by handling missing values, outliers, and creating features like lag and rolling statistics.
- **Prediction**: Uses XGBoost to forecast workload based on historical resource metrics.
- **Optimization**: Employs convex optimization (CVXPY) to allocate resources efficiently under capacity constraints.
- **Visualization**: Provides detailed visualizations of trends, correlations, distributions, and model performance using Matplotlib, Seaborn, and Plotly.
- **Dashboard**: Interactive Streamlit app with tabs for visualization, predictions, optimization, and operational insights.

This project is designed for managing resource-intensive systems, such as cloud environments, to optimize performance and minimize resource waste.

### Key Technologies
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost
- **Optimization**: CVXPY (with SCIP solver)
- **Dashboard**: Streamlit

## Features

- **Data Preprocessing**: Handles missing values, outliers, and feature engineering (e.g., lag features, rolling mean/std, expanding mean).
- **Workload Prediction**: XGBoost model with MSE ~28.22 and R² ~0.772 for workload forecasting.
- **Resource Optimization**: Maximizes served demand while respecting CPU, memory, and storage constraints.
- **Visualizations**:
  - Time-series plots for CPU, memory, storage, and resource allocation trends.
  - Correlation matrix heatmap for resource metrics.
  - Scatter plots of resource allocation vs. metrics with trendlines.
  - Histograms for distribution analysis of CPU, memory, storage, and workload.
  - Feature importance bar chart for model insights.
  - Actual vs. predicted workload comparison.
  - Optimized allocation vs. predicted workload plot.
- **Interactive Dashboard**: Streamlit app with tabs for:
  - Visualization: Trends, correlations, and distributions.
  - Prediction: Model performance and predictions.
  - Optimization: Resource allocation results.
  - Insights: Actionable recommendations for resource management.
- **Data Export**: Saves processed datasets, model, predictions, and metrics.

## Prerequisites

- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - cvxpy
  - streamlit
  - plotly
  - joblib
  - scipy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/youssef6765/IEEE-CAMP.git
   cd IEEE-CAMP/Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset (`Resource_utilization.csv`) is in the `Data/` folder.

## Usage

### Step 1: Data Preprocessing
Run the preprocessing notebook to clean and engineer features:
```bash
jupyter notebook 1-Data_preprocessing.ipynb
```
This generates:
- `Data/Processed_Train_Resource_utilization.csv` (training data)
- `Data/Processed_Test_Resource_utilization.csv` (testing data)
- `Data/Processed_Resource_utilization.csv` (full processed data)

### Step 2: Model Training and Prediction
Train the XGBoost model for workload prediction:
```bash
jupyter notebook 3-Model.ipynb
```
This saves:
- `Model/xgb_model.pkl`: Trained XGBoost model
- `Model/model_metrics.json`: MSE, R², and accuracy
- `Model/model_params.json`: Model hyperparameters
- `Model/predictions.csv`: Actual vs. predicted workload
- `Model/feature_importance.csv`: Feature importance scores
- `Data/Predicted_Resource_utilization.csv`: Dataset with predictions

### Step 3: Optimization
Run the optimization notebook to allocate resources:
```bash
jupyter notebook 4-Optmization.ipynb
```
This uses CVXPY to optimize resource allocation for the first 50 timesteps and generates a plot comparing workload, historical, and optimized allocations.

### Step 4: Launch the Dashboard
Run the Streamlit app for interactive visualization and analysis:
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`. The dashboard includes:
- **Visualization Tab**:
  - Time-series plots for resource allocation, CPU, memory, and storage.
  - Correlation matrix heatmap.
  - Scatter plots with trendlines for resource allocation vs. metrics.
  - Histograms for CPU, memory, storage, and workload distributions.
- **Prediction Tab**:
  - Model performance metrics (MSE, R², accuracy).
  - Line plot of actual vs. predicted workload.
  - Feature importance bar chart.
- **Optimization Tab**:
  - Table of optimized allocations (first 10 rows).
  - Plot of optimized allocation vs. predicted workload.
- **Insights Tab**:
  - Key observations and recommendations for cloud resource management.

## Project Structure

```
IEEE-CAMP/
├── Project/
│   ├── Data/
│   │   ├── Resource_utilization.csv                # Raw dataset
│   │   ├── Processed_Train_Resource_utilization.csv # Processed training data
│   │   ├── Processed_Test_Resource_utilization.csv  # Processed testing data
│   │   ├── Processed_Resource_utilization.csv       # Full processed data
│   │   └── Predicted_Resource_utilization.csv       # Dataset with predictions
│   ├── Model/
│   │   ├── xgb_model.pkl                           # Trained XGBoost model
│   │   ├── model_metrics.json                      # Model performance metrics
│   │   ├── model_params.json                       # Model hyperparameters
│   │   ├── predictions.csv                         # Actual vs. predicted workload
│   │   └── feature_importance.csv                  # Feature importance scores
│   ├── 1-Data_preprocessing.ipynb                  # Preprocessing notebook
│   ├── 3-Model.ipynb                               # Modeling and prediction notebook
│   ├── 4-Optmization.ipynb                         # Optimization notebook
│   ├── app.py                                      # Streamlit dashboard
│   ├── requirements.txt                            # Dependencies
│   └── README.md                                   # This file
```

## Visualizations

The project provides a range of visualizations to explore the data and results:
- **Time-Series Trends**: Show resource allocation, CPU, memory, and storage usage over time.
- **Correlation Heatmap**: Displays relationships between workload, resource allocation, CPU, memory, and storage.
- **Scatter Plots**: Illustrate resource allocation vs. individual metrics with OLS trendlines.
- **Distribution Histograms**: Analyze the distribution of CPU, memory, storage, and workload with key statistics (e.g., max, 95th percentile).
- **Prediction Plots**: Compare actual vs. predicted workload over time.
- **Feature Importance**: Bar chart showing the contribution of each feature to the XGBoost model.
- **Optimization Plot**: Compares predicted workload, historical allocation, and optimized allocation.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please adhere to PEP8 standards and include relevant tests or documentation updates.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [XGBoost](https://xgboost.readthedocs.io/) for robust prediction modeling.
- [CVXPY](https://www.cvxpy.org/) and SCIP for optimization.
- [Streamlit](https://streamlit.io/) for the interactive dashboard.
- [Plotly](https://plotly.com/) and [Seaborn](https://seaborn.pydata.org/) for advanced visualizations.
- IEEE CAMP for providing the inspiration and context for this project.

For questions or issues, please open a GitHub issue or contact the repository owner.

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

import matplotlib
matplotlib.use("agg")

def generate_iid_data(num_samples, num_features, num_classes, random_seed, imbalance):
    np.random.seed(random_seed)

    # Generate feature matrix
    features = np.random.normal(0, 1, size=(num_samples, num_features))

    # Generate target variable
    if not imbalance:
        targets = np.random.choice(range(num_classes), size=num_samples)
    else:
        alpha = [0.2] * num_classes
        p = np.random.dirichlet(alpha)
        targets = np.random.choice(range(num_classes), size=num_samples, p=p)

    df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(num_features)])
    df['Target'] = targets

    return df

def plot_feature_histograms(df, features):
    n_features = len(features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)

def calculate_chi_square(df, feature_col, target_col):
    df['temp_bin'] = pd.qcut(df[feature_col], q=5, duplicates='drop', labels=False)
    contingency_table = pd.crosstab(df['temp_bin'], df[target_col])
    _, p, _, _ = chi2_contingency(contingency_table)
    df.drop(columns='temp_bin', inplace=True)
    return p

def run():
    st.set_page_config(page_title="IID Data", page_icon="📊", layout="centered")
    st.title("IID Data Generation")

    st.write("""
        Generate an **IID (Independent and Identically Distributed)** dataset.

        - Features are drawn from independent **Normal(0, 1)** distributions.
        - The **target** can have between 2 and 10 classes.
        - Optionally, introduce **class imbalance**.
        - A **random seed** ensures reproducibility.
    """)

    # Sidebar inputs
    st.sidebar.header("Generation Settings")
    num_samples = st.sidebar.number_input("Number of samples", 100, 100000, 1000)
    num_features = st.sidebar.slider("Number of features", 2, 10, 2)
    num_classes = st.sidebar.slider("Number of classes (target)", 2, 10, 2)
    random_seed = st.sidebar.number_input("Random Seed", 0, 999999, 42)
    imbalance = st.sidebar.checkbox("Introduce class imbalance?", value=False)

    if st.button("Generate IID Data"):
        df = generate_iid_data(num_samples, num_features, num_classes, random_seed, imbalance)
        st.session_state["iid_dataset"] = df  # Store dataset in session state

    # Ensure dataset persists across refreshes
    if "iid_dataset" in st.session_state:
        df = st.session_state["iid_dataset"]

        st.subheader("Preview of IID Dataset")
        st.dataframe(df.head())

        # CSV Download without Page Refresh
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Dataset as CSV", csv, "IID_dataset.csv", "text/csv")

        # Target distribution
        st.subheader("Target Distribution")
        fig_t, ax_t = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Target', data=df, ax=ax_t)
        ax_t.set_title("Target Class Distribution")
        fig_t.tight_layout()
        st.pyplot(fig_t)

        # Feature histograms
        st.subheader("Feature Distributions")
        plot_feature_histograms(df, df.columns[:-1])

        # Correlation matrix
        if num_features > 1:
            st.subheader("Correlation Matrix (Features Only)")
            corr_matrix = df.iloc[:, :-1].corr()
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            fig.tight_layout()
            st.pyplot(fig)
        
        # Explanation of correlation matrix
            st.markdown(
                """
                **How to Interpret the Correlation Matrix**  
                - Each cell shows the **pairwise Pearson correlation** between two features.
                - **Values range** from -1 to +1:
                  - **+1**: perfect positive linear relationship
                  - **-1**: perfect negative linear relationship
                  - **0**: no linear relationship
                - **Color Scale**: Red = positive correlation, Blue = negative correlation.
                - Larger absolute values suggest stronger linear relationships.
                - Correlation **does not** capture non-linear dependencies.
                """
            )

        # Mutual Information
        st.subheader("Mutual Information (Features vs. Target)")
        X, y = df.iloc[:, :-1].values, df['Target'].values
        mi_scores = mutual_info_classif(X, y, discrete_features=False)
        mi_df = pd.DataFrame({'Feature': df.columns[:-1], 'MI Score': mi_scores}).sort_values('MI Score', ascending=False)

        st.dataframe(mi_df)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='MI Score', y='Feature', data=mi_df, ax=ax)
        ax.set_title("Mutual Information Scores")
        fig.tight_layout()
        st.pyplot(fig)
        
        st.markdown(
            """
            **What is Mutual Information (MI)?**  
            MI measures how much knowing a feature reduces uncertainty about the target.  
            - **Higher MI** indicates a stronger relationship between the feature and the target.  
            - **Lower MI** suggests the feature and target are closer to independent.
            """
        )

        # Chi-Square Test
        st.subheader("Chi-Square Tests (Feature vs. Target)")
        chi_results = [(feat, calculate_chi_square(df, feat, 'Target')) for feat in df.columns[:-1]]
        chi_df = pd.DataFrame(chi_results, columns=['Feature', 'p-value'])

        st.dataframe(chi_df)
        st.markdown("""
        - **Low p-value (< 0.05)** → Feature **depends** on the target.
        - **High p-value** → Feature is **independent** of the target.
        """)

if __name__ == "__main__":
    run()
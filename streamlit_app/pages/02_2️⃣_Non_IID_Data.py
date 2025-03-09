import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

import matplotlib
matplotlib.use("agg")

def generate_non_iid_data(num_samples, num_features, num_classes, random_seed, imbalance):
    """
    Generate Non-IID data where:
      - ALL features depend on the target class.
      - Each (feature, class) pair has a distinct mean AND scale.
    """
    np.random.seed(random_seed)

    # Generate target variable
    if not imbalance:
        targets = np.random.choice(range(num_classes), size=num_samples)
    else:
        alpha = [0.2] * num_classes
        p = np.random.dirichlet(alpha)
        targets = np.random.choice(range(num_classes), size=num_samples, p=p)

    # Create random mean & scale offsets for each (feature, class)
    offsets_mean = np.random.uniform(-2, 2, size=(num_features, num_classes))
    offsets_scale = np.random.uniform(0.5, 2.0, size=(num_features, num_classes))

    # Generate feature matrix efficiently
    features = np.zeros((num_samples, num_features))
    for f in range(num_features):
        features[:, f] = np.random.normal(
            loc=[offsets_mean[f, label] for label in targets],
            scale=[offsets_scale[f, label] for label in targets]
        )

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
    # MUST BE FIRST Streamlit command
    st.set_page_config(page_title="Non-IID Data", page_icon="📊", layout="centered")
    st.title("Non-IID Data Generation")

    st.write("""
        Generate a **Non-IID (Non-Independent and Identically Distributed)** dataset.

        - Features depend on the **target class** by adjusting both their **mean** and **scale**.
        - The **target** can have between 2 and 10 classes.
        - Optionally, introduce **class imbalance**.
        - A **random seed** ensures reproducibility.
    """)

    # Sidebar inputs
    st.sidebar.header("Generation Settings")
    num_samples = st.sidebar.number_input("Number of samples", 100, 100000, 1000)
    num_features = st.sidebar.slider("Number of features", 2, 10, 5)
    num_classes = st.sidebar.slider("Number of classes (target)", 2, 10, 3)
    random_seed = st.sidebar.number_input("Random Seed", 0, 999999, 42)
    imbalance = st.sidebar.checkbox("Introduce class imbalance?", value=False)

    if st.button("Generate Non-IID Data"):
        df = generate_non_iid_data(num_samples, num_features, num_classes, random_seed, imbalance)
        st.session_state["non_iid_dataset"] = df  # Store dataset in session state

    # Ensure dataset persists across refreshes
    if "non_iid_dataset" in st.session_state:
        df = st.session_state["non_iid_dataset"]

        st.subheader("Preview of Non-IID Dataset")
        st.dataframe(df.head())

        # CSV Download without Page Refresh
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Dataset as CSV", csv, "Non_IID_dataset.csv", "text/csv")

        # Target distribution
        st.subheader("Target Class Distribution")
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
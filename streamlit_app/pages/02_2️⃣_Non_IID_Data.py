import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

import matplotlib
matplotlib.use("agg")

def non_iid_dataset(num_samples, num_features, num_classes, random_seed, imbalance):
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

    # Build the feature matrix
    data = np.zeros((num_samples, num_features))
    for i in range(num_samples):
        label = targets[i]
        for f in range(num_features):
            mean = offsets_mean[f, label]
            scale = offsets_scale[f, label]
            data[i, f] = np.random.normal(loc=mean, scale=scale)

    df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(num_features)])
    df['Target'] = targets

    return df

def plot_feature_histograms(df, features):
    n_features = len(features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)

def calculate_chi_square(df, feature_col, target_col):
    df['temp_bin'] = pd.qcut(df[feature_col], q=5, duplicates='drop', labels=False)
    contingency_table = pd.crosstab(df['temp_bin'], df[target_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    df.drop(columns='temp_bin', inplace=True)
    return p

def run():
    # MUST BE FIRST Streamlit command
    st.set_page_config(
	    page_title="Non IID Data",
	    page_icon="📊",
	    layout="centered",
    )
    st.title("Non-IID Data Generation")

    st.write(
        """
        In this page, you can generate a **Non-IID (Non-Independent and Identically Distributed)** dataset.

        - All features depend on the **target class** by adjusting both their **mean** and **scale**.
        - You can specify the **number of features** (2 to 10).
        - The **target** can have between 2 and 10 classes.
        - Optionally, you can introduce **class imbalance** (otherwise it's balanced).

        **About the Random Seed**:
        - A random seed ensures **reproducibility** of your data.  
        - Keeping the same seed yields the same random data each time.
        - Changing the seed will give you a different random dataset.
        """
    )


    # Sidebar inputs
    st.sidebar.header("Generation Settings")
    num_samples = st.sidebar.number_input("Number of samples", 100, 100000, 1000)
    num_features = st.sidebar.slider("Number of features", 2, 10, 2)
    num_classes = st.sidebar.slider("Number of classes (target)", 2, 10, 2)
    random_seed = st.sidebar.number_input(
        "Random Seed", 0, 999999, 42,
        help="Use a fixed seed for reproducible random data."
    )
    imbalance = st.sidebar.checkbox("Introduce class imbalance?", value=False)

    if st.button("Generate Non-IID Data"):
        df = non_iid_dataset(
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            random_seed=random_seed,
            imbalance=imbalance
        )
        st.session_state["non_iid_dataset"] = df

        if "non_iid_dataset" in st.session_state:
            df = st.session_state["non_iid_dataset"]

            st.subheader("Preview of Non IID Dataset")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Full Dataset as CSV",
                data=csv,
                file_name='Non_IID_dataset.csv',
                mime='text/csv'
            )

        # Target distribution
        st.subheader("Target Class Distribution")
        fig_t, ax_t = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Target', data=df, ax=ax_t)
        ax_t.set_title("Target Class Distribution")
        fig_t.tight_layout()
        st.pyplot(fig_t)

        # Feature histograms
        st.subheader("Feature Distributions")
        feature_cols = [col for col in df.columns if col.startswith("Feature_")]
        plot_feature_histograms(df, feature_cols)

        # Correlation matrix
        if len(feature_cols) > 1:
            st.subheader("Correlation Matrix (Features Only)")
            corr_matrix = df[feature_cols].corr()
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",  # Show correlation with 2 decimals
                cmap='coolwarm',
                center=0,
                ax=ax
            )
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
        X = df[feature_cols].values
        y = df['Target'].values
        mi_scores = mutual_info_classif(X, y, discrete_features=False)

        mi_df = pd.DataFrame({
            'Feature': feature_cols,
            'MI Score': mi_scores
        }).sort_values('MI Score', ascending=False)
        st.dataframe(mi_df)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='MI Score', y='Feature', data=mi_df, ax=ax)
        ax.set_title("Mutual Information Scores")
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **Interpretation**:  
        With per-class means and scales, **all** features tend to show
        a strong mutual information with the target.
        """)

        # Chi-Square
        st.subheader("Chi-Square Tests (Feature vs. Target)")
        chi_results = []
        for feat in feature_cols:
            p_val = calculate_chi_square(df, feat, 'Target')
            chi_results.append((feat, p_val))

        chi_df = pd.DataFrame(chi_results, columns=['Feature', 'p-value'])
        st.dataframe(chi_df)

        st.markdown("""
        **Interpretation**:  
        - Low p-values (< 0.05) for every feature confirm each one
          depends on the target in a statistically significant way.
        """)

if __name__ == "__main__":
    run()


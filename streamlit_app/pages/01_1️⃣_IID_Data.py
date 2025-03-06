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
    """
    Generate IID synthetic data:
      - Each feature is drawn from a standard normal distribution, independently.
      - The target variable has a user-defined number of classes (2..10).
      - If imbalance=True, we generate a skewed distribution for the target.
      - Otherwise, it's evenly distributed among all classes.
    """
    np.random.seed(random_seed)

    # Generate feature matrix
    feature_list = []
    for _ in range(num_features):
        feat = np.random.normal(loc=0, scale=1, size=num_samples)
        feature_list.append(feat)
    X = np.column_stack(feature_list)

    # Generate target variable
    if not imbalance:
        # Balanced distribution
        targets = np.random.choice(range(num_classes), size=num_samples)
    else:
        # Imbalanced with a Dirichlet-based skew
        alpha = [0.2] * num_classes
        p = np.random.dirichlet(alpha)
        targets = np.random.choice(range(num_classes), size=num_samples, p=p)

    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(num_features)])
    df['Target'] = targets

    return df

def plot_feature_histograms(df, features):
    """
    Plot histograms + KDE for each feature in a grid, avoiding overlapping subplots.
    """
    n_features = len(features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')

    # Hide any unused subplots (if there's an odd number of features)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)

def calculate_chi_square(df, feature_col, target_col):
    """
    Perform chi-square test for independence between a numerical feature and a categorical target.
    - Bin the feature into 5 quantiles
    - Create a contingency table with the target
    """
    df['temp_bin'] = pd.qcut(df[feature_col], q=5, duplicates='drop', labels=False)
    contingency_table = pd.crosstab(df['temp_bin'], df[target_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    df.drop(columns='temp_bin', inplace=True)
    return p

def run():
# MUST BE FIRST Streamlit command
    st.set_page_config(
	    page_title="IID Data",
	    page_icon="📊",
	    layout="centered",
    )
    st.title("IID Data Generation")

    st.write(
        """
        In this page, you can generate an **IID (Independent and Identically Distributed)** dataset.
        
        - All features are drawn from independent **Normal(0, 1)** distributions.
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
    num_samples = st.sidebar.number_input(
        "Number of samples",
        min_value=100,
        max_value=100000,
        value=1000
    )
    num_features = st.sidebar.slider(
        "Number of features",
        min_value=2,
        max_value=10,
        value=2
    )
    num_classes = st.sidebar.slider(
        "Number of classes (target)",
        min_value=2,
        max_value=10,
        value=2
    )
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        help="Use a fixed seed for reproducible random data. Changing it yields a different dataset."
    )
    imbalance = st.sidebar.checkbox("Introduce class imbalance?", value=False)

    if st.button("Generate IID Data"):
        # Generate the data
        df = generate_iid_data(
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            random_seed=random_seed,
            imbalance=imbalance
        )

        st.subheader("Preview of IID Dataset")
        st.dataframe(df.head())

        # Plot target distribution
        st.subheader("Target Distribution")
        fig_t, ax_t = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Target', data=df, ax=ax_t)
        ax_t.set_title("Target Class Distribution")
        fig_t.tight_layout()
        st.pyplot(fig_t)

        # Plot histograms of features
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


        # Mutual Information (feature vs target)
        st.subheader("Mutual Information (Features vs. Target)")
        X = df[feature_cols].values
        y = df['Target'].values
        mi_scores = mutual_info_classif(X, y, discrete_features=False)

        mi_df = pd.DataFrame({
            'Feature': feature_cols,
            'MI Score': mi_scores
        }).sort_values('MI Score', ascending=False)

        st.write(mi_df)
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

        # Chi-Square tests (each feature vs target)
        st.subheader("Chi-Square Tests (Feature vs. Target)")
        st.markdown("Testing whether each numerical feature is independent of the target.")

        chi_results = []
        for feat in feature_cols:
            p_val = calculate_chi_square(df, feat, 'Target')
            chi_results.append((feat, p_val))

        chi_df = pd.DataFrame(chi_results, columns=['Feature', 'p-value'])
        st.write(chi_df)

        st.markdown(
            """
            **Interpretation**:
            - A low p-value (below 0.05) suggests the feature is likely dependent on the target.
            - A high p-value suggests independence.
            """
        )

if __name__ == "__main__":
    run()


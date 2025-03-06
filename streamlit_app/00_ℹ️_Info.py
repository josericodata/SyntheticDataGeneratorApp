import streamlit as st

def run():
    # MUST BE FIRST Streamlit command
    st.set_page_config(
	    page_title="Synthetic Data Generator",
	    page_icon="📊",
	    layout="centered",
    )
    st.title("Synthetic Data Generator: Overview & Motivation")

    st.markdown(
        """
        ## What is This App?
        Welcome to the **Synthetic Data Generator**!  
        This Streamlit application helps you create synthetic datasets under two settings:
        1. **IID (Independent and Identically Distributed)**  
        2. **Non-IID (where features or distributions depend on the target)**  

        You can select the generation mode from the **sidebar**, adjust parameters (number of samples, features, classes, random seed, etc.), and quickly download or analyse the resulting data.

        ## Why Synthetic Data?
        - Synthetic datasets allow you to explore, prototype, or test algorithms **without** handling real, often sensitive data.
        - By customising parameters such as class imbalance, number of features, or distributions, you can simulate scenarios tailored to your needs.
        - **Free** and ready-to-use: you can just generate, explore, and export as needed for any project.
        - From personal experience: during my **MSc project**, I struggled to find appropriate synthetic datasets online. Creating your own synthetic data becomes a reliable, flexible solution, ensuring you always have data that fits your specific research or testing needs.

        ## IID vs. Non-IID
        1. **IID (Independent and Identically Distributed)**
           - Each feature is drawn from the **same type of distribution** (often Normal).
           - Features (and the target) are **independent** of each other.
           - In practice, we expect **low correlations**, low mutual information with the target, and chi-square tests indicating independence.

        2. **Non-IID**
           - At least one feature **depends on** the target (e.g., shifting its distribution based on the target class).
           - Another feature might be **skewed** or drawn from a different distribution (e.g., exponential).
           - Typically shows higher correlation, higher mutual information, or chi-square tests indicating dependence.

        ## How We Validate IID vs. Non-IID
        - **Correlation Matrix**: Checks how linearly related each pair of features is.
        - **Mutual Information (MI)**: Measures how much knowing a feature **reduces uncertainty** about the target. 
          - Higher MI = stronger dependency.
        - **Chi-Square Test**: 
          - Bins each feature and tests independence with the target classes.
          - A **low p-value** (< 0.05) suggests dependence; a **high p-value** suggests independence.

        Together, these tests help confirm whether your generated data truly reflects the **IID** or **Non-IID** setting you intend.

         ## Python Libraries Used
         This app uses several essential Python libraries for data generation, analysis, and visualisation:

         - **Streamlit**:  
            Provides the interactive web interface to easily generate, visualise, and explore synthetic datasets.

         - **NumPy**:  
            Used for generating random numbers and handling numerical operations, which is the core of creating both IID and Non-IID synthetic data.

         - **Pandas**:  
            Manages the generated data in structured DataFrames, making it easy to manipulate and display the datasets.

         - **Matplotlib & Seaborn**:  
            These libraries are responsible for visualising the feature distributions, target distributions, and correlation heatmaps.

         - **Scikit-learn**:  
            Provides statistical tools like **Mutual Information (MI)**, which helps to assess how much each feature depends on the target.

         - **SciPy**:  
            Used for statistical testing, particularly the **Chi-Square Test**, which evaluates the independence of each feature relative to the target.

        ---
        **Next Steps**:
        1. Use the **sidebar** to navigate:
           - **IID Data** generation page
           - **Non-IID Data** generation page
        2. Adjust the sliders and inputs as needed.
        3. Generate, visualise, and interpret your synthetic data!

        **Enjoy generating your free synthetic datasets!**
        """
    )

if __name__ == "__main__":
    run()


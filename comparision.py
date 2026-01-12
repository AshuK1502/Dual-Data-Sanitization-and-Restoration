import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

# Normalizing dataset
def normalize_data(df):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    return normalized_data, scaler

# for single sanitizatiion adn restoration
@st.cache_data
def ecdo_single(data, iterations=50, population_size=10):
    start_time = time.time()
    n, d = data.shape
    population = np.random.uniform(-0.05, 0.05, (population_size, d)) 
    best_solution = population[0].copy()
    best_fitness = float('inf')

    for i in range(iterations):
        for j in range(population_size):
            # generating random solutions
            new_solution = population[j] + np.random.uniform(-0.02, 0.02, d) * (best_solution - population[j])
            new_solution = np.clip(new_solution, -0.05, 0.05)  

            # sanitizating and restoring
            sanitized_data = data + new_solution
            restored_data = sanitized_data - new_solution

            fitness = np.mean((data - restored_data) ** 2)  

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_solution.copy()

        # updatw th population with hybridization
        population = np.clip(population + 0.5 * (best_solution - population), -0.05, 0.05)
    
    execution_time = time.time() - start_time

    return best_solution, execution_time

# for dual key santization and restoration
@st.cache_data
def ecdo_algorithm(data, iterations=50, population_size=10):
    start_time = time.time()
    n, d = data.shape
    #  two populations for key1 (additive) and key2 (multiplicative)
    population_key1 = np.random.uniform(-0.05, 0.05, (population_size, d))
    population_key2 = np.random.uniform(-0.05, 0.05, (population_size, d))
    
    best_key1 = population_key1[0].copy()
    best_key2 = population_key2[0].copy()
    best_fitness = float('inf')

    for i in range(iterations):
        for j in range(population_size):
            # updating kwy1
            new_key1 = population_key1[j] + np.random.uniform(-0.02, 0.02, d) * (best_key1 - population_key1[j])
            new_key1 = np.clip(new_key1, -0.05, 0.05)
            
            # updating kwy2
            new_key2 = population_key2[j] + np.random.uniform(-0.02, 0.02, d) * (best_key2 - population_key2[j])
            new_key2 = np.clip(new_key2, -0.05, 0.05)

            #  dual sanitization
            sanitized_data = data + new_key1  # 1st saniti
            sanitized_data = sanitized_data * (1 + new_key2)  #2nd saniti
            
            # dual restoration
            restored_data = sanitized_data / (1 + new_key2)  # 1st restore
            restored_data = restored_data - new_key1  #2nd restore

            fitness = np.mean((data - restored_data) ** 2)

            if fitness < best_fitness:
                best_fitness = fitness
                best_key1 = new_key1.copy()
                best_key2 = new_key2.copy()

        # updating  populations
        population_key1 = np.clip(population_key1 + 0.5 * (best_key1 - population_key1), -0.05, 0.05)
        population_key2 = np.clip(population_key2 + 0.5 * (best_key2 - population_key2), -0.05, 0.05)

    execution_time = time.time() - start_time

    return best_key1, best_key2, execution_time

# --- Key Space Calculation 
def calculate_key_space(key_range=0.05, dimensions=1, precision=1e-6):
    values_per_dim = int((2 * key_range) / precision)
    return values_per_dim ** dimensions


def main():
    st.title("Comparing Single and Dual Sanitization and Restoration")
    
    # uploading dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values='?')
        df.dropna(inplace=True)
        st.write("Original Dataset:", df.head())

        # normalixing the dataset
        normalized_data, scaler = normalize_data(df)
        st.write("Normalized Data (after scaling):", normalized_data[:5])

        st.subheader("Single Sanitization Process")
        # single sanitization
        optimal_key_single, time_single = ecdo_single(normalized_data)  #  generating  key
        sanitized_data_single = normalized_data + optimal_key_single  # sanitizing the data 
        sanitized_data_single = np.clip(sanitized_data_single, 0.0, 1.0)  #  keeping data within (0,1)

        sanitized_df_single = pd.DataFrame(sanitized_data_single, columns=df.columns)
        st.write("Sanitized Dataset:", sanitized_df_single.head())

        sanitized_csv_single = sanitized_df_single.to_csv(index=False).encode('utf-8')
        st.download_button("Download  Single Sanitized Dataset", sanitized_csv_single, "sanitized_data.csv", "text/csv",key="single_sanitized_data")

        optimal_key_df_single = pd.DataFrame([optimal_key_single], columns=df.columns)
        st.write("Optimal Key:", optimal_key_df_single)
        optimal_key_csv_single = optimal_key_df_single.to_csv(index=False).encode('utf-8')
        st.download_button("Download Optimal Single Key", optimal_key_csv_single, "optimal_key.csv", "text/csv",key="single_optimal_key")

        st.subheader("Data Restoration")
        sanitized_file = st.file_uploader("Upload the Single Sanitized Data CSV", type=["csv"])
        key_file = st.file_uploader("Upload the Optimal Single Key CSV", type=["csv"])

        if sanitized_file is not None and key_file is not None:
            sanitized_df_single = pd.read_csv(sanitized_file)
            optimal_key_df_single = pd.read_csv(key_file)

            # restoring data
            restored_data = sanitized_df_single.to_numpy() - optimal_key_df_single.to_numpy()
            restored_data = scaler.inverse_transform(restored_data)  # denormalize

            restored_df = pd.DataFrame(restored_data, columns=df.columns)
            st.write("Restored Dataset:", restored_df.head())

            restored_csv = restored_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Restored Dataset", restored_csv, "restored_data.csv", "text/csv",key="single_restored_data" )

            original_normalized = scaler.transform(df)  
            restored_normalized = scaler.transform(restored_df) 

            restoration_accuracy_single = 1 - np.mean(np.abs(restored_normalized - original_normalized))
            restoration_mse_single = np.mean((restored_normalized - original_normalized) ** 2)

            st.write(f"📌 **Restoration Accuracy:** {restoration_accuracy_single:.6f}")
            st.write(f"📌 **Restoration MSE:** {restoration_mse_single:.6f}")

            key_space_single = calculate_key_space(dimensions=normalized_data.shape[1])




            # Dual Sanitization
        st.subheader("Dual Sanitization Process")
        key1, key2, time_dual= ecdo_algorithm(normalized_data)
        
        sanitized_data = normalized_data + key1  # 1st sanitization 
        sanitized_data = sanitized_data * (1 + key2)  # 2nd sanitization 
        sanitized_data = np.clip(sanitized_data, 0.0, 1.0)  # keeping within [0,1]

        sanitized_df = pd.DataFrame(sanitized_data, columns=df.columns)
        st.write(" Sanitized Dataset:", sanitized_df.head())

        sanitized_csv = sanitized_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Sanitized Dataset", sanitized_csv, "sanitized_data.csv", "text/csv",key="dual_sanitized_data" )

        st.write("Optimal Keys:")
        col1, col2 = st.columns(2)
        with col1:
            key1_df = pd.DataFrame([key1], columns=df.columns)
            st.write("Additive Key (Key1):", key1_df)
            key1_csv = key1_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Key1", key1_csv, "additive_key.csv", "text/csv",key="dual_key1")
        with col2:
            key2_df = pd.DataFrame([key2], columns=df.columns)
            st.write("Multiplicative Key (Key2):", key2_df)
            key2_csv = key2_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Key2", key2_csv, "multiplicative_key.csv", "text/csv",key="dual_key2")

        # restoring
        st.subheader("Dual Restoration Process")
        sanitized_file = st.file_uploader("Upload Sanitized Data CSV", type=["csv"])
        key1_file = st.file_uploader("Upload Additive Key (Key1) CSV", type=["csv"])
        key2_file = st.file_uploader("Upload Multiplicative Key (Key2) CSV", type=["csv"])

        if sanitized_file and key1_file and key2_file:
            sanitized_df = pd.read_csv(sanitized_file)
            key1_df = pd.read_csv(key1_file)
            key2_df = pd.read_csv(key2_file)

            # dual restoring
            restored_data = sanitized_df.to_numpy() / (1 + key2_df.to_numpy())  # 1st restoration 
            restored_data = restored_data - key1_df.to_numpy()  # 2nd restoration 
            
            restored_data = scaler.inverse_transform(restored_data)
            restored_df = pd.DataFrame(restored_data, columns=df.columns)
            
            st.write("Restored Dataset:", restored_df.head())
            restored_csv = restored_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Restored Dataset", restored_csv, "restored_data.csv", "text/csv",key="dual_restored_data")

            original_normalized = scaler.transform(df)
            restored_normalized = scaler.transform(restored_df)
            
            restoration_accuracy_dual = 1 - np.mean(np.abs(restored_normalized - original_normalized))
            restoration_mse_dual = np.mean((restored_normalized - original_normalized) ** 2)
            
            st.success(f"📌 **Restoration Accuracy:** {restoration_accuracy_dual:.6f}")
            st.success(f"📌 **Restoration MSE:** {restoration_mse_dual:.6f}")

            key_space_dual = calculate_key_space(dimensions=normalized_data.shape[1]) ** 2 

                    # --- Comparison Table ---
        st.subheader(" Comparison Results")
        comparison_df = pd.DataFrame({
            "Metric": ["Restoration Accuracy","Restoration MSE", "Key Space Size", "Execution Time (s)"],
            "Single Sanitization": [f"{restoration_accuracy_single:.6f}",f"{restoration_mse_single:.6f}", f"{key_space_single:.2e}", f"{time_single:.4f}"],
            "Dual Sanitization": [f"{restoration_accuracy_dual:.6f}",f"{restoration_mse_dual:.6f}", f"{key_space_dual:.2e}", f"{time_dual:.4f}"],
        })
        st.table(comparison_df)

        # --- Interpretation ---
        st.markdown("""
        ###  Conclusion
        - **Dual sanitization has a MUCH larger key space** (squared complexity), making brute-force attacks harder.
        - **Restoration accuracy (MSE) is similar** in both methods when keys are optimized.
        - **Dual sanitization is slower** due to two-key optimization.
        - **Use dual sanitization for high-security needs**, single for speed.
        """)


if __name__ == "__main__":
    main()
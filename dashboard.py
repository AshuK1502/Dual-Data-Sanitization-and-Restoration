import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Normalizing dataset
def normalize_data(df):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    return normalized_data, scaler

# Hybrid ECDO (EFO + DSOA ) algorithm
def ecdo_algorithm(data, iterations=50, population_size=10):
    n, d = data.shape
    population = np.random.uniform(-0.05, 0.05, (population_size, d))  # Smaller key range
    best_solution = population[0].copy()
    best_fitness = float('inf')

    for i in range(iterations):
        for j in range(population_size):
            # generating solution with bounded randomness
            new_solution = population[j] + np.random.uniform(-0.02, 0.02, d) * (best_solution - population[j])
            new_solution = np.clip(new_solution, -0.05, 0.05)  

            # sanitize-restore process for fitness calculation
            sanitized_data = data + new_solution
            restored_data = sanitized_data - new_solution

            # calculatign fitness 
            fitness = np.mean((data - restored_data) ** 2)  

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_solution.copy()

        # updating population with hybridization
        population = np.clip(population + 0.5 * (best_solution - population), -0.05, 0.05)

    return best_solution

#Streamlit code
def main():
    st.title("Dual Data Sanitization & Restoration using ECDO")
    
    # uploading dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values='?')
        df.dropna(inplace=True)
        st.write("Original Dataset:", df.head())

        # NOrmalization
        normalized_data, scaler = normalize_data(df)
        st.write("Normalized Data (after scaling):", normalized_data[:5])

        # sanitization
        optimal_key = ecdo_algorithm(normalized_data)  #  generating optimal key
        sanitized_data = normalized_data + optimal_key  # Sanitizing the data with key
        sanitized_data = np.clip(sanitized_data, 0.0, 1.0)  #  restricting data within (0,1)

        # converting sanitized data to dataframe
        sanitized_df = pd.DataFrame(sanitized_data, columns=df.columns)
        st.write("Sanitized Dataset:", sanitized_df.head())

        sanitized_csv = sanitized_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Sanitized Dataset", sanitized_csv, "sanitized_data.csv", "text/csv")

        optimal_key_df = pd.DataFrame([optimal_key], columns=df.columns)
        st.write("Optimal Key:", optimal_key_df)
        optimal_key_csv = optimal_key_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Optimal Key", optimal_key_csv, "optimal_key.csv", "text/csv")

        # upload sanitized dataset and optimal key for restoration
        st.subheader("Data Restoration")
        sanitized_file = st.file_uploader("Upload the Sanitized Data CSV", type=["csv"])
        key_file = st.file_uploader("Upload the Optimal Key CSV", type=["csv"])

        if sanitized_file is not None and key_file is not None:
            sanitized_df = pd.read_csv(sanitized_file)
            optimal_key_df = pd.read_csv(key_file)

            # restoring data
            restored_data = sanitized_df.to_numpy() - optimal_key_df.to_numpy()
            restored_data = scaler.inverse_transform(restored_data)  # denormalize

            restored_df = pd.DataFrame(restored_data, columns=df.columns)
            st.write("Restored Dataset:", restored_df.head())

            restored_csv = restored_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Restored Dataset", restored_csv, "restored_data.csv", "text/csv")

            # calculating restoration metrics
            original_normalized = scaler.transform(df)  
            restored_normalized = scaler.transform(restored_df) 

            restoration_accuracy = 1 - np.mean(np.abs(restored_normalized - original_normalized))
            restoration_mse = np.mean((restored_normalized - original_normalized) ** 2)

            st.write(f"📌 **Restoration Accuracy:** {restoration_accuracy:.6f}")
            st.write(f"📌 **Restoration MSE:** {restoration_mse:.6f}")

if __name__ == "__main__":
    main()
```markdown
```
# Dual Data Sanitization and Restoration

## Description
A Python research prototype that implements and compares single-key and dual-key data sanitization and restoration techniques using hybrid metaheuristic algorithms (ECDO / hybrid DSOA+EFO variants). The repository contains two Streamlit apps and an analysis notebook for experimenting with sanitization (adding/transformation keys), restoring data, and measuring restoration accuracy and MSE.

## Features
- Single-key sanitization & restoration (additive key).
- Dual-key sanitization & restoration (additive + multiplicative keys).
- Optimization routines to search for keys that minimize restoration error.
- Streamlit UIs for interactive upload, sanitization, restoration, and download of datasets and keys.
- Notebook with multiple algorithm implementations (ECDO, DSOA, EFO, SMO, J-SSO) and analysis scripts.
- Exportable CSV outputs for sanitized data, restored data, and optimal keys.
- Comparison metrics: restoration accuracy, restoration MSE, key-space estimation, execution time.

## Tech Stack
- Python 3.x
- Streamlit (web UI)
- pandas (CSV I/O, DataFrames)
- numpy (numerical operations)
- scikit-learn (MinMaxScaler for normalization)
- matplotlib (used in the included Jupyter notebook)

## Installation
1. Create and activate a Python 3 virtual environment (recommended).
2. Install required packages:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib jupyter
```

(Only the packages imported in the codebase are required; install additional tools as needed.)

## Configuration / Environment Variables
- No environment variables or external configuration files are required by the code in this repository.
- The Streamlit apps accept CSV uploads via the web UI.

## Usage
Run either Streamlit app from the repository root:

- To run the comparison app:
```bash
streamlit run comparision.py
```
- To run the single/dual sanitization dashboard:
```bash
streamlit run dashboard.py
```

Open the provided local URL (printed by Streamlit) in your browser. Use the upload widgets to provide a dataset CSV; the apps will:
- Normalize the dataset,
- Run optimization to produce keys,
- Produce sanitized data,
- Offer downloadable CSVs for sanitized data and keys,
- Allow uploading sanitized data + key(s) to restore and evaluate results.

To inspect experiment code and extended analyses, open the notebook:
```bash
jupyter notebook Final_dual_ds_and_dr.ipynb
# or
jupyter lab Final_dual_ds_and_dr.ipynb
```

## Scripts / Commands
- `streamlit run comparision.py` — Launch the comparison Streamlit app (single vs dual sanitization).
- `streamlit run dashboard.py` — Launch the dashboard Streamlit app (ECDO-based sanitization/restoration).
- `jupyter notebook Final_dual_ds_and_dr.ipynb` — Open the analysis notebook.

There are no additional npm/Makefile scripts or automated installers present.

## Project Structure
- `comparision.py` — Streamlit app comparing single and dual sanitization/restoration (includes download/upload UI, metrics).
- `dashboard.py` — Streamlit app demonstrating ECDO-based sanitization and restoration.
- `Final_dual_ds_and_dr.ipynb` — Jupyter notebook with algorithm implementations, experiments, and plots.
- `dataset.csv` — example dataset present in the repo.
- `sanitized_data.csv`, `restored_data .csv`, `optimal_key.csv` — example/output CSV files (provided or generated).
- `plots/` — folder containing plots produced by experiments (if present).
- `Results/` — CSV results from experiments:
	- `combined results.csv`
	- `dsoa_results.csv`
	- `ecdo_results.csv`
	- `efo_results.csv`
	- `j_sso_results.csv`
	- `smo_results.csv`
- `README.md` / `Readme.md` / `readme.txt` — documentation and notes.

(Top-level files and folders only; see repository root for full contents.)

## Contributing
- This repository appears to be a research prototype. If you want to contribute:
	- Open issues describing reproducible steps, inputs, and expected outputs.
	- Submit pull requests that keep changes focused and include tests or clear reproducible demonstrations where applicable.
	- Follow existing code style and limit scope to one feature or fix per PR.
  
## License

This repository is under a proprietary license.

All rights are reserved. You may NOT:
- Copy, redistribute, or sublicense the code.
- Modify, adapt, or create derivative works.
- Use this code for commercial purposes.

Permission to use, modify, or redistribute this code may only be granted explicitly in writing by the copyright holder.

See the [LICENSE](./LICENSE) file for full details.
    
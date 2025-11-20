This repo contains a small Streamlit app for visualizing a power network and highlighting anomalous nodes based on simple anomaly scores.

The demo is built on a cleaned subset of the Pecan Street dataset (e.g., \`CLEAN_Pecan_House1.csv\`).  
Each numeric column is treated as a "bus" (node), and we construct a synthetic power network graph over these buses.

## Features

- Interactive graph visualization of the power network
- Nodes sized/colored by anomaly score
- One-click preview of top-K anomalous nodes (no LLM)
- (Optional) LLM-style textual reasoning panel (currently mocked for demo)

## File structure (key files)

- \`viz_app.py\`: main Streamlit app entry
- \`prep_pecan.py\`: scripts to clean / subsample Pecan data into a smaller CSV
- \`utils.py\`: shared utilities for loading data and computing anomaly scores
- \`real_data/CLEAN_Pecan_House1.csv\`: example cleaned CSV (small demo file)

> Note: do **not** upload the full 2.6GB raw Pecan CSV to this repo.  
> Instead, use a much smaller cleaned/subsampled CSV for demo purposes.

## How to run locally

```bash
# from the project root
pip install -r requirements.txt

streamlit run viz_app.py
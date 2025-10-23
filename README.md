# CRC-DataLab

CRC-DataLab is an interactive Streamlit application for exploring gene expression and genomic variant data related to colorectal cancer (CRC). It integrates exploratory visualizations, differential expression (DEGs) analyses, genomic variant inspection, machine learning workflows, basic NLP text-mining for gene insights, and survival analysis.

This repository contains the Streamlit app `CC-app.py` and example CSV data files used by the app (`expression-file.csv`, `variant_data.csv`, `path_to_survival_data.csv`). The app is intended for research and educational use.

## Table of contents

- Features
- Quick start
- Prerequisites
- Installation
- Running the app
- Data format and example files
- Pages (what each app page does)
- Development and testing
- Troubleshooting
- Contributing
- License

## Features

- Interactive gene search and expression visualizations (bar plots, violin plots)
- Differential expression analysis visualizations: heatmap, volcano plot, PCA
- Genomic variant exploration with detailed variant metadata and links
- Machine learning tools: K-Means clustering and Logistic Regression classification with evaluation metrics
- Text-mining (NER) using a transformer-based model to extract entities and quick PubMed links
- Survival analysis: Kaplan-Meier curves and Cox proportional hazards model
- Export functionality for filtered data and generated visualizations

## Quick start

Summary steps to run the app locally:

1. Install Python 3.10+ (3.11 recommended).
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Place the CSV data files in the project root (see Data format section).
5. Run the app with Streamlit: `streamlit run CC-app.py`.

See the detailed instructions below.

## Prerequisites

- Python 3.10 or 3.11 (3.11 recommended)
- Windows, macOS, or Linux
- Recommended: 8+ GB RAM for using the transformer-based NER model locally

## Installation

1. Clone the repository:

	git clone https://github.com/se7en69/CRC-DataLab.git
	cd CRC-DataLab

2. Create and activate a virtual environment (Windows PowerShell example):

	python -m venv .venv; .\.venv\Scripts\Activate.ps1

3. Upgrade pip and install dependencies:

	python -m pip install --upgrade pip
	pip install -r requirements.txt

Note: The `transformers` model used for NER may download model weights during the first run and requires additional disk space.

## Running the app

Start the Streamlit app from the project root:

	streamlit run CC-app.py

Open the URL printed by Streamlit (usually http://localhost:8501) in your browser.

## Data format and example files

The app expects three CSV files in the repository root (names used in the app):

- `expression-file.csv` — gene expression and DE analysis results. Expected columns (example):
  - `Gene_Name` (string)
  - `Sample_Type` (e.g., `Cancerous`, `Normal`)
  - `Expression_Level` (numeric)
  - `Fold_Change` (numeric)
  - `P_Value` (numeric)
  - `Pathway` (comma-separated strings)

- `variant_data.csv` — genomic variants table. Expected columns (example):
  - `Gene_Name`, `HGVSc`, `HGVSp`, `Chromosome`, `Start_Position`, `End_Position`, `Consequence`, `Variant_Classification`, `dbSNP_RS`, `Transcript_ID`, `RefSeq`, `Protein_position`, `Codons`, `Hotspot`, `Annotation_Status`

- `path_to_survival_data.csv` — survival metadata for Kaplan-Meier and Cox models. Expected columns (example):
  - `Survival_5_years` (`Yes`/`No` or numeric)
  - `Mortality` (`Yes`/`No` or numeric)
  - `Age`, `Gender`, `Cancer_Stage`, `Treatment_Type`, `Smoking_History`, `Obesity_BMI`, `Country`, etc.

If your files have different column names, either rename them or update `CC-app.py` accordingly.

## Pages (what each app page does)

- Home: project overview, authorship, images and navigation.
- Gene Expression Analysis: search genes, view expression metrics, pathway networks, and download gene data.
- DEGs Analysis: heatmap, volcano plot, PCA, violin plots, scatter matrix for DE analysis.
- Genomic Variant Analysis: search variants by gene/dbSNP/classification and view detailed metadata.
- Machine Learning Insights: K-Means clustering and Logistic Regression classification with evaluation metrics and downloads.
- Text Mining for Gene Insights: simple NER-based extraction and PubMed quick links.
- Survival Analysis: Kaplan-Meier curves, mortality bar charts, and Cox proportional hazards modeling.
- Export Data: filter and export tables and visualizations.

## Development and testing

- Format and linting: run your preferred tools (black, flake8, ruff) locally.
- Quick syntax check:

	python -m pyflakes CC-app.py || true

- Unit tests: none included by default. Consider adding pytest-based tests for data-processing functions if you refactor the app logic into testable modules.

## Troubleshooting

- Transformer model errors: if NER fails because of model download or memory, disable the Text Mining page or run the app on a machine with a GPU and sufficient RAM. You can also change device from CPU (device=-1) to GPU device id (e.g., 0) if PyTorch GPU is available.
- Missing image files: `logo.png` and `gene_abstract.png` are used on the Home page. If missing, replace them or remove the image-loading lines from `CC-app.py`.
- CSV parsing errors: ensure CSV files use UTF-8 encoding and the expected columns exist.
- Lifelines/Cox model errors: ensure enough rows after filtering and that categorical variables are converted to dummy variables.

## Contributing

Please see `CONTRIBUTING.md` for contributor guidelines.

## License

This project is licensed under the Apache License 2.0 — see the `LICENSE` file for details.

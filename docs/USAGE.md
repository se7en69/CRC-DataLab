# Usage Guide — CRC-DataLab

This file provides step-by-step usage examples, sample commands, and expectations when using CRC-DataLab locally.

## Before you start

Ensure you have:

- Python 3.10 or 3.11 installed
- Virtual environment activated
- Dependencies installed from `requirements.txt`
- CSV data files present in the project root: `expression-file.csv`, `variant_data.csv`, `path_to_survival_data.csv`

## Running the app

Start the app from the repository root:

   streamlit run CC-app.py

Streamlit will open the app in your browser (commonly at http://localhost:8501).

## Navigation overview

Use the left sidebar to navigate pages:

- Home — overview and images
- Gene Expression Analysis — search genes, view expression plots and pathway network
- DEGs Analysis — heatmaps, volcano plots, PCA, violin plots
- Genomic Variant Analysis — search variants and view metadata
- Machine Learning Insights — clustering and classification
- Text Mining for Gene Insights — short NER demo and PubMed link
- Survival Analysis — Kaplan-Meier and Cox regression
- Export Data — filter and export results

## Example workflows

1. Gene lookup and download

- Go to "Gene Expression Analysis"
- Type or select a gene name from the dropdown
- Review expression metrics and pathway network
- Click "Download as CSV" to save gene-specific data

2. Inspect top DEGs

- Go to "DEGs Analysis"
- Adjust the slider for number of top genes to display (heatmap) and thresholds for volcano plot
- Hover points in the volcano plot to view gene names; use PCA to visualize group separation

3. Variant search

- Go to "Genomic Variant Analysis"
- Choose search by Gene Name, dbSNP ID or Variant Classification
- Select a variant from results to view detailed metadata and expression comparison
- Click "Download Variant Data as CSV" to export results

4. Run clustering and download assignments

- Go to "Machine Learning Insights"
- Select number of clusters (k)
- After clustering, click "Download Cluster Assignments as CSV"

5. Survival analysis

- Go to "Survival Analysis"
- Apply filters (country, cancer stage, treatment) to subset the data
- View Kaplan-Meier curve and download it using the provided button
- If data is sufficient, fit the Cox model and download results

## Expected input formats (CSV examples)

- `expression-file.csv` (example row):

  Gene_Name,Sample_Type,Expression_Level,Fold_Change,P_Value,Pathway
  APC,Cancerous,12.5,3.2,0.0005,Wnt signaling, Cell cycle

- `variant_data.csv` (example row):

  Gene_Name,HGVSc,HGVSp,Chromosome,Start_Position,End_Position,Consequence,Variant_Classification,dbSNP_RS,Transcript_ID,RefSeq,Protein_position,Codons,Hotspot,Annotation_Status
  APC,c.3920T>A,p.Ile1307Lys,5,1123456,1123456,missense_variant,Missense_Mutation,rs123456,ENST00000312345,NP_001,1307,ATA>AAC,No,Reviewed

- `path_to_survival_data.csv` (example row):

  Survival_5_years,Mortality,Age,Gender,Cancer_Stage,Treatment_Type,Smoking_History,Obesity_BMI,Country
  Yes,No,65,Male,Stage II,Surgery,No,25,USA

## Tips

- If the NER transformer model raises memory or download issues, comment out or skip the Text Mining page when running locally.
- Ensure CSV files are UTF-8 encoded and that columns match the names used in `CC-app.py`.

## Exported files

- The app provides CSV downloads for gene-specific data, variant results, cluster assignments, mortality rates, and Cox model results.
- Visualizations (Kaplan-Meier, heatmap) can be downloaded as PNG images.

## Further customization

- To change expected filenames or column names, edit `CC-app.py` accordingly.
- Consider refactoring heavy data-processing into separate modules to enable unit testing.

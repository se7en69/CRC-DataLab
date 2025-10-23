import os
import io
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from transformers import pipeline
from lifelines import KaplanMeierFitter, CoxPHFitter

# configure tool page 
st.set_page_config(
    page_title="GeneExplorer",
    page_icon="🧬",
    layout="wide", 
)

# Load and cache the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("expression-file.csv")
    return data

df = load_data()

# loading varaints data
@st.cache_data
def load_variants_data():
    variants_data = pd.read_csv("variant_data.csv")  
    return variants_data

variants_df = load_variants_data()

def front_page():
    st.markdown("---")  

# Sidebar navigation and Navigation options
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    [
        "Home",
        "Gene Expression Analysis",
        "DEGs Analysis",
        "Genomic Variant Analysis",
        "Machine Learning Insights",
        "Text Mining for Gene Insights",
        "Survival Analysis",
        "Export Data"
    ]
)

st.sidebar.markdown("---")  

# Additional resources or information
st.sidebar.header("About")
st.sidebar.write(
    """
    This tool is designed for researchers, clinicians, and students to explore gene expression and genomic variants 
    in colorectal cancer. It integrates pre-analyzed data, statistical analyses, machine learning, and text mining 
    to provide actionable insights.
    """
)

# Database-Related Links
database_link_dict = {
        "Research Paper": "https://example.com/cancer-research-paper",
        "GitHub Page": "https://github.com/your-repo/geneexplorer",
    }
st.sidebar.markdown("## Database-Related Links")
for link_text, link_url in database_link_dict.items():
        st.sidebar.markdown(f"[{link_text}]({link_url})")


# Community-Related Links
community_link_dict = {
        "Colorectal Cancer Alliance": "https://www.ccalliance.org",
        "Cancer Genomics Hub": "https://www.cancergenomicshub.org",
        "RASopathies Network": "https://rasopathiesnet.org",
    }

st.sidebar.markdown("## Community-Related Links")
for link_text, link_url in community_link_dict.items():
        st.sidebar.markdown(f"[{link_text}]({link_url})")

    # Software-Related Links
software_link_dict = {
        "BioPython": "https://biopython.org",
        "Pandas": "https://pandas.pydata.org",
        "NumPy": "https://numpy.org",
        "SciPy": "https://scipy.org",
        "Scikit-learn": "https://scikit-learn.org",
        "Matplotlib": "https://matplotlib.org",
        "Seaborn": "https://seaborn.pydata.org",
        "Streamlit": "https://streamlit.io",
    }

st.sidebar.markdown("## Software-Related Links")
link_1_col, link_2_col, link_3_col = st.sidebar.columns(3)

i = 0
link_col_dict = {0: link_1_col, 1: link_2_col, 2: link_3_col}
for link_text, link_url in software_link_dict.items():
        st_col = link_col_dict[i]
        i += 1
        if i == len(link_col_dict.keys()):
            i = 0
        st_col.markdown(f"[{link_text}]({link_url})")

# Home Page
if page == "Home":
    left_col, right_col = st.columns(2)

    # Display logo
    img = Image.open("logo.jpg")  
    left_col.image(img, width=400)  

    # Display the title and description on the right
    right_col.markdown("# CRC DataLab")
    right_col.markdown("### A tool for analyzing gene expression and genomic variants in colorectal cancer")
    right_col.markdown("##### Created by Abdul Rehman Ikram")
    right_col.markdown("**For Research and Educational Purposes**")

    st.markdown("---")

    st.markdown(
        """
        ### Summary
        **CRC DataLab** is a tool for analyzing gene expression and genomic variants in colorectal cancer. 
        The tool integrates pre-analyzed gene expression data, statistical analyses, machine learning, and text mining 
        to provide insights into cancer biology. Researchers can explore differentially expressed genes (DEGs), 
        investigate genomic variants, and visualize data using interactive plots.

        Details of our work are provided in the [*Cancer Research*](https://example.com/cancer-research-paper) paper, 
        **Exploring Gene Expression and Genomic Variants in Colorectal Cancer**.
        We hope that researchers will use **CRC DataLab** to gain novel insights into colorectal cancer biology 
        and drug discovery.
        """
    )

    st.markdown("---")

    left_col, right_col = st.columns(2)

    # Display an abstract image or visualization
    img = Image.open("gene_abstract.png")  
    right_col.image(img, caption="CRC DataLab Workflow", width=650)  

    left_col.markdown(
        """
        ### Usage

        To the left, is a dropdown main menu for navigating to 
        each page in the **CRC DataLab** tool:

        - **Home Page:** We are here!
        - **Gene Expression Analysis:** Explore gene expression data in colorectal cancer.
        - **DEGs Analysis:** Analyze differentially expressed genes (DEGs) using heatmaps, volcano plots, and boxplots.
        - **Genomic Variant Analysis:** Investigate genomic variants and their impact on protein structure.
        - **Machine Learning Insights:** Apply machine learning techniques to analyze gene expression and variant data.
        - **Text Mining for Gene Insights:** Extract insights from biomedical literature using NLP.
        - **Survival Analysis:** Analyze survival rates and factors influencing patient outcomes in colorectal cancer.
        - **Export Data:** Export filtered data, visualizations, and analysis results.
        """
    )
    st.markdown("---")

    left_info_col, right_info_col = st.columns(2)

    left_info_col.markdown(
        """
        ### Authors
        Please feel free to contact us with any issues, comments, or questions.

        ##### Abdul Rehman Ikram [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg?style=social&label=Follow%20%40YourHandle)](https://twitter.com/YourHandle)

        - Email:  <hanzo7n@gmail.com>
        - GitHub: https://github.com/se7en69
        - LinkedIn: https://www.linkedin.com/in/hanzo7/
        - Portfolio: https://abdulrehmanikramportfolio.netlify.app/
        """
    )

    right_info_col.markdown(
        """
        ### Funding

        - Your Funding Source 1
        - Your Funding Source 2
         """
    )

    right_info_col.markdown(
        """
        ### License
        Apache License 2.0
        """
    )

# Gene Search Page
elif page == "Gene Expression Analysis":
    st.title("Gene Expression Analysis")
    st.write("""
        Explore gene expression data in colorectal cancer. Search for specific genes, view their expression levels in cancerous and normal tissues, 
        and analyze fold changes and statistical significance. Use interactive visualizations to compare expression patterns and identify differentially expressed genes (DEGs).
    """)
    st.markdown("---")
    # Autocomplete Search Box
    gene_list = df["Gene_Name"].unique()
    gene_query = st.selectbox("Enter a gene name:", options=gene_list, help="Start typing to search for a gene.")

    if gene_query:
        result = df[df["Gene_Name"].str.contains(gene_query, case=False)]
        st.write("### Search Results")
        st.write(result)

        if not result.empty:
            # Gene Details in a Card Layout
            st.markdown("---")
            st.write(f"### Gene: **{gene_query}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fold Change", value=f"{result['Fold_Change'].values[0]:.2f}")
            with col2:
                st.metric("P-Value", value=f"{result['P_Value'].values[0]:.2e}")
            with col3:
                st.metric("Expression Level (Cancerous)", value=f"{result[result['Sample_Type'] == 'Cancerous']['Expression_Level'].values[0]:.2f}")
            st.metric("Expression Level (Normal)", value=f"{result[result['Sample_Type'] == 'Normal']['Expression_Level'].values[0]:.2f}")

            # Expression Level Comparison Chart
            st.markdown("---")
            st.write("### Expression Level Comparison")
            expression_data = {
                "Sample Type": ["Cancerous", "Normal"],
                "Expression Level": [
                    result[result['Sample_Type'] == 'Cancerous']['Expression_Level'].values[0],
                    result[result['Sample_Type'] == 'Normal']['Expression_Level'].values[0]
                ]
            }
            fig = px.bar(expression_data, x="Sample Type", y="Expression Level", color="Sample Type", text="Expression Level")
            st.plotly_chart(fig)

            # Pathway Network Graph
            st.markdown("---")
            st.write("### Pathway Associations")

            # Extract pathways for the queried gene
            pathways = result["Pathway"].values[0].split(", ")

            # Create a network graph
            G = nx.Graph()
            G.add_node(gene_query, size=20, title=gene_query, color="blue") 

            # Add pathways as nodes and connect them to the gene
            for pathway in pathways:
                G.add_node(pathway, size=15, title=pathway, color="green")
                G.add_edge(gene_query, pathway)

            # Render the graph using pyvis
            net = Network(height="400px", width="100%", notebook=True, cdn_resources="remote")
            net.from_nx(G)
            net.show("pathway_graph.html")

            # Display the graph in Streamlit
            with open("pathway_graph.html", "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=500)

            # Download Gene Data
            st.markdown("---")
            st.write("### Download Gene Data")
            csv = result.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"{gene_query}_data.csv",
                mime="text/csv",
            )
        else:
            # Error Handling for Invalid Gene Names
            st.warning(f"No results found for gene: {gene_query}. Please check the gene name and try again.")

# DEGs Analysis Page
elif page == "DEGs Analysis":
    st.title("Differentially Expressed Genes (DEGs) Analysis")
    st.write("""
        Analyze differentially expressed genes (DEGs) in colorectal cancer. Visualize top DEGs using heatmaps, volcano plots, and boxplots. 
        Apply filters to explore genes based on fold change and p-value thresholds. Export results for further analysis.
    """)
    st.markdown("---")
    # Heatmap: Top Differentially Expressed Genes
    st.write("### Heatmap: Top Differentially Expressed Genes")
    top_n_genes = st.slider("Select Number of Top Genes to Display", min_value=10, max_value=100, value=20)
    top_genes = df.nlargest(top_n_genes, "Fold_Change")
    heatmap_data = top_genes.pivot_table(index="Gene_Name", columns="Sample_Type", values="Expression_Level", aggfunc="mean")  # Fixed line

    # Create Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    ax.set_title(f"Top {top_n_genes} Differentially Expressed Genes")
    ax.set_xlabel("Sample Type")
    ax.set_ylabel("Gene Name")
    st.pyplot(fig)

    # Volcano Plot: Differentially Expressed Genes
    st.markdown("---")
    st.write("### Volcano Plot: Differentially Expressed Genes")
    df["-log10(P_Value)"] = -np.log10(df["P_Value"])
    significance_threshold = st.slider("Significance Threshold (-log10 P-Value)", min_value=1.0, max_value=10.0, value=2.0)
    fold_change_threshold = st.slider("Fold Change Threshold", min_value=1.0, max_value=10.0, value=2.0)

    # Create Volcano Plot
    fig = px.scatter(df, 
                     x="Fold_Change", 
                     y="-log10(P_Value)", 
                     color="Sample_Type", 
                     hover_name="Gene_Name",
                     title="Volcano Plot of Differentially Expressed Genes",
                     labels={"Fold_Change": "Fold Change", "-log10(P_Value)": "-log10(P-Value)"})
    fig.add_hline(y=significance_threshold, line_dash="dash", line_color="red")
    fig.add_vline(x=fold_change_threshold, line_dash="dash", line_color="red")
    fig.add_vline(x=-fold_change_threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

    # PCA Plot: Dimensionality Reduction
    st.markdown("---")
    st.write("### PCA Plot: Dimensionality Reduction")
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df[["Fold_Change", "Expression_Level"]])
    df["PC1"] = df_pca[:, 0]
    df["PC2"] = df_pca[:, 1]

    # Create PCA Plot
    fig = px.scatter(df, 
                     x="PC1", 
                     y="PC2", 
                     color="Sample_Type", 
                     hover_name="Gene_Name",
                     title="PCA Plot of Gene Expression Data",
                     labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"})
    st.plotly_chart(fig)

    # Violin Plot: Expression Distribution
    st.markdown("---")
    st.write("### Violin Plot: Expression Distribution")
    gene_for_violin = st.selectbox("Select a gene for violin plot:", df["Gene_Name"].unique())
    if gene_for_violin:
        fig = px.violin(df[df["Gene_Name"] == gene_for_violin], 
                        x="Sample_Type", 
                        y="Expression_Level", 
                        color="Sample_Type", 
                        box=True, 
                        points="all",
                        title=f"Expression Distribution for {gene_for_violin}",
                        labels={"Expression_Level": "Expression Level", "Sample_Type": "Sample Type"})
        st.plotly_chart(fig)

    # Scatter Plot Matrix: Pairwise Relationships
    st.markdown("---")
    st.write("### Scatter Plot Matrix: Pairwise Relationships")
    selected_genes = st.multiselect("Select genes for scatter plot matrix:", df["Gene_Name"].unique(), default=df["Gene_Name"].unique()[:5])
    if selected_genes:
        scatter_data = df[df["Gene_Name"].isin(selected_genes)]
        fig = px.scatter_matrix(scatter_data, 
                                dimensions=["Fold_Change", "Expression_Level", "P_Value"], 
                                color="Sample_Type",
                                title="Scatter Plot Matrix of Selected Genes")
        st.plotly_chart(fig)

# Variant Analysis Page
elif page == "Genomic Variant Analysis":
    st.title("Genomic Variant Analysis")
    st.write("""
        Explore genomic variants associated with colorectal cancer. Search for variants by gene name, dbSNP ID, or variant classification. 
        View detailed information about each variant, including its impact on protein structure and functional consequences. 
        Link variants to gene expression data for a comprehensive analysis.
    """)
    # Fill NaN values in the dbSNP_RS column
    variants_df["dbSNP_RS"] = variants_df["dbSNP_RS"].fillna("")
    
    # Variant Search
    st.markdown("---")
    st.write("### Variant Search")
    search_option = st.radio("Search by:", ["Gene Name", "dbSNP ID", "Variant Classification"])
    
    if search_option == "Gene Name":
        # Autocomplete Search Box for Gene Name
        gene_list = variants_df["Gene_Name"].unique()
        gene_query = st.selectbox("Select or type a gene name:", options=gene_list)
        if gene_query:
            result = variants_df[variants_df["Gene_Name"].str.contains(gene_query, case=False)]
    elif search_option == "dbSNP ID":
        # Autocomplete Search Box for dbSNP ID
        dbsnp_list = variants_df["dbSNP_RS"].unique()
        dbsnp_query = st.selectbox("Select or type a dbSNP ID:", options=dbsnp_list)
        if dbsnp_query:
            result = variants_df[variants_df["dbSNP_RS"].str.contains(dbsnp_query, case=False)]
    elif search_option == "Variant Classification":
        # Autocomplete Search Box for Variant Classification
        variant_class = st.selectbox("Select a variant classification:", variants_df["Variant_Classification"].unique())
        result = variants_df[variants_df["Variant_Classification"] == variant_class]

    if "result" in locals() and not result.empty:
        st.write("### Search Results")
        st.write(result)

        # Display Variant Details in a Card Layout
        st.markdown("---")
        st.write("### Variant Details")
        selected_variant = st.selectbox("Select a variant to view details:", result["HGVSc"].unique())
        variant_details = result[result["HGVSc"] == selected_variant]

        if not variant_details.empty:
            # Card Layout for Variant Details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Gene Name**")
                st.write(variant_details["Gene_Name"].values[0])
            with col2:
                st.markdown("**Variant**")
                st.write(selected_variant)
            with col3:
                st.markdown("**Protein Change**")
                st.write(variant_details["HGVSp"].values[0])

            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown("**Chromosome**")
                st.write(variant_details["Chromosome"].values[0])
            with col5:
                st.markdown("**Start Position**")
                st.write(variant_details["Start_Position"].values[0])
            with col6:
                st.markdown("**End Position**")
                st.write(variant_details["End_Position"].values[0])
            col7, col8, col9 = st.columns(3)
            with col7:
                st.markdown("**Consequence**")
                st.write(variant_details["Consequence"].values[0])
            with col8:
                st.markdown("**Variant Classification**")
                st.write(variant_details["Variant_Classification"].values[0])
            with col9:
                st.markdown("**dbSNP ID**")
                st.write(variant_details["dbSNP_RS"].values[0])
            col10, col11, col12 = st.columns(3)
            with col10:
                st.markdown("**Transcript**")
                st.write(variant_details["Transcript_ID"].values[0])
            with col11:
                st.markdown("**RefSeq**")
                st.write(variant_details["RefSeq"].values[0])
            with col12:
                st.markdown("**Protein Position**")
                st.write(variant_details["Protein_position"].values[0])
            col13, col14, col15 = st.columns(3)
            with col13:
                st.markdown("**Codons**")
            st.write(variant_details["Codons"].values[0])
            with col14:
                st.markdown("**Hotspot**")
                st.write(variant_details["Hotspot"].values[0])
            with col15:
                st.markdown("**Annotation Status**")
                st.write(variant_details["Annotation_Status"].values[0])

        # Expression Level Comparison Chart
        st.markdown("---")
        st.write("### Expression Level Comparison")
        gene_name = variant_details["Gene_Name"].values[0]
        gene_expression_data = df[df["Gene_Name"] == gene_name]

        if not gene_expression_data.empty:
            # Prepare data for the bar chart
            expression_data = {
                "Sample Type": ["Cancerous", "Normal"],
                "Expression Level": [
                    gene_expression_data[gene_expression_data['Sample_Type'] == 'Cancerous']['Expression_Level'].values[0],
                    gene_expression_data[gene_expression_data['Sample_Type'] == 'Normal']['Expression_Level'].values[0]
                ]
            }

            # Create bar chart
            fig = px.bar(expression_data, x="Sample Type", y="Expression Level", color="Sample Type", 
                         title=f"Expression Levels for {gene_name}",
                         labels={"Expression Level": "Expression Level", "Sample Type": "Sample Type"})
            st.plotly_chart(fig)
        else:
            st.warning(f"No gene expression data found for {gene_name}.")

        # Export Variant Data
        st.markdown("---")
        st.write("### Export Variant Data")
        csv = result.to_csv(index=False)
        st.download_button(
            label="Download Variant Data as CSV",
            data=csv,
            file_name="variant_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("No results found. Please refine your search.")

# Machine Learning Page
elif page == "Machine Learning Insights":
    st.title("Machine Learning Insights")
    st.write("""
        Apply machine learning techniques to analyze gene expression and variant data. Use clustering to identify groups of genes or samples with similar expression patterns. 
        Classify samples as cancerous or normal using pre-trained models. Visualize high-dimensional data using PCA and t-SNE.
    """)
    
    # --- Clustering: K-Means ---
    st.markdown("---")
    st.write("### Clustering: K-Means")
    n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

    # Prepare data for clustering
    X = df[["Fold_Change", "Expression_Level"]]

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    # Visualize clusters
    fig = px.scatter(df, 
                     x="Fold_Change", 
                     y="Expression_Level", 
                     color="Cluster", 
                     hover_name="Gene_Name",
                     title=f"K-Means Clustering (k={n_clusters})",
                     labels={"Fold_Change": "Fold Change", "Expression_Level": "Expression Level"})
    st.plotly_chart(fig)

    # --- Download Cluster Assignments ---
    st.markdown("---")
    st.write("### Download Cluster Assignments")
    
    # Extract gene names and their cluster assignments
    cluster_data = df[["Gene_Name", "Cluster", "Fold_Change", "Expression_Level"]]
    
    # Convert to CSV
    csv = cluster_data.to_csv(index=False)
    st.download_button(
        label="Download Cluster Assignments as CSV",
        data=csv,
        file_name=f"gene_clusters_k={n_clusters}.csv",
        mime="text/csv",
        help="Download a CSV file mapping genes to their assigned clusters."
    )

    # --- Classification: Logistic Regression ---
    st.markdown("---")
    st.write("### Classification: Logistic Regression")
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)

    # Prepare data for classification
    X = df[["Fold_Change", "Expression_Level"]]
    y = df["Sample_Type"].apply(lambda x: 1 if x == "Cancerous" else 0)  # Convert labels to binary

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display classification report
    st.write("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve
    st.write("#### ROC Curve")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.2f})",
                  labels={"x": "False Positive Rate", "y": "True Positive Rate"})
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig)
    
# Text Mining Page
elif page == "Text Mining for Gene Insights":
    st.title("Text Mining for Gene Insights")
    st.write("Extract insights about genes and variants from biomedical literature.")

    # Autocomplete Search Box for Gene Name
    gene_list = df["Gene_Name"].unique()
    gene_query = st.selectbox("Select or type a gene name:", options=gene_list)

    if gene_query:
        st.write(f"### Gene: {gene_query}")

        try:
            # Initialize the NLP pipeline with better parameters
            @st.cache_resource
            def load_nlp_model():
                return pipeline(
                    "ner", 
                    model="dmis-lab/biobert-v1.1", 
                    tokenizer="dmis-lab/biobert-v1.1",
                    aggregation_strategy="simple",
                    device=-1  # Use CPU (change to 0 for GPU if available)
                )
            
            nlp = load_nlp_model()
            
            # Create a more natural query that will tokenize better
            query = f"The gene {gene_query} (also known as {gene_query}) has been studied in relation to colorectal cancer."
            
            # Process the text
            text_result = nlp(query)
            
            # Filter and display only meaningful entities
            st.write("#### Text Mining Insights")
            st.write("Extracted entities from the query:")
            
            meaningful_entities = []
            for entity in text_result:
                # Skip subword tokens (those starting with ##)
                if not entity['word'].startswith('##'):
                    meaningful_entities.append(entity)
                    st.write(f"- **{entity['word']}**: {entity['entity_group']} (confidence: {entity['score']:.2f})")
            
            if not meaningful_entities:
                st.warning("No meaningful entities were extracted. Try a different query format.")
            
            # Display references from PubMed - now with actual search link
            st.write("#### References from PubMed")
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={gene_query}+colorectal+cancer"
            st.write(f"Search query: `{gene_query} AND colorectal cancer`")
            st.write(f"View actual PubMed search results: [PubMed Search]({pubmed_url})")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Technical details for debugging:")
            st.exception(e)
    else:
        st.warning("Please select or type a gene name to extract insights.")
# Survival Analysis Page
elif page == "Survival Analysis":
    st.title("Survival Analysis")
    st.write("Analyze survival rates and factors influencing patient outcomes in colorectal cancer.")

    # Load the survival data
    @st.cache_data
    def load_survival_data():
        data = pd.read_csv("path_to_survival_data.csv")  # Replace with your file path
        return data

    survival_df = load_survival_data()

    # Convert "Yes"/"No" to numeric values
    survival_df["Survival_5_years"] = survival_df["Survival_5_years"].map({"Yes": 1, "No": 0})
    survival_df["Mortality"] = survival_df["Mortality"].map({"Yes": 1, "No": 0})

    # Ensure the columns are numeric
    survival_df["Survival_5_years"] = pd.to_numeric(survival_df["Survival_5_years"], errors='coerce')
    survival_df["Mortality"] = pd.to_numeric(survival_df["Mortality"], errors='coerce')

    # Check for NaN values
    if survival_df[["Survival_5_years", "Mortality"]].isnull().any().any():
        st.warning("Warning: NaN values detected in 'Survival_5_years' or 'Mortality'. Rows with NaN values will be dropped.")
        survival_df = survival_df.dropna(subset=["Survival_5_years", "Mortality"])

    # Display dataset overview
    st.write("### Dataset Overview")
    st.write(survival_df.head())
    st.write(f"Dataset Shape: {survival_df.shape}")

    # Interactive Filters
    st.markdown("---")
    st.write("### Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        country_filter = st.multiselect("Select Country:", options=survival_df["Country"].unique(), default=survival_df["Country"].unique())
    with col2:
        cancer_stage_filter = st.multiselect("Select Cancer Stage:", options=survival_df["Cancer_Stage"].unique(), default=survival_df["Cancer_Stage"].unique())
    with col3:
        treatment_filter = st.multiselect("Select Treatment Type:", options=survival_df["Treatment_Type"].unique(), default=survival_df["Treatment_Type"].unique())

    # Apply filters
    filtered_df = survival_df[
        (survival_df["Country"].isin(country_filter)) &
        (survival_df["Cancer_Stage"].isin(cancer_stage_filter)) &
        (survival_df["Treatment_Type"].isin(treatment_filter))
    ]

    # Survival Rate Analysis
    st.markdown("---")
    st.write("### Survival Rate Analysis")
    st.write("Analyze 5-year survival rates based on selected filters.")

    # Kaplan-Meier Curve
    kmf = KaplanMeierFitter()
    kmf.fit(durations=filtered_df["Survival_5_years"], event_observed=filtered_df["Mortality"])

    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)

    # Export Kaplan-Meier Curve
    # Convert the figure to a PNG byte stream
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Download button
    st.download_button(
        label="Download Kaplan-Meier Curve as PNG",
        data=buf,
        file_name="kaplan_meier_curve.png",
        mime="image/png",
    )

    # Mortality Rate Analysis
    st.markdown("---")
    st.write("### Mortality Rate Analysis")
    st.write("Analyze mortality rates based on selected filters.")

    # Bar Chart for Mortality Rates
    mortality_rates = filtered_df.groupby("Cancer_Stage")["Mortality"].mean().reset_index()
    fig = px.bar(mortality_rates, x="Cancer_Stage", y="Mortality", title="Mortality Rates by Cancer Stage")
    st.plotly_chart(fig)

    # Export Mortality Rates Data
    st.download_button(
        label="Download Mortality Rates Data as CSV",
        data=mortality_rates.to_csv(index=False),
        file_name="mortality_rates.csv",
        mime="text/csv",
    )

# Cox Proportional Hazards Model
    st.markdown("---")
    st.write("### Cox Proportional Hazards Model")
    st.write("Identify significant predictors of survival using Cox regression.")

    # Prepare data for Cox regression
    cox_df = filtered_df[["Survival_5_years", "Mortality", "Age", "Gender", "Cancer_Stage", "Treatment_Type", "Smoking_History", "Obesity_BMI"]]

    # Convert categorical variables to dummy variables
    cox_df = pd.get_dummies(cox_df, columns=["Gender", "Cancer_Stage", "Treatment_Type", "Smoking_History", "Obesity_BMI"], drop_first=True)

    # Check the prepared data
    st.write("#### Prepared Data for Cox Model")
    st.write(cox_df.head())
    st.write(f"Shape of Cox Data: {cox_df.shape}")

    # Check for NaN values
    if cox_df.isnull().any().any():
        st.warning("Warning: NaN values detected in Cox data. Rows with NaN values will be dropped.")
        cox_df = cox_df.dropna()

    # Check for sufficient data
    if len(cox_df) < 10:  # Adjust threshold as needed
        st.error("Insufficient data for Cox model. Please adjust filters to include more rows.")
    else:

        # Check for collinearity
        st.write("#### Correlation Matrix")
        st.write(cox_df.corr())

        # Fit Cox model
        cph = CoxPHFitter()
        try:
            cph.fit(cox_df, duration_col="Survival_5_years", event_col="Mortality")
            
            # Display Cox model summary as an interactive table
            st.write("#### Cox Model Summary")
            st.dataframe(cph.summary)

            # Visualizations
            st.write("#### Hazard Ratio Plot")
            hazard_ratios = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]].reset_index()
            hazard_ratios.columns = ["Covariate", "Hazard_Ratio", "Lower_CI", "Upper_CI"]
            fig = px.scatter(hazard_ratios, x="Hazard_Ratio", y="Covariate", error_x="Lower_CI", error_x_minus="Upper_CI",
                            title="Hazard Ratios with 95% Confidence Intervals",
                            labels={"Hazard_Ratio": "Hazard Ratio (exp(coef))", "Covariate": "Covariate"})
            fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Reference", annotation_position="bottom right")
            st.plotly_chart(fig)

            st.write("#### Coefficient Plot")
            coefficients = cph.summary[["coef", "coef lower 95%", "coef upper 95%"]].reset_index()
            coefficients.columns = ["Covariate", "Coefficient", "Lower_CI", "Upper_CI"]
            fig = px.scatter(coefficients, x="Coefficient", y="Covariate", error_x="Lower_CI", error_x_minus="Upper_CI",
                            title="Coefficients with 95% Confidence Intervals",
                            labels={"Coefficient": "Coefficient (coef)", "Covariate": "Covariate"})
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Reference", annotation_position="bottom right")
            st.plotly_chart(fig)

            st.write("#### Significance Plot")
            p_values = cph.summary[["p"]].reset_index()
            p_values.columns = ["Covariate", "p_value"]
            p_values["Significant"] = p_values["p_value"] < 0.05
            p_values["neg_log10_p_value"] = -np.log10(p_values["p_value"])  # Calculate -log10(p-value)
            fig = px.bar(p_values, x="Covariate", y="neg_log10_p_value", color="Significant",
                        title="Significance of Covariates (-log10 p-value)",
                        labels={"neg_log10_p_value": "-log10(p-value)", "Covariate": "Covariate"})
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", annotation_text="p = 0.05", annotation_position="bottom right")
            st.plotly_chart(fig)

            # Export Cox Model Results
            st.download_button(
                label="Download Cox Model Results as CSV",
                data=cph.summary.to_csv(),
                file_name="cox_model_results.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error fitting Cox model: {e}")

# Export Data Page
elif page == "Export Data":
    st.title("Export Data")
    st.write("""
        Export filtered data and visualizations for further use. Download gene expression data, variant information, 
        and analysis results in CSV, Excel, or image formats for reports and presentations.
    """)
    st.markdown("---")
    # Section 1: Data Filters
    st.write("### Data Filters")
    st.write("Use the filters below to select the data you want to export.")

    col1, col2 = st.columns(2)
    with col1:
        # Default to only two genes instead of all genes
        gene_filter = st.multiselect(
            "Select Genes:",
            options=df["Gene_Name"].unique(),
            default=df["Gene_Name"].unique()[:40],
            help="Select one or more genes to filter the data."
        )
        sample_type_filter = st.multiselect(
            "Select Sample Types:",
            options=df["Sample_Type"].unique(),
            default=df["Sample_Type"].unique(),
            help="Select sample types (e.g., Cancerous, Normal) to filter the data."
        )
    with col2:
        fold_change_threshold = st.slider(
            "Fold Change Threshold:",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            help="Filter genes with a fold change greater than or equal to this value."
        )
        p_value_threshold = st.slider(
            "P-Value Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            help="Filter genes with a p-value less than or equal to this value."
        )

    # Apply filters
    filtered_data = df[
        (df["Gene_Name"].isin(gene_filter)) &
        (df["Sample_Type"].isin(sample_type_filter)) &
        (df["Fold_Change"] >= fold_change_threshold) &
        (df["P_Value"] <= p_value_threshold)
    ]

    # Display filtered data
    st.write("#### Filtered Data Preview")
    st.dataframe(filtered_data)

    # Section 2: Export Data
    st.markdown("---")
    st.write("### Export Data")
    st.write("Choose the format and file name for exporting the filtered data.")

    # Customizable file name
    file_name = st.text_input(
        "Enter a file name (without extension):",
        value="filtered_data",
        help="Specify a custom name for the exported file."
    )

    # Export options
    export_format = st.radio(
        "Select export format:",
        ["CSV", "JSON"],
        help="Choose the format for exporting the data."
    )

    if export_format == "CSV":
        data = filtered_data.to_csv(index=False)
        file_extension = "csv"
        mime_type = "text/csv"
    elif export_format == "JSON":
        data = filtered_data.to_json(orient="records")
        file_extension = "json"
        mime_type = "application/json"

    # Download button
    st.download_button(
        label=f"Download Filtered Data as {export_format}",
        data=data,
        file_name=f"{file_name}.{file_extension}",
        mime=mime_type,
    )

    # Section 3: Export Visualizations
    st.markdown("---")
    st.write("### Export Visualizations")
    st.write("Generate and export visualizations based on the filtered data.")

    # Visualization options
    visualization_option = st.selectbox(
        "Select a visualization to export:",
        ["Bar Chart", "Heatmap"],
        help="Choose a visualization to generate and export."
    )

    if visualization_option == "Bar Chart":
        st.write("#### Bar Chart")
        st.write("This visualization shows the expression levels of selected genes.")
        # Generate a bar chart
        if not filtered_data.empty:
            fig = px.bar(filtered_data, x="Gene_Name", y="Expression_Level", color="Sample_Type",
                         title="Gene Expression Levels",
                         labels={"Expression_Level": "Expression Level", "Gene_Name": "Gene Name"})
            st.plotly_chart(fig)

        else:
            st.warning("No data available to generate the bar chart.")

    elif visualization_option == "Heatmap":
        st.write("#### Heatmap")
        st.write("This visualization shows the expression levels of selected genes across sample types.")
        # Generate a heatmap
        if not filtered_data.empty:
            heatmap_data = filtered_data.pivot_table(index="Gene_Name", columns="Sample_Type", values="Expression_Level", aggfunc="mean")
            fig, ax = plt.subplots()
            sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Gene Expression Heatmap")
            ax.set_xlabel("Sample Type")
            ax.set_ylabel("Gene Name")
            st.pyplot(fig)

            # Export heatmap as PNG
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="Download Heatmap as PNG",
                data=buf,
                file_name=f"{file_name}_heatmap.png",
                mime="image/png",
            )
        else:
            st.warning("No data available to generate the heatmap.")
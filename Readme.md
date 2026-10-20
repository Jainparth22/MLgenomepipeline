# Genome Analysis Pipeline

This repository contains a comprehensive genome analysis pipeline that encompasses data preprocessing, genome assembly, quality assessment, variant calling, variant annotation, and pathogenicity prediction using machine learning models. This project is designed for both technical and non-technical audiences to explore genomic insights and predictions.

---

## **Key Features**

1. **Data Preprocessing**
   - Simulates processing raw genomic data.
   - Outputs preprocessed data with quality metrics.

2. **Genome Assembly**
   - Constructs contigs from preprocessed sequences.
   - Includes assembly quality assessment metrics like N50 and GC content.

3. **Variant Calling and Annotation**
   - Identifies genomic variants (SNPs, indels).
   - Annotates variants with functional impacts.

4. **Pathogenicity Prediction**
   - Uses advanced machine learning models (MLP and CNN) to classify variants as pathogenic or benign.
   - Visualizes prediction results and evaluates model performance with metrics like ROC and confusion matrices.

5. **Visualization Tools**
   - Training history plots.
   - ROC curves, confusion matrices, and more.

---

## **Installation**

### Prerequisites
Install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

---
Project Report: Genome Analysis and Variant Pathogenicity Prediction

1. Introduction
This project involves building a pipeline for analyzing genome data and predicting the pathogenicity (disease-causing potential) of genetic variants. The pipeline integrates multiple steps, from data preprocessing to variant annotation and machine learning-based pathogenicity prediction.
Objective
The primary goal is to create a streamlined and automated process for genome analysis that can:
Assemble genomic data.
Identify and annotate genetic variants.
Predict whether a genetic variant is likely to cause a disease.
2. Methodology
The project is divided into several stages, each simulating a real-world genome analysis workflow.
2.1. Data Preprocessing
Purpose: Converts raw sequencing data into a format suitable for analysis.
Key Task: Processes sequences and evaluates their quality.
Output: A table of sequences with quality scores, showing the number of sequences retained after preprocessing.
2.2. Genome Assembly
Purpose: Combines short sequences into longer, continuous genome fragments called contigs.
Key Task: Simulates assembling genomic sequences into three contigs.
Output: A list of contigs, including the longest one.
2.3. Assembly Quality Assessment
Purpose: Evaluates the quality of the assembled genome.
Key Metrics:
N50: Measures the length of contigs, favoring longer ones.
GC Content: Indicates the percentage of guanine (G) and cytosine (C) in the genome.
Output: A report summarizing the quality metrics.
2.4. Variant Calling
Purpose: Detects differences (variants) between the assembled genome and a reference genome.
Key Task: Simulates identifying five variants.
Output: A table of variants with details such as chromosome, position, and nucleotide changes.
2.5. Variant Annotation
Purpose: Adds biological significance to each variant, e.g., whether it affects gene function.
Key Task: Annotates variants with their effects, such as "missense" (changes protein function) or "synonymous" (no effect).
Output: A summary table with annotated variants and their categories.
2.6. Pathogenicity Prediction
Purpose: Uses a machine learning model to predict the likelihood of variants being disease-causing.
Key Features Used:
CADD_PHRED: A score predicting variant impact.
SIFT/PolyPhen: Scores assessing protein function impact.
Model Architecture: Combines two neural network types:
MLP (Multi-Layer Perceptron) for numerical features.
CNN-LSTM for sequential features.
Output: Predictions of pathogenicity probability (0 to 1).
3. Outputs and Their Significance
3.1. Preprocessed Data
Output: Displays sequences and their quality scores.
Significance: Ensures only high-quality data is used, avoiding errors in downstream analysis.
3.2. Assembled Genome
Output: Three contigs with details of the longest one.
Significance: Provides a near-complete representation of the genome for further analysis.
3.3. Quality Assessment Report
Output: Metrics such as N50 and GC content.
Significance: Evaluates the reliability and completeness of the assembly.
3.4. Identified Variants
Output: A table listing five genetic variants.
Significance: Identifies differences that may lead to diseases or traits of interest.
3.5. Annotated Variants
Output: Adds functional descriptions to variants.
Significance: Provides biological context, aiding researchers in prioritizing significant variants.
3.6. Pathogenicity Predictions
Output: Probabilities for each variant being pathogenic.
Significance: Aids in identifying disease-causing variants for research or clinical use.
3.7. Visualizations
Training History: Shows model's learning progress.
Predictions Distribution: Highlights model confidence in predictions.
ROC Curve: Demonstrates model accuracy.
Confusion Matrix: Provides a detailed breakdown of model performance.
4. Importance of the Project
For someone unfamiliar with deep learning or biotechnology:
This pipeline automates a complex and time-consuming process of genome analysis.
Identifying disease-causing variants can help in diagnosing genetic disorders and developing treatments.
Using machine learning enhances accuracy and scalability, making this pipeline a valuable tool for modern genetic research.
5. Conclusion
This project demonstrates the integration of bioinformatics and machine learning to tackle real-world challenges in genomics. The outputs highlight the power of automation and prediction models in advancing genetic research.
---


## **Project Components**

### **Pipeline Overview**

1. **Data Preprocessing**
   ```python
   processed_data = preprocessor.preprocess("sample_data.fastq")
   ```
   - Input: Raw sequence data.
   - Output: Processed sequence dataset with quality scores.

2. **Genome Assembly**
   ```python
   assembly = assembler.assemble(processed_data)
   ```
   - Input: Preprocessed data.
   - Output: Assembled genome contigs.

3. **Assembly Quality Assessment**
   ```python
   quality_report = quality_assessor.assess(assembly)
   ```
   - Outputs key metrics like N50, L50, and GC content.

4. **Variant Calling and Annotation**
   ```python
   variants = variant_caller.call_variants(assembly, "reference_genome.fasta")
   annotated_variants = annotator.annotate(variants)
   ```
   - Outputs annotated genomic variants with functional classifications.

5. **Pathogenicity Prediction**
   ```python
   predictor.build_model(input_shape)
   predictor.train_model(X_train, X_train_cnn, y_train, X_val, X_val_cnn, y_val)
   ```
   - Utilizes combined MLP and CNN architecture.

6. **Visualization Tools**
   ```python
   visualizer.plot_training_history(history)
   visualizer.plot_roc_curve(y_true, y_pred)
   ```
   - Generates plots for model evaluation and predictions.

---

## **Outputs and Results**

### Example Outputs:
- **Processed Data**:
  ```plaintext
  sequence     quality_score
  ATCG         30
  GCTA         25
  TGCA         35
  ```

- **Assembled Contigs**:
  ```plaintext
  Assembled contigs: ['ATCGGCTATGCA', 'GCTATGCAATCG', 'TGCAATCGGCTA']
  Longest contig: ATCGGCTATGCA
  ```

- **Variant Annotation**:
  ```plaintext
  CHROM POS REF ALT ANNOTATION
  chr1  1000 A   T   missense
  chr1  2000 C   G   synonymous
  ```

- **Pathogenicity Prediction**:
  ```plaintext
  Predicted Probability Distribution: Benign: 80%, Pathogenic: 20%
  ```

---

## **Visualization Examples**

1. **Training History**:
   ![Training Loss and Accuracy](#)

2. **ROC Curve**:
   ![ROC Curve](#)

3. **Confusion Matrix**:
   ![Confusion Matrix](#)

---

## **Usage**

Run the pipeline by executing:
```bash
python main.py
```

---
The outputs of genome analysis pipeline represent different stages of genomic data processing and machine learning predictions. Here’s a detailed explanation for each type of output:

---

### **1. Processed Data**
- **Example Output:**
  ```plaintext
  sequence     quality_score
  ATCG         30
  GCTA         25
  TGCA         35
  ```
- **Explanation:**
  - This output represents the cleaned and processed genomic sequences.
  - Each sequence has an associated `quality_score`, indicating the reliability of the sequence data.
  - This is the input for further steps like assembly and analysis.

---

### **2. Assembled Contigs**
- **Example Output:**
  ```plaintext
  Assembled contigs: ['ATCGGCTATGCA', 'GCTATGCAATCG', 'TGCAATCGGCTA']
  Longest contig: ATCGGCTATGCA
  ```
- **Explanation:**
  - Contigs are longer stretches of DNA assembled from smaller sequence fragments.
  - The output shows all assembled contigs and identifies the longest one, which is often a key metric for assembly success.

---

### **3. Assembly Quality Report**
- **Example Output:**
  ```plaintext
  Metric           Value
  N50              50000
  L50              100
  Total length     5000000
  GC content       0.45
  ```
- **Explanation:**
  - **N50**: Indicates the length of the smallest contig in the set that contains 50% of the total assembly length. Higher values suggest better assembly.
  - **L50**: The number of contigs needed to reach 50% of the total assembly length. Lower values indicate better assembly.
  - **Total length**: The cumulative length of all contigs.
  - **GC content**: The proportion of guanine (G) and cytosine (C) bases, useful for assessing genome composition.

---

### **4. Variant Calling and Annotation**
- **Example Output:**
  ```plaintext
  CHROM POS  REF ALT  ANNOTATION
  chr1  1000 A   T    missense
  chr1  2000 C   G    synonymous
  ```
- **Explanation:**
  - **CHROM** and **POS**: Indicate the chromosome and position of the variant.
  - **REF** and **ALT**: Represent the reference and alternate alleles at the position.
  - **ANNOTATION**: Describes the functional impact of the variant:
    - **Missense**: A single nucleotide change that alters the protein.
    - **Synonymous**: A nucleotide change that doesn’t affect the protein sequence.
    - Other annotations like **nonsense** or **intronic** may also appear.

---

### **5. Pathogenicity Prediction**
- **Example Output:**
  ```plaintext
  Predicted Probability Distribution: Benign: 80%, Pathogenic: 20%
  ```
- **Explanation:**
  - The model predicts whether a variant is **pathogenic** (disease-causing) or **benign**.
  - The output shows the probability distribution for all variants analyzed.
  - A higher pathogenic probability suggests greater confidence in the variant being disease-related.

---

### **6. Visualization Outputs**
- **Training History**:
  - **Example Graph**: Line plots showing loss and accuracy over epochs.
  - **Explanation**: 
    - Training loss and accuracy indicate how well the model learns from the data.
    - Validation metrics help assess performance on unseen data.

- **ROC Curve**:
  - **Example Graph**: A curve showing the trade-off between sensitivity and specificity.
  - **Explanation**:
    - AUC (Area Under Curve) quantifies model performance (1.0 is perfect, 0.5 is random).
    - A high AUC indicates effective discrimination between pathogenic and benign variants.

- **Confusion Matrix**:
  - **Example Matrix**:
    ```plaintext
    Predicted →  Benign  Pathogenic
    True ↓      
    Benign         95         5
    Pathogenic      4        96
    ```
  - **Explanation**:
    - True positives (correct pathogenic predictions) and true negatives (correct benign predictions) are along the diagonal.
    - Off-diagonal elements indicate errors (e.g., False Positives or False Negatives).

- **Predictions Distribution**:
  - **Example Graph**: Histogram of prediction probabilities.
  - **Explanation**:
    - The graph shows the spread of prediction probabilities for variants.
    - Clusters at extremes (0 or 1) indicate strong confidence in predictions.

---

### **Summary of Outputs**
These outputs together provide a complete picture of the pipeline's functionality, from raw data to actionable insights. Each output is accompanied by metrics and visualizations that ensure reliability and transparency in genomic analysis. Let me know if you'd like further details or assistance!

---

## **Contributing**

Contributions are welcome! Please submit pull requests or raise issues for new features or bug fixes.

---

## **License**

This project is licensed under the MIT License.

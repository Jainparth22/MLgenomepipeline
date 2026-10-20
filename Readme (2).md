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

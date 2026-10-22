# 🧬 Genome Analysis Pipeline & Pathogenic Variant Identification

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Genomics-green)](https://en.wikipedia.org/wiki/Bioinformatics)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20MLP%20%2B%20CNN--LSTM-purple)](#-machine-learning-architecture)

A comprehensive, modular bioinformatics and machine learning pipeline that simulates genomic data preprocessing, *de novo* genome assembly, assembly quality evaluation, variant calling and annotation, and predicts the clinical **pathogenicity of genetic variants** using a **hybrid Multi-Layer Perceptron (MLP) and CNN-LSTM deep learning architecture**.

---

## 📑 Table of Contents

- [Overview & Project Statement](#-overview--project-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Genomic & Bioinformatic Metrics Explained](#-genomic--bioinformatic-metrics-explained)
- [Repository Structure](#-repository-structure)
- [Installation & Prerequisites](#-installation--prerequisites)
- [Usage & Quick Start](#-usage--quick-start)
- [Pipeline Execution Walkthrough](#-pipeline-execution-walkthrough)
- [Visual Diagnostics & Model Evaluation](#-visual-diagnostics--model-evaluation)
- [Project Roadmap & Future Enhancements](#-project-roadmap--future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔬 Overview & Project Statement

High-throughput Next-Generation Sequencing (NGS) generates massive amounts of raw sequence reads that must be assembled, quality-checked, and compared against reference genomes to detect mutations. However, determining whether an identified genetic variant is **benign** or **disease-causing (pathogenic)** remains one of modern precision medicine's greatest bottlenecks.

This project delivers an end-to-end framework designed to:
1. **Model the Bioinformatics Workflow**: Clean raw reads, perform *de novo* assembly into contigs, compute quality statistics, and identify/annotate genetic variations.
2. **Automate Pathogenicity Classification**: Leverage multi-modal deep learning combining tabular genomic metadata with sequential deep feature extraction to predict variant disease risk.
3. **Provide Transparent Diagnostics**: Equip researchers with clear statistical visualizations (ROC curves, confusion matrices, prediction distribution curves, and training history logs).

---

## ✨ Key Features

- **Raw Data Preprocessing**: Simulates FASTQ sequencing data quality control and Phred quality score calculation.
- **Genome Assembly Engine**: Models *de novo* sequence assembly into contiguous sequences (contigs) and identifies maximal contig lengths.
- **Assembly Quality Assessment**: Computes assembly contiguity and composition metrics, including **N50**, **L50**, total length, and **GC content**.
- **Variant Calling & Annotation**: Identifies Single Nucleotide Polymorphisms (SNPs) across chromosomes and assigns functional consequences (missense, synonymous, nonsense, intronic, splice site).
- **Hybrid Deep Learning Model**:
  - **MLP Branch**: Captures non-linear relationships across tabular functional scores (`CADD_PHRED`, `SIFT`, `PolyPhen`, `AF`, `DP`, `QUAL`).
  - **CNN-LSTM Branch**: Employs 1D temporal convolutions and an LSTM cell to model sequential dependencies.
  - **Early Stopping & Regularization**: Protects against overfitting using Dropout and validation loss patience monitoring.
- **Diagnostic Visualizations**: Generates publication-ready plots for ROC curves (with AUC), confusion matrices, prediction probability histograms, and loss/accuracy trajectories.

---

## 🏗 System Architecture

The pipeline processes data through sequential bioinformatics stages culminating in the deep learning pathogenicity classifier:

```mermaid
flowchart TD
    A["Raw Sequencing Reads<br/>(FASTQ Format)"] --> B["Data Preprocessor<br/>(Quality Scoring & Trimming)"]
    B --> C["Genome Assembler<br/>(Contig Generation)"]
    C --> D["Assembly Quality Assessor<br/>(N50, L50, GC Content)"]
    D --> E["Variant Caller<br/>(Reference Genome Alignment)"]
    E --> F["Variant Annotator<br/>(Functional Classification)"]
    
    F --> G["Feature Synthesis & Normalization<br/>(StandardScaler)"]
    
    subgraph "Hybrid Deep Learning Engine"
        G --> H["MLP Branch<br/>Dense(64) -> Dropout(0.3) -> Dense(32)"]
        G --> I["CNN-LSTM Branch<br/>Conv1D(32) -> Conv1D(64) -> LSTM(32)"]
        H --> J["Concatenation Layer"]
        I --> J
        J --> K["Dense Output (Sigmoid)<br/>Pathogenicity Probability (0 to 1)"]
    end
    
    K --> L["Evaluation & Diagnostics<br/>(ROC-AUC, Confusion Matrix, Distributions)"]
```

---

## 🧠 Machine Learning Architecture

The pathogenicity prediction engine in [`main.py`](file:///a:/Github/ML_Proj_Genome_Analysis_Pipeline/main.py) fuses tabular clinical/functional features through two parallel neural network pathways:

```
Inputs: [AF, DP, QUAL, CADD_PHRED, SIFT_score, PolyPhen_score]
  │
  ├──► [MLP Branch] ────────────────────────────────────────┐
  │     Input: (features,)                                  │
  │     Dense (64 units, ReLU)                              │
  │     Dropout (rate = 0.3)                                │
  │     Dense (32 units, ReLU)                              │
  │                                                         ▼
  └──► [CNN-LSTM Branch] ──────────────────────────► [Concatenate] ──► [Dense(1, Sigmoid)]
        Input: (features, 1)                                │                │
        Conv1D (32 filters, kernel=3, ReLU)                 │                ▼
        Conv1D (64 filters, kernel=3, ReLU)                 │        P(Pathogenic) ∈ [0, 1]
        LSTM (32 units) ────────────────────────────────────┘
```

### Feature Dictionary

| Feature | Type | Biological Description |
| :--- | :--- | :--- |
| **`AF`** | Float `[0, 1]` | **Allele Frequency**: The frequency of the variant in the population. Rare variants are more often pathogenic. |
| **`DP`** | Integer | **Read Depth**: Sequencing depth at the variant locus. Higher depth increases confidence. |
| **`QUAL`** | Float | **Phred-scaled Quality**: Confidence that a polymorphism exists at this position. |
| **`CADD_PHRED`** | Float `[0, 35]` | **Combined Annotation Dependent Depletion**: Integrates diverse genomic annotations; scores > 20 indicate high deleteriousness. |
| **`SIFT_score`** | Float `[0, 1]` | **Sorting Intolerant From Tolerant**: Predicts amino acid substitution effects; scores < 0.05 are deleterious. |
| **`PolyPhen_score`** | Float `[0, 1]` | **Polymorphism Phenotyping v2**: Structural impact on proteins; scores > 0.85 indicate probably damaging variants. |

---

## 📊 Genomic & Bioinformatic Metrics Explained

### Assembly Metrics
- **N50**: The sequence length of the shortest contig at 50% of the total assembly length. A higher N50 reflects a more contiguous, high-quality assembly.
- **L50**: The minimum number of contigs whose cumulative length represents 50% of the entire genome. Lower L50 values indicate fewer, larger contigs.
- **GC Content**: The proportion of nitrogenous bases that are either Guanine (G) or Cytosine (C). Deviations can highlight contamination or sequencing bias.

### Functional Variant Classifications
- **Missense**: An SNP that alters a codon, resulting in an amino acid substitution in the synthesized protein.
- **Synonymous**: A silent mutation where the changed codon still codes for the same amino acid.
- **Nonsense**: A mutation introducing a premature stop codon, leading to protein truncation.
- **Splice Site**: A mutation at intron-exon junctions that alters mRNA splicing.
- **Intronic**: A mutation within a non-coding region, typically benign unless regulatory elements are disrupted.

---

## 📁 Repository Structure

```plaintext
ML_Proj_Genome_Analysis_Pipeline/
├── assets/                     # Publication-ready diagnostic and evaluation plots
│   ├── training_history.png    # Loss and accuracy learning trajectories
│   ├── roc_curve.png           # Receiver Operating Characteristic (ROC-AUC) curve
│   ├── confusion_matrix.png    # Test cohort confusion matrix
│   └── predictions_distribution.png # Pathogenicity score bimodal distribution
├── src/                        # Modular, production-grade source code
│   ├── __init__.py             # Package descriptor
│   ├── preprocessor.py         # FASTQ parsing, quality scoring & read filtering
│   ├── assembler.py            # De novo contig assembly & N50, L50, GC content algorithms
│   ├── variant_caller.py       # Variant detection & functional annotation
│   ├── dataset.py              # Population-scale genomic data synthesizer
│   ├── predictor.py            # Hybrid MLP + CNN-LSTM deep learning classifier
│   └── visualizer.py           # Seaborn & Matplotlib diagnostic plotting engine
├── tests/                      # Automated unit test suite (compatible with unittest & pytest)
│   ├── __init__.py
│   └── test_pipeline.py        # Algorithmic & dimensional validation tests
├── docs/                       # Project documentation & architectural blueprints
│   └── Project_Specification_and_Architecture.pdf
├── main.py                     # CLI entry point & end-to-end pipeline orchestrator
├── requirements.txt            # Pinned dependency requirements
├── .gitignore                  # Git ignore rules for virtualenvs, caches, and checkpoints
└── Readme.md                   # Comprehensive project documentation
```

---

## ⚙️ Installation & Prerequisites

### Prerequisites
- **Python**: Version **3.9**, **3.10**, **3.11**, or **3.12** is recommended (TensorFlow wheels are readily supported across these versions).
- Operating System: Windows, macOS, or Linux.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jainparth22/MLgenomepipeline.git
   cd MLgenomepipeline
   ```

2. **Create and activate a virtual environment**:
   - **On Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **On Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 Usage & Quick Start

Execute the complete end-to-end pipeline with:

```bash
python main.py
```

The script will sequentially:
1. Synthesize a 10,000-sample balanced variant dataset.
2. Execute simulated preprocessing, *de novo* assembly, quality reporting, and variant annotation.
3. Split the data (80% train, 20% test) and scale features using `StandardScaler`.
4. Train the dual-branch **MLP + CNN-LSTM** model for up to 50 epochs with `EarlyStopping`.
5. Display test accuracy, loss, and launch 4 interactive visualization windows.

---

## 🖥 Pipeline Execution Walkthrough

### 1. Preprocessed Data
```plaintext
Preprocessing sample_data.fastq...

Preprocessed data sample:
  sequence  quality_score
0     ATCG             30
1     GCTA             25
2     TGCA             35

Total sequences: 3
```

### 2. Assembly & Contig Generation
```plaintext
Assembling genome from 3 sequences...

Assembled 3 contigs
Longest contig: ATCGGCTATGCA
```

### 3. Assembly Quality Report
```plaintext
Metric           Value
N50              50000.00
L50                100.00
Total length   5000000.00
GC content           0.45
```

### 4. Variant Calling & Annotation
```plaintext
CHROM   POS REF ALT   ANNOTATION
 chr1  1000   A   T     missense
 chr1  2000   C   G   synonymous
 chr2  3000   G   T     nonsense
 chr3  4000   T   C     intronic
 chr4  5000   G   A  splice_site
```

### 5. Pathogenicity Model Training & Accuracy
```plaintext
Model: "model"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
input_mlp (InputLayer)       [(None, 6)]               0         
input_cnn (InputLayer)       [(None, 6, 1)]            0         
conv1d (Conv1D)             [(None, 4, 32)]           128       
dense_mlp1 (Dense)          [(None, 64)]              448       
conv1d_1 (Conv1D)           [(None, 2, 64)]           6208      
dropout (Dropout)           [(None, 64)]              0         
lstm (LSTM)                 [(None, 32)]              12416     
dense_mlp2 (Dense)          [(None, 32)]              2080      
concatenate (Concatenate)   [(None, 64)]              0         
dense_out (Dense)           [(None, 1)]               65        
=================================================================
Total params: 21,345
```

---

## 📈 Visual Diagnostics & Model Evaluation

Upon training completion, the pipeline automatically generates publication-ready diagnostic figures saved into the `assets/` directory:

### 1. Training History & Learning Trajectory
![Training Loss and Accuracy](assets/training_history.png)
*Figure 1: Cross-entropy loss (left) and classification accuracy (right) across epochs, demonstrating smooth convergence with no overfitting divergence.*

---

### 2. Receiver Operating Characteristic (ROC) & Confusion Matrix

| ROC-AUC Performance Curve | Test Cohort Confusion Matrix |
| :---: | :---: |
| ![ROC Curve](assets/roc_curve.png) | ![Confusion Matrix](assets/confusion_matrix.png) |
| *Figure 2: Receiver Operating Characteristic (ROC) curve yielding high diagnostic discrimination.* | *Figure 3: Confusion matrix on unseen test variants demonstrating high sensitivity and specificity.* |

---

### 3. Pathogenicity Probability Score Distribution
![Pathogenicity Prediction Distribution](assets/predictions_distribution.png)
*Figure 4: Bimodal distribution of pathogenicity probabilities $P(Y=1)$ for benign (green) vs. pathogenic (red) variants, highlighting strong polarization away from the decision boundary ($0.50$).*

---

## 🧪 Automated Testing

This repository includes a unit test suite validating algorithmic logic (such as deterministic N50/L50 and GC content computation) and input/output bounds:

```bash
# Run tests using Python's built-in test runner
python -m unittest discover -s tests -p "test_*.py"

# Or if pytest is installed
pytest tests/ -v
```

---

## 🗺 Project Roadmap & Future Enhancements

As outlined in the [Project Specification PDF](docs/Project_Specification_and_Architecture.pdf), the project roadmap comprises six development phases:

- [x] **Phase 1: Research & Prototype Architecture** (Current status: modular OOP structure, deep learning model, and visualization suite).
- [ ] **Phase 2: Live Tool Integration**:
  - Integrate [`Biopython`](https://biopython.org/) and [`pysam`](https://github.com/pysam-developers/pysam) for direct FASTQ/BAM parsing.
  - Implement a Python wrapper for [`SPAdes`](https://github.com/ablab/spades) for real *de novo* assembly.
- [ ] **Phase 3: Automated Quality & Variant Calling**:
  - Integrate [`QUAST`](https://github.com/ablab/quast) API for assembly validation.
  - Support [`cyvcf2`](https://github.com/brentp/cyvcf2) / `PyVCF` to parse standard VCF 4.2 files.
- [ ] **Phase 4: Real-world Clinical Databases**:
  - Connect with ClinVar, gnomAD, and dbSNP APIs for live variant impact lookup.
- [ ] **Phase 5: Flask Web Dashboard**:
  - Develop an interactive web interface allowing users to upload VCF/FASTQ files and inspect interactive Plotly graphs.
- [ ] **Phase 6: Containerization**:
  - Provide a Docker image for reproducible execution across genomic high-performance computing (HPC) clusters.

---

## 🤝 Contributing

Contributions, feedback, and issue submissions are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/BioinformaticFeature`).
3. Commit your Changes (`git commit -m 'Add new genome assembly metric'`).
4. Push to the Branch (`git push origin feature/BioinformaticFeature`).
5. Open a Pull Request.

---

## 📄 License

This project is distributed under the **MIT License**. See `LICENSE` for further details.

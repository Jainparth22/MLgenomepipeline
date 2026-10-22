"""
Genomic Dataset Generator & Synthesizer.
Produces realistic population-scale variant datasets with clinical and functional scores.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def create_sample_dataset(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic population-scale genomic variant dataset with realistic feature distributions:
    - AF: Allele Frequency (Beta distribution)
    - DP: Read Depth (Negative Binomial distribution)
    - QUAL: Phred Quality (Gamma distribution)
    - CADD_PHRED: In-silico deleteriousness score [0, 35]
    - SIFT_score: Protein tolerance score [0, 1]
    - PolyPhen_score: Structural damage score [0, 1]
    - is_pathogenic: Binary target (0 = benign, 1 = pathogenic)

    Args:
        n_samples: Number of variant records to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing annotated genomic variants.
    """
    np.random.seed(random_seed)

    chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr7', 'chr11', 'chr17', 'chrX']
    bases = ['A', 'C', 'G', 'T']

    af = np.random.beta(0.5, 2.0, n_samples)
    dp = np.random.negative_binomial(20, 0.3, n_samples) + 10
    qual = np.random.gamma(30, 2, n_samples)
    cadd = np.clip(np.random.exponential(7, n_samples), 0, 35)
    sift = np.random.beta(0.8, 1.2, n_samples)
    polyphen = np.random.beta(1.2, 0.8, n_samples)

    # Heuristic probabilistic ground truth for pathogenicity
    logit = (
        0.24 * (cadd - 18.0)
        - 3.8 * (sift - 0.10)
        + 3.6 * (polyphen - 0.70)
        - 2.8 * (af - 0.05)
    )
    prob_pathogenic = 1.0 / (1.0 + np.exp(-logit))
    is_pathogenic = (np.random.rand(n_samples) < prob_pathogenic).astype(int)

    data = pd.DataFrame({
        'CHROM': np.random.choice(chromosomes, n_samples),
        'POS': np.random.randint(10000, 150000000, n_samples),
        'REF': np.random.choice(bases, n_samples),
        'ALT': np.random.choice(bases, n_samples),
        'AF': np.round(af, 4),
        'DP': dp,
        'QUAL': np.round(qual, 2),
        'CADD_PHRED': np.round(cadd, 2),
        'SIFT_score': np.round(sift, 4),
        'PolyPhen_score': np.round(polyphen, 4),
        'is_pathogenic': is_pathogenic
    })

    return data

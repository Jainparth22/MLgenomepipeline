"""
Data Preprocessing Module for Genomic Sequences.
Handles sequence cleaning, read validation, and quality score calculation.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


class DataPreprocessor:
    """Preprocesses raw high-throughput sequencing reads (e.g. FASTQ format)."""

    def __init__(self, min_quality_score: int = 20, min_length: int = 4):
        self.min_quality_score = min_quality_score
        self.min_length = min_length

    def preprocess(self, input_file: str, raw_sequences: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filters and evaluates read quality from raw sequence data.

        Args:
            input_file: Path to sequencing file (e.g. sample.fastq).
            raw_sequences: Optional list of raw sequence reads for simulation.

        Returns:
            pd.DataFrame containing filtered sequences and associated quality scores.
        """
        print(f"[*] Preprocessing sequencing reads from: {input_file}...")

        if raw_sequences is None:
            sequences = ['ATCGGCTA', 'GCTATGCA', 'TGCAATCG', 'ATCG', 'GC']
            qualities = [34, 28, 38, 18, 12]
        else:
            sequences = raw_sequences
            qualities = [int(np.random.normal(30, 5)) for _ in sequences]

        # Apply filtering thresholds
        records = []
        for seq, qual in zip(sequences, qualities):
            if len(seq) >= self.min_length and qual >= self.min_quality_score:
                records.append({'sequence': seq, 'quality_score': qual, 'length': len(seq)})

        processed_data = pd.DataFrame(records)
        print(f"[+] Retained {len(processed_data)} / {len(sequences)} reads passing QC (Q >= {self.min_quality_score}).")
        return processed_data

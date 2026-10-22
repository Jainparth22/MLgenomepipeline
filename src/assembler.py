"""
Genome Assembly & Assembly Quality Assessment Module.
Constructs contigs and computes N50, L50, GC content, and contiguity statistics.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np


class GenomeAssembler:
    """Assembles sequence reads into contigs."""

    def assemble(self, processed_data: pd.DataFrame) -> List[str]:
        """
        Simulates de novo contig assembly from preprocessed sequence reads.

        Args:
            processed_data: DataFrame with validated sequence reads.

        Returns:
            List of assembled contig strings.
        """
        print(f"[*] Assembling genome from {len(processed_data)} preprocessed sequences...")
        contigs = [
            'ATCGGCTATGCAATCGGCTATGCA',
            'GCTATGCAATCGGCTATGCAATCG',
            'TGCAATCGGCTATGCA'
        ]
        longest = max(contigs, key=len)
        print(f"[+] Successfully assembled {len(contigs)} contigs. Longest contig: {len(longest)} bp")
        return contigs


class AssemblyQualityAssessor:
    """Computes assembly quality metrics including N50, L50, GC content, and cumulative length."""

    @staticmethod
    def calculate_n50_l50(contig_lengths: List[int]) -> Dict[str, Any]:
        """
        Calculates exact N50 and L50 metrics given a list of contig lengths.

        N50 is the length of the shortest contig at 50% of the total assembly length.
        L50 is the minimal number of contigs representing at least 50% of total length.
        """
        sorted_lengths = sorted(contig_lengths, reverse=True)
        total_len = sum(sorted_lengths)
        half_len = total_len / 2.0

        cum_len = 0
        n50 = 0
        l50 = 0

        for idx, length in enumerate(sorted_lengths, start=1):
            cum_len += length
            if cum_len >= half_len:
                n50 = length
                l50 = idx
                break

        return {'N50': n50, 'L50': l50, 'total_length': total_len}

    @staticmethod
    def calculate_gc_content(sequences: List[str]) -> float:
        """Calculates the proportion of Guanine (G) and Cytosine (C) across all sequences."""
        total_bases = 0
        gc_count = 0
        for seq in sequences:
            seq_upper = seq.upper()
            total_bases += len(seq_upper)
            gc_count += seq_upper.count('G') + seq_upper.count('C')
        return round(gc_count / total_bases, 4) if total_bases > 0 else 0.0

    def assess(self, contigs: List[str]) -> pd.DataFrame:
        """
        Generates a comprehensive assembly quality assessment report.
        """
        print(f"[*] Assessing quality of {len(contigs)} contigs...")
        lengths = [len(c) for c in contigs]
        metrics = self.calculate_n50_l50(lengths)
        gc_content = self.calculate_gc_content(contigs)

        report = pd.DataFrame({
            'Metric': ['Contig Count', 'Total Length (bp)', 'Longest Contig (bp)', 'N50 (bp)', 'L50 (contigs)', 'GC Content'],
            'Value': [len(contigs), metrics['total_length'], max(lengths), metrics['N50'], metrics['L50'], gc_content]
        })
        print("\n[+] Assembly Quality Report:")
        print(report.to_string(index=False))
        return report

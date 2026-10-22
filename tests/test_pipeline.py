"""
Automated Test Suite for Genome Analysis Pipeline.
Compatible with both `python -m unittest` and `pytest`.
"""

import unittest
import numpy as np
import pandas as pd
from src.preprocessor import DataPreprocessor
from src.assembler import GenomeAssembler, AssemblyQualityAssessor
from src.variant_caller import VariantCaller, VariantAnnotator
from src.dataset import create_sample_dataset
from src.predictor import PathogenicityPredictor


class TestGenomePipeline(unittest.TestCase):

    def test_data_preprocessor(self):
        preprocessor = DataPreprocessor(min_quality_score=20, min_length=4)
        df = preprocessor.preprocess("test.fastq", raw_sequences=['ATCG', 'GC', 'AAAAAA'])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertTrue(all(len(s) >= 4 for s in df['sequence']))

    def test_n50_l50_calculation(self):
        # Deterministic contig lengths: [50, 30, 20] -> Total = 100, Half = 50.
        res = AssemblyQualityAssessor.calculate_n50_l50([50, 30, 20])
        self.assertEqual(res['total_length'], 100)
        self.assertEqual(res['N50'], 50)
        self.assertEqual(res['L50'], 1)

        # [30, 25, 25, 20] -> Total = 100, Half = 50.
        # Contig 30 (cum 30), Contig 25 (cum 55 >= 50) -> N50 = 25, L50 = 2.
        res2 = AssemblyQualityAssessor.calculate_n50_l50([30, 25, 25, 20])
        self.assertEqual(res2['N50'], 25)
        self.assertEqual(res2['L50'], 2)

    def test_gc_content(self):
        # 'ATGC' -> 2 GC out of 4 bases = 0.50
        gc = AssemblyQualityAssessor.calculate_gc_content(['ATGC'])
        self.assertEqual(gc, 0.50)

        # 'GGCC' -> 4 GC out of 4 bases = 1.00
        gc_full = AssemblyQualityAssessor.calculate_gc_content(['GGCC'])
        self.assertEqual(gc_full, 1.00)

    def test_dataset_generation(self):
        n_samples = 500
        df = create_sample_dataset(n_samples=n_samples, random_seed=42)
        self.assertEqual(len(df), n_samples)
        required_cols = ['CHROM', 'POS', 'REF', 'ALT', 'AF', 'DP', 'QUAL', 'CADD_PHRED', 'SIFT_score', 'PolyPhen_score', 'is_pathogenic']
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertTrue(set(df['is_pathogenic'].unique()).issubset({0, 1}))

    def test_pathogenicity_predictor_flow(self):
        df = create_sample_dataset(n_samples=300, random_seed=42)
        predictor = PathogenicityPredictor()
        X_train, X_train_cnn, X_test, X_test_cnn, y_train, y_test = predictor.prepare_data(df)

        predictor.build_model(X_train.shape[1])
        predictor.train_model(X_train, X_train_cnn, y_train, X_test, X_test_cnn, y_test, epochs=2)

        loss, acc = predictor.evaluate(X_test, X_test_cnn, y_test)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        preds = predictor.predict(df[predictor.FEATURES])
        self.assertEqual(len(preds), len(df))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(preds <= 1.0))


if __name__ == '__main__':
    unittest.main()

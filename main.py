"""
Genome Analysis Pipeline & Pathogenic Variant Identification
Command-line Interface & Main Orchestrator.
"""

import sys
import argparse

# Ensure cross-platform safe encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.preprocessor import DataPreprocessor
from src.assembler import GenomeAssembler, AssemblyQualityAssessor
from src.variant_caller import VariantCaller, VariantAnnotator
from src.dataset import create_sample_dataset
from src.predictor import PathogenicityPredictor
from src.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Genome Assembly Analysis & Pathogenic Variant Identification Pipeline"
    )
    parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic genomic variants to generate (default: 10000)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs for the deep learning model (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model training (default: 32)")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive plot display (useful in headless/CI environments)")
    parser.add_argument("--save-assets", action="store_true", default=True, help="Save diagnostic plots to assets/ directory")
    parser.add_argument("--output-dir", type=str, default="assets", help="Directory to save generated plots")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("GENOME ANALYSIS & PATHOGENIC VARIANT IDENTIFICATION PIPELINE")
    print("=" * 80)

    # 1. Dataset Generation
    print(f"\n[Stage 1/6] Generating population-scale genomic variant cohort ({args.samples:,} samples)...")
    data = create_sample_dataset(n_samples=args.samples)
    print(f"[+] Dataset created with {len(data)} variants.")
    print("    Sample preview:")
    print(data[['CHROM', 'POS', 'REF', 'ALT', 'CADD_PHRED', 'SIFT_score', 'PolyPhen_score', 'is_pathogenic']].head())

    # 2. Sequence Preprocessing
    print("\n[Stage 2/6] Sequence Preprocessing & Quality Filtering...")
    preprocessor = DataPreprocessor(min_quality_score=20, min_length=4)
    processed_reads = preprocessor.preprocess("sample_reads.fastq")

    # 3. Genome Assembly & Quality Assessment
    print("\n[Stage 3/6] De Novo Contig Assembly & Contiguity Evaluation...")
    assembler = GenomeAssembler()
    contigs = assembler.assemble(processed_reads)

    quality_assessor = AssemblyQualityAssessor()
    quality_report = quality_assessor.assess(contigs)

    # 4. Variant Calling & Annotation
    print("\n[Stage 4/6] Variant Calling & Functional Consequence Annotation...")
    caller = VariantCaller()
    variants = caller.call_variants(contigs, reference_genome="GRCh38")

    annotator = VariantAnnotator()
    annotated_variants = annotator.annotate(variants)

    # 5. Machine Learning Pathogenicity Prediction
    print("\n[Stage 5/6] Training Pathogenicity Deep Learning Model...")
    predictor = PathogenicityPredictor()
    X_train, X_train_cnn, X_test, X_test_cnn, y_train, y_test = predictor.prepare_data(data)

    predictor.build_model(input_shape=X_train.shape[1])
    predictor.train_model(
        X_train, X_train_cnn, y_train,
        X_test, X_test_cnn, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # 6. Evaluation & Predictions
    print("\n[Stage 6/6] Model Evaluation & Clinical Diagnostics...")
    test_loss, test_accuracy = predictor.evaluate(X_test, X_test_cnn, y_test)
    print(f"[+] Test Accuracy: {test_accuracy * 100:.2f}% | Test Loss: {test_loss:.4f}")

    print("\n[*] Generating variant pathogenicity probability distribution...")
    predictions = predictor.predict(data[predictor.FEATURES])

    # Visualizations
    visualizer = Visualizer(output_dir=args.output_dir if args.save_assets else None)
    show_plots = not args.no_plot

    print("[*] Rendering diagnostic visualizations...")
    visualizer.plot_training_history(predictor.history, show=show_plots, save=args.save_assets)
    visualizer.plot_predictions_distribution(predictions, y_true=data['is_pathogenic'].values, show=show_plots, save=args.save_assets)
    visualizer.plot_roc_curve(y_test, predictor.predict(X_test), show=show_plots, save=args.save_assets)
    visualizer.plot_confusion_matrix(y_test, predictor.predict(X_test), show=show_plots, save=args.save_assets)

    print("\n" + "=" * 80)
    print("[SUCCESS] Pipeline executed successfully! All reports and diagnostic plots generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()

# pip install pandas numpy matplotlib seaborn scikit-learn tensorflow 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

class DataPreprocessor:
    def preprocess(self, input_file):
        print(f"Preprocessing {input_file}...")
        # Simulating preprocessing
        processed_data = pd.DataFrame({
            'sequence': ['ATCG', 'GCTA', 'TGCA'],
            'quality_score': [30, 25, 35]
        })
        print("\nPreprocessed data sample:")
        print(processed_data.head())
        print(f"\nTotal sequences: {len(processed_data)}")
        return processed_data

class GenomeAssembler:
    def assemble(self, processed_data):
        print(f"Assembling genome from {len(processed_data)} sequences...")
        # Simulating assembly
        contigs = ['ATCGGCTATGCA', 'GCTATGCAATCG', 'TGCAATCGGCTA']
        print(f"\nAssembled {len(contigs)} contigs")
        print(f"Longest contig: {max(contigs, key=len)}")
        return contigs

class AssemblyQualityAssessor:
    def assess(self, assembly):
        print(f"Assessing quality of assembly with {len(assembly)} contigs...")
        # Simulating quality assessment
        quality_report = pd.DataFrame({
            'Metric': ['N50', 'L50', 'Total length', 'GC content'],
            'Value': [50000, 100, 5000000, 0.45]
        })
        print("\nAssembly quality report:")
        print(quality_report)
        return quality_report

class VariantCaller:
    def call_variants(self, assembly, reference_genome):
        print(f"Calling variants for assembly against {reference_genome}...")
        # Simulating variant calling
        variants = pd.DataFrame({
            'CHROM': ['chr1', 'chr1', 'chr2', 'chr3', 'chr4'],
            'POS': [1000, 2000, 3000, 4000, 5000],
            'REF': ['A', 'C', 'G', 'T', 'G'],
            'ALT': ['T', 'G', 'T', 'C', 'A']
        })
        print("\nVariant calling results:")
        print(variants)
        print(f"\nTotal variants called: {len(variants)}")
        return variants

class VariantAnnotator:
    def annotate(self, variants):
        print("Annotating variants...")
        # Simulating annotation
        variants['ANNOTATION'] = ['missense', 'synonymous', 'nonsense', 'intronic', 'splice_site']
        print("\nAnnotated variants:")
        print(variants)
        print("\nAnnotation summary:")
        print(variants['ANNOTATION'].value_counts())
        return variants

class PathogenicityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def prepare_data(self, data):
        features = ['AF', 'DP', 'QUAL', 'CADD_PHRED', 'SIFT_score', 'PolyPhen_score']
        X = data[features]
        y = data['is_pathogenic']

        X_scaled = self.scaler.fit_transform(X)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train_cnn, X_test_cnn, _, _ = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

        print("\nData preparation summary:")
        print(f"Total samples: {len(data)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Pathogenic variants: {sum(y)}")
        print(f"Benign variants: {len(y) - sum(y)}")

        return X_train, X_train_cnn, X_test, X_test_cnn, y_train, y_test

    def build_model(self, input_shape):
        input_mlp = Input(shape=(input_shape,))
        input_cnn = Input(shape=(input_shape, 1))

        x_mlp = Dense(64, activation='relu')(input_mlp)
        x_mlp = Dropout(0.3)(x_mlp)
        x_mlp = Dense(32, activation='relu')(x_mlp)

        x_cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(input_cnn)
        x_cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(x_cnn)
        x_cnn = LSTM(32)(x_cnn)

        combined = concatenate([x_mlp, x_cnn])
        output = Dense(1, activation='sigmoid')(combined)

        self.model = Model(inputs=[input_mlp, input_cnn], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        print("\nModel architecture:")
        self.model.summary()

    def train_model(self, X_train, X_train_cnn, y_train, X_val, X_val_cnn, y_val, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.history = self.model.fit(
            [X_train, X_train_cnn], y_train,
            validation_data=([X_val, X_val_cnn], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        return self.model.predict([X_scaled, X_cnn])

    def evaluate(self, X_test, X_test_cnn, y_test):
        return self.model.evaluate([X_test, X_test_cnn], y_test)

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Model Loss Over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        plt.title('Model Accuracy Over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

        print("\nTraining History Explanation:")
        print("- The left graph shows how the model's loss (error) decreases over time for both training and validation data.")
        print("- The right graph shows how the model's accuracy improves over time for both training and validation data.")
        print("- Ideally, we want to see both loss decreasing and accuracy increasing, with the training and validation lines close together.")
        print("- If the validation line starts to diverge significantly from the training line, it may indicate overfitting.")

    @staticmethod
    def plot_predictions_distribution(predictions):
        plt.figure(figsize=(12, 6))
        sns.histplot(predictions, bins=50, kde=True, color='purple')
        plt.title('Distribution of Pathogenicity Predictions', fontsize=14)
        plt.xlabel('Predicted Probability of Pathogenicity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        print("\nPredictions Distribution Explanation:")
        print("- This histogram shows the distribution of pathogenicity predictions for all variants.")
        print("- The x-axis represents the predicted probability of a variant being pathogenic (0 to 1).")
        print("- The y-axis shows how many variants received each prediction score.")
        print("- A peak on the left indicates more benign predictions, while a peak on the right indicates more pathogenic predictions.")
        print("- The shape of this distribution can help identify if the model is biased towards certain predictions.")

    @staticmethod
    def plot_roc_curve(y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        print("\nROC Curve Explanation:")
        print("- The ROC curve shows the performance of the classification model at all threshold settings.")
        print("- The x-axis represents the False Positive Rate, and the y-axis represents the True Positive Rate.")
        print("- The diagonal dashed line represents random guessing (AUC = 0.5).")
        print("- The closer the curve follows the top-left corner, the better the model's performance.")
        print(f"- The Area Under the Curve (AUC) of {roc_auc:.2f} indicates the model's overall performance.")
        print("- An AUC of 1.0 represents a perfect model, while 0.5 represents random guessing.")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred.round())
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.show()

        print("\nConfusion Matrix Explanation:")
        print("- The confusion matrix shows the model's performance in terms of True Positives, True Negatives, False Positives, and False Negatives.")
        print("- The rows represent the true labels, and the columns represent the predicted labels.")
        print("- The diagonal elements represent correct predictions, while off-diagonal elements are incorrect predictions.")
        print("- This matrix helps identify which types of errors the model is making most frequently.")

def create_sample_dataset(n_samples=10000):
    np.random.seed(42)

    data = pd.DataFrame({
        'CHROM': np.random.choice(['chr1', 'chr2', 'chr3', 'chr4', 'chr5'], n_samples),
        'POS': np.random.randint(1, 1000000, n_samples),
        'REF': np.random.choice(['A', 'C', 'G', 'T'], n_samples),
        'ALT': np.random.choice(['A', 'C', 'G', 'T'], n_samples),
        'AF': np.random.uniform(0, 1, n_samples),
        'DP': np.random.randint(10, 100, n_samples),
        'QUAL': np.random.uniform(0, 100, n_samples),
        'CADD_PHRED': np.random.uniform(0, 35, n_samples),
        'SIFT_score': np.random.uniform(0, 1, n_samples),
        'PolyPhen_score': np.random.uniform(0, 1, n_samples)
    })

    data['is_pathogenic'] = (data['CADD_PHRED'] > 20) & (data['SIFT_score'] < 0.05) & (data['PolyPhen_score'] > 0.95)
    data['is_pathogenic'] = data['is_pathogenic'].astype(int)

    print("\nSample dataset summary:")
    print(data.describe())
    print("\nSample data (first 5 rows):")
    print(data.head())

    return data

def main():
    # Create sample dataset
    print("Creating sample dataset...")
    data = create_sample_dataset()

    # Initialize components
    preprocessor = DataPreprocessor()
    assembler = GenomeAssembler()
    quality_assessor = AssemblyQualityAssessor()
    variant_caller = VariantCaller()
    annotator = VariantAnnotator()
    predictor = PathogenicityPredictor()
    visualizer = Visualizer()

    # Preprocess data (simulated)
    processed_data = preprocessor.preprocess("sample_data.fastq")

    # Genome assembly (simulated)
    assembly = assembler.assemble(processed_data)

    # Quality assessment (simulated)
    quality_report = quality_assessor.assess(assembly)

    # Variant calling (simulated)
    variants = variant_caller.call_variants(assembly, "reference_genome.fasta")

    # Variant annotation (simulated)
    annotated_variants = annotator.annotate(variants)

    # Pathogenicity prediction
    print("\nTraining pathogenicity prediction model...")
    X_train, X_train_cnn, X_test, X_test_cnn, y_train, y_test = predictor.prepare_data(data)
    predictor.build_model(X_train.shape[1])
    predictor.train_model(X_train, X_train_cnn, y_train, X_test, X_test_cnn, y_test, epochs=50)

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = predictor.evaluate(X_test, X_test_cnn, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(data[['AF', 'DP', 'QUAL', 'CADD_PHRED', 'SIFT_score', 'PolyPhen_score']])

    # Visualizations
    print("\nGenerating visualizations...")
    visualizer.plot_training_history(predictor.history)
    visualizer.plot_predictions_distribution(predictions)
    visualizer.plot_roc_curve(y_test, predictor.predict(X_test))
    visualizer.plot_confusion_matrix(y_test, predictor.predict(X_test))

    print("\nGenome analysis pipeline completed successfully!")

if __name__ == "__main__":
    main()



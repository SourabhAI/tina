#!/usr/bin/env python3
"""
Train the supervised intent classifier using weak supervision labels.

This script:
1. Loads training data
2. Applies labeling functions to generate weak labels
3. Trains the supervised classifier
4. Evaluates performance on validation set
5. Saves the trained model
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter, defaultdict

from intent_classifier.main import IntentClassificationPipeline, PipelineConfig
from intent_classifier.core.classifier import IntentClassifier, ClassifierConfig
from intent_classifier.core.labeling_functions import LabelingFunctions
from intent_classifier.core.entity_extractor import EntityExtractor
from intent_classifier.core.taxonomy_loader import TaxonomyLoader
from intent_classifier.core.preprocessor import Preprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Pipeline for training the intent classifier."""
    
    def __init__(self, config: ClassifierConfig = None):
        """Initialize training pipeline."""
        self.config = config or ClassifierConfig()
        
        # Initialize components
        logger.info("Initializing training components...")
        self.taxonomy = TaxonomyLoader('taxonomy.json')
        self.preprocessor = Preprocessor(enable_spell_correction=False)
        
        # Get entity definitions from taxonomy
        entity_definitions = self.taxonomy.get_entity_definitions()
        self.entity_extractor = EntityExtractor(entity_definitions)
        self.labeling_functions = LabelingFunctions(self.taxonomy, self.entity_extractor)
        
        # Get entity keys from taxonomy
        entity_schemas = self.taxonomy.get_entity_definitions()
        entity_keys = [schema['key'] for schema in entity_schemas]
        
        # Initialize classifier
        self.classifier = IntentClassifier(self.config, entity_keys)
        
        logger.info("Training components initialized")
    
    def load_data(self, data_file: str) -> List[Dict]:
        """Load training/validation data."""
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def generate_weak_labels(self, data: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Generate weak supervision labels for training data.
        
        Returns:
            texts: List of preprocessed texts
            labels: List of intent labels
            features: List of feature dictionaries
        """
        logger.info("Generating weak supervision labels...")
        
        texts = []
        labels = []
        features = []
        label_stats = defaultdict(int)
        no_label_examples = []
        
        for i, example in enumerate(data):
            # Preprocess text
            text = example.get('question_text', '')
            preprocessed_text = self.preprocessor.process(text)
            
            # Extract entities
            entity_result = self.entity_extractor.extract(preprocessed_text)
            entities = entity_result.entities if hasattr(entity_result, 'entities') else entity_result
            
            # Apply labeling functions
            votes = self.labeling_functions.apply_all(preprocessed_text, entities)
            
            if votes:
                # Get top prediction
                top_predictions = self.labeling_functions.get_top_predictions(votes, top_k=1)
                if top_predictions:
                    intent_code, confidence = top_predictions[0]
                    
                    # Only use high-confidence labels for training
                    if confidence >= 0.7:
                        texts.append(preprocessed_text)
                        labels.append(intent_code)
                        features.append({
                            'entities': entities,
                            'original_text': text,
                            'confidence': confidence
                        })
                        label_stats[intent_code] += 1
                    else:
                        no_label_examples.append({
                            'text': text,
                            'intent': intent_code,
                            'confidence': confidence
                        })
                else:
                    no_label_examples.append({'text': text, 'reason': 'no_predictions'})
            else:
                no_label_examples.append({'text': text, 'reason': 'no_votes'})
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(data)} examples")
        
        logger.info(f"Generated {len(labels)} weak labels from {len(data)} examples")
        logger.info(f"Skipped {len(no_label_examples)} examples (no confident label)")
        
        # Log label distribution
        logger.info("\nLabel distribution:")
        for intent, count in sorted(label_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {intent}: {count}")
        
        # Log some examples without labels
        if no_label_examples:
            logger.info("\nSample examples without confident labels:")
            for ex in no_label_examples[:5]:
                if 'confidence' in ex:
                    logger.info(f"  '{ex['text'][:60]}...' -> {ex.get('intent', 'N/A')} (conf: {ex['confidence']:.2f})")
                else:
                    logger.info(f"  '{ex['text'][:60]}...' -> No votes")
        
        return texts, labels, features
    
    def train(self, train_file: str, val_file: str = None):
        """Train the classifier."""
        # Load training data
        train_data = self.load_data(train_file)
        
        # Generate weak labels
        train_texts, train_labels, train_features = self.generate_weak_labels(train_data)
        
        if len(train_texts) == 0:
            logger.error("No training examples with confident labels!")
            return
        
        # Extract entity features for training
        train_entities = [f['entities'] for f in train_features]
        
        # Encode labels
        unique_labels = sorted(set(train_labels))
        label_encoder = {label: i for i, label in enumerate(unique_labels)}
        encoded_labels = [label_encoder[label] for label in train_labels]
        
        # Set the label encoder/decoder in classifier
        self.classifier.label_encoder = label_encoder
        self.classifier.label_decoder = {i: label for label, i in label_encoder.items()}
        
        # Train classifier
        logger.info("\nTraining classifier...")
        start_time = time.time()
        
        self.classifier.train(
            texts=train_texts,
            labels=encoded_labels,
            entities_list=train_entities
        )
        
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate on validation set if provided
        if val_file:
            self.evaluate(val_file)
        
        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "intent_classifier.pkl"
        
        logger.info(f"\nSaving model to {model_path}")
        self.classifier.save(str(model_path))
        logger.info("Model saved successfully")
        
        # Generate training report
        self.generate_training_report(train_texts, train_labels, train_features)
    
    def evaluate(self, val_file: str):
        """Evaluate classifier on validation set."""
        logger.info("\nEvaluating on validation set...")
        
        # Load validation data
        val_data = self.load_data(val_file)
        
        # Generate predictions
        predictions = []
        true_labels = []
        
        for example in val_data:
            text = example.get('question_text', '')
            preprocessed_text = self.preprocessor.process(text)
            entity_result = self.entity_extractor.extract(preprocessed_text)
            entities = entity_result.entities if hasattr(entity_result, 'entities') else entity_result
            
            # Get weak label as ground truth
            votes = self.labeling_functions.apply_all(preprocessed_text, entities)
            if votes:
                top_predictions = self.labeling_functions.get_top_predictions(votes, top_k=1)
                if top_predictions and top_predictions[0][1] >= 0.7:
                    true_label = top_predictions[0][0]
                    
                    # Get classifier prediction
                    pred_intent, pred_conf = self.classifier.predict(preprocessed_text, entities)
                    
                    predictions.append(pred_intent)
                    true_labels.append(true_label)
        
        if predictions:
            # Calculate metrics
            accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
            logger.info(f"\nValidation Accuracy: {accuracy:.3f}")
            
            # Classification report
            logger.info("\nClassification Report:")
            report = classification_report(true_labels, predictions, zero_division=0)
            logger.info("\n" + report)
            
            # Top misclassifications
            misclassified = [(t, p) for t, p in zip(true_labels, predictions) if t != p]
            if misclassified:
                logger.info("\nTop Misclassifications:")
                mis_counter = Counter(misclassified)
                for (true, pred), count in mis_counter.most_common(10):
                    logger.info(f"  {true} -> {pred}: {count}")
        else:
            logger.warning("No validation examples with confident labels")
    
    def generate_training_report(self, texts: List[str], labels: List[str], features: List[Dict]):
        """Generate detailed training report."""
        report = {
            'training_stats': {
                'total_examples': len(texts),
                'unique_intents': len(set(labels)),
                'label_distribution': dict(Counter(labels)),
                'avg_text_length': np.mean([len(t.split()) for t in texts])
            },
            'intent_examples': defaultdict(list),
            'entity_coverage': defaultdict(int)
        }
        
        # Collect examples per intent
        for text, label, feature in zip(texts, labels, features):
            if len(report['intent_examples'][label]) < 3:  # Keep top 3 examples
                report['intent_examples'][label].append({
                    'text': feature['original_text'],
                    'confidence': feature['confidence'],
                    'entities': list(feature['entities'].keys())
                })
        
        # Entity coverage
        for feature in features:
            for entity_type in feature['entities']:
                report['entity_coverage'][entity_type] += 1
        
        # Save report
        report_path = Path("models") / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nTraining report saved to {report_path}")
        
        # Log summary
        logger.info("\nTraining Summary:")
        logger.info(f"  Total examples: {report['training_stats']['total_examples']}")
        logger.info(f"  Unique intents: {report['training_stats']['unique_intents']}")
        logger.info(f"  Avg text length: {report['training_stats']['avg_text_length']:.1f} words")
        
        logger.info("\nTop 10 intent classes:")
        sorted_intents = sorted(
            report['training_stats']['label_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for intent, count in sorted_intents[:10]:
            logger.info(f"  {intent}: {count}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train intent classifier')
    parser.add_argument('--train-file', default='train_questions_labeled.json',
                       help='Training data file')
    parser.add_argument('--val-file', default='val_questions.json',
                       help='Validation data file')
    parser.add_argument('--model-type', default='logistic',
                       choices=['logistic', 'svm', 'random_forest'],
                       help='Model type to train')
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Maximum number of TF-IDF features')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Configure classifier
    config = ClassifierConfig(
        model_type=args.model_type,
        tfidf_max_features=args.max_features,
        cross_validation_folds=args.cv_folds,
        random_state=42
    )
    
    # Create training pipeline
    pipeline = TrainingPipeline(config)
    
    # Train classifier
    pipeline.train(args.train_file, args.val_file)
    
    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

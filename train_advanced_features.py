#!/usr/bin/env python3
"""
Train advanced features for the intent classification pipeline:
1. KNN Backstop for similar query matching
2. Confidence Calibration for better probability estimates
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

from intent_classifier.main import IntentClassificationPipeline, PipelineConfig
from intent_classifier.core.knn_backstop import KNNBackstop, KNNConfig, QueryExample
from intent_classifier.core.confidence import ConfidenceCalibrator, CalibrationConfig, CalibrationSample
from intent_classifier.core.preprocessor import Preprocessor
from intent_classifier.core.entity_extractor import EntityExtractor
from intent_classifier.core.taxonomy_loader import TaxonomyLoader
from intent_classifier.core.labeling_functions import LabelingFunctions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedFeatureTrainer:
    """Trains KNN backstop and confidence calibration."""
    
    def __init__(self):
        """Initialize trainer."""
        # Initialize components
        logger.info("Initializing components...")
        self.taxonomy = TaxonomyLoader('taxonomy.json')
        self.preprocessor = Preprocessor(enable_spell_correction=False)
        
        # Get entity definitions from taxonomy
        entity_definitions = self.taxonomy.get_entity_definitions()
        self.entity_extractor = EntityExtractor(entity_definitions)
        self.labeling_functions = LabelingFunctions(self.taxonomy, self.entity_extractor)
        
        # Model directory
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info("Components initialized")
    
    def load_data(self, data_file: str) -> List[Dict]:
        """Load data from file."""
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def prepare_examples(self, data: List[Dict]) -> List[Tuple[str, str, Dict, float]]:
        """
        Prepare examples for KNN training.
        
        Returns:
            List of (text, intent, entities, confidence) tuples
        """
        logger.info("Preparing examples...")
        examples = []
        
        for i, example in enumerate(data):
            # Get text
            text = example.get('question_text', '')
            preprocessed_text = self.preprocessor.process(text)
            
            # Extract entities
            entity_result = self.entity_extractor.extract(preprocessed_text)
            entities = entity_result.entities if hasattr(entity_result, 'entities') else entity_result
            
            # Get label from labeling functions
            votes = self.labeling_functions.apply_all(preprocessed_text, entities)
            
            if votes:
                # Get top prediction
                top_predictions = self.labeling_functions.get_top_predictions(votes, top_k=1)
                if top_predictions and top_predictions[0][1] >= 0.7:
                    intent_code, confidence = top_predictions[0]
                    examples.append((preprocessed_text, intent_code, entities, confidence))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(data)} examples")
        
        logger.info(f"Prepared {len(examples)} examples with confident labels")
        return examples
    
    def train_knn_backstop(self, train_file: str, val_file: str = None):
        """Train KNN backstop model."""
        logger.info("\n" + "="*60)
        logger.info("Training KNN Backstop")
        logger.info("="*60)
        
        # Load training data
        train_data = self.load_data(train_file)
        train_examples = self.prepare_examples(train_data)
        
        if len(train_examples) < 10:
            logger.error("Not enough training examples for KNN")
            return
        
        # Initialize KNN backstop
        config = KNNConfig(
            n_neighbors=5,
            min_similarity=0.3,
            max_features=3000,
            use_entity_similarity=True,
            entity_weight=0.3,
            text_weight=0.7
        )
        knn_backstop = KNNBackstop(config)
        
        # Add training examples
        logger.info("Adding examples to KNN index...")
        query_examples = []
        for text, intent, entities, confidence in train_examples:
            example = QueryExample(
                text=text,  # Using preprocessed text
                preprocessed_text=text,  # Already preprocessed
                entities=entities,
                intent_code=intent,
                confidence=confidence,
                source='training'
            )
            query_examples.append(example)
        
        knn_backstop.add_examples(query_examples)
        
        # Build index
        logger.info("Building KNN index...")
        knn_backstop.build_index()
        
        # Evaluate on validation set if provided
        if val_file:
            logger.info("\nEvaluating KNN on validation set...")
            val_data = self.load_data(val_file)
            val_examples = self.prepare_examples(val_data)
            
            correct = 0
            total = 0
            
            for text, true_intent, entities, _ in val_examples[:50]:  # Test on subset
                neighbors = knn_backstop.find_similar(text, entities, k=1)
                if neighbors:
                    pred_intent = neighbors[0].example.intent_code
                    if pred_intent == true_intent:
                        correct += 1
                total += 1
            
            if total > 0:
                accuracy = correct / total
                logger.info(f"KNN Validation Accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Save model
        model_path = self.model_dir / "knn_backstop.pkl"
        logger.info(f"\nSaving KNN model to {model_path}")
        knn_backstop.save(str(model_path))
        logger.info("KNN model saved successfully")
        
        # Log statistics
        stats = knn_backstop.get_statistics()
        logger.info(f"\nKNN Statistics:")
        logger.info(f"  Total examples: {stats['total_examples']}")
        logger.info(f"  Unique intents: {stats['unique_intents']}")
        if 'avg_examples_per_intent' in stats:
            logger.info(f"  Examples per intent: {stats['avg_examples_per_intent']:.1f}")
        else:
            avg_per_intent = stats['total_examples'] / stats['unique_intents'] if stats['unique_intents'] > 0 else 0
            logger.info(f"  Examples per intent: {avg_per_intent:.1f}")
    
    def train_confidence_calibration(self, train_file: str, val_file: str):
        """Train confidence calibration model."""
        logger.info("\n" + "="*60)
        logger.info("Training Confidence Calibration")
        logger.info("="*60)
        
        # We need to collect predictions and true labels
        # For this, we'll use the trained classifier
        try:
            # Initialize pipeline with classifier only
            pipeline_config = PipelineConfig(
                use_spacy_splitter=False,
                use_knn_backstop=False,
                use_confidence_calibration=False,
                enable_spell_correction=False
            )
            pipeline = IntentClassificationPipeline(pipeline_config)
            
            # Load models if not already loaded
            if not pipeline.classifier:
                pipeline.load_models()
                
            if not pipeline.classifier:
                logger.error("No trained classifier found. Train classifier first.")
                return
            
            # Collect calibration data
            logger.info("Collecting calibration data...")
            
            # Load validation data
            val_data = self.load_data(val_file)
            
            predictions = []
            true_labels = []
            confidences = []
            
            for example in val_data:
                text = example.get('question_text', '')
                
                # Get pipeline prediction
                result = pipeline.classify(text)
                
                if result.intents:
                    # Get predicted intent and confidence
                    pred_intent = result.intents[0]
                    pred_code = pred_intent.intent_code
                    pred_conf = pred_intent.confidence
                    
                    # Get true label from labeling functions
                    preprocessed_text = pipeline.preprocessor.process(text)
                    entity_result = pipeline.entity_extractor.extract(preprocessed_text)
                    entities = entity_result.entities if hasattr(entity_result, 'entities') else entity_result
                    votes = pipeline.weak_labeler.apply_all(preprocessed_text, entities)
                    
                    if votes:
                        top_predictions = pipeline.weak_labeler.get_top_predictions(votes, top_k=1)
                        if top_predictions and top_predictions[0][1] >= 0.7:
                            true_intent = top_predictions[0][0]
                            
                            predictions.append(pred_code)
                            true_labels.append(true_intent)
                            confidences.append(pred_conf)
            
            if len(predictions) < 20:
                logger.error("Not enough data for calibration")
                pipeline.shutdown()
                return
            
            # Initialize calibrator
            config = CalibrationConfig(
                method='isotonic',  # Best for small datasets
                n_bins=10,
                min_samples_per_bin=5
            )
            calibrator = ConfidenceCalibrator(config)
            
            # Create calibration samples
            calibration_samples = []
            for pred, true, conf in zip(predictions, true_labels, confidences):
                sample = CalibrationSample(
                    predicted_confidence=conf,
                    true_label=true,
                    predicted_label=pred,
                    is_correct=(pred == true)
                )
                calibration_samples.append(sample)
            
            # Train calibrator
            logger.info(f"Training calibrator with {len(calibration_samples)} examples...")
            calibrator.fit(calibration_samples)
            
            # Log calibration summary
            logger.info("\nCalibration Summary:")
            logger.info(f"  Method: {config.method}")
            logger.info(f"  Samples: {len(calibration_samples)}")
            logger.info(f"  Accuracy: {sum(s.is_correct for s in calibration_samples) / len(calibration_samples):.3f}")
            
            # Check calibration on a few examples
            logger.info("\nSample calibrations:")
            for i in range(min(5, len(calibration_samples))):
                sample = calibration_samples[i]
                calibrated_conf = calibrator.calibrate(sample.predicted_label, sample.predicted_confidence)
                logger.info(f"  {sample.predicted_label}: {sample.predicted_confidence:.3f} -> {calibrated_conf:.3f} (correct: {sample.is_correct})")
            
            # Save model
            model_path = self.model_dir / "confidence_calibrator.pkl"
            logger.info(f"\nSaving calibrator to {model_path}")
            calibrator.save(str(model_path))
            logger.info("Calibrator saved successfully")
            
            # Clean up
            pipeline.shutdown()
            
        except Exception as e:
            logger.error(f"Error during calibration training: {e}")
            raise
    
    def verify_models(self):
        """Verify that all models are properly saved."""
        logger.info("\n" + "="*60)
        logger.info("Verifying Saved Models")
        logger.info("="*60)
        
        models_to_check = [
            ("intent_classifier.pkl", "Supervised Classifier"),
            ("knn_backstop.pkl", "KNN Backstop"),
            ("confidence_calibrator.pkl", "Confidence Calibrator")
        ]
        
        all_present = True
        for filename, name in models_to_check:
            path = self.model_dir / filename
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ {name}: {path} ({size_mb:.1f} MB)")
            else:
                logger.warning(f"✗ {name}: NOT FOUND")
                all_present = False
        
        if all_present:
            logger.info("\n✓ All models are present and ready to use!")
        else:
            logger.warning("\n⚠ Some models are missing. Train them before enabling advanced features.")
        
        return all_present


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train advanced features')
    parser.add_argument('--train-file', default='train_questions.json',
                       help='Training data file')
    parser.add_argument('--val-file', default='val_questions.json',
                       help='Validation data file')
    parser.add_argument('--feature', choices=['knn', 'calibration', 'all'],
                       default='all', help='Which feature to train')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AdvancedFeatureTrainer()
    
    # Train requested features
    if args.feature in ['knn', 'all']:
        trainer.train_knn_backstop(args.train_file, args.val_file)
    
    if args.feature in ['calibration', 'all']:
        trainer.train_confidence_calibration(args.train_file, args.val_file)
    
    # Verify all models
    trainer.verify_models()
    
    logger.info("\n✓ Advanced feature training completed!")


if __name__ == "__main__":
    main()

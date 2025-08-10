"""
Confidence calibration module for intent classification.
Calibrates prediction confidence scores to be more reliable and interpretable.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import joblib

from intent_classifier.models.schemas import Intent


logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration."""
    method: str = "isotonic"  # "isotonic", "platt", "temperature", "binning"
    n_bins: int = 10  # For binning calibration
    temperature: float = 1.5  # For temperature scaling
    min_samples_per_bin: int = 5
    smoothing: float = 1e-3  # For numerical stability
    confidence_thresholds: Dict[str, float] = None  # Per-intent thresholds
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = {
                'high': 0.8,
                'medium': 0.5,
                'low': 0.3
            }


@dataclass
class CalibrationSample:
    """Sample for calibration."""
    predicted_confidence: float
    true_label: str
    predicted_label: str
    is_correct: bool


class ConfidenceCalibrator:
    """
    Calibrates confidence scores to be more reliable.
    Maps raw confidence scores to calibrated probabilities.
    """
    
    def __init__(self, config: CalibrationConfig):
        """
        Initialize the confidence calibrator.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        self.calibrators = {}  # Per-intent calibrators
        self.global_calibrator = None
        self.temperature = config.temperature
        self.calibration_stats = {}
        self.is_calibrated = False
        
    def collect_calibration_data(self, predictions: List[Tuple[str, float]], 
                               true_labels: List[str]) -> List[CalibrationSample]:
        """
        Collect calibration samples from predictions.
        
        Args:
            predictions: List of (intent, confidence) tuples
            true_labels: List of true intent labels
            
        Returns:
            List of calibration samples
        """
        samples = []
        
        for (pred_intent, conf), true_intent in zip(predictions, true_labels):
            is_correct = pred_intent == true_intent
            
            sample = CalibrationSample(
                predicted_confidence=conf,
                true_label=true_intent,
                predicted_label=pred_intent,
                is_correct=is_correct
            )
            samples.append(sample)
        
        return samples
    
    def fit(self, calibration_samples: List[CalibrationSample]):
        """
        Fit calibration models on collected samples.
        
        Args:
            calibration_samples: List of calibration samples
        """
        if not calibration_samples:
            logger.warning("No calibration samples provided")
            return
        
        logger.info(f"Fitting calibration with {len(calibration_samples)} samples")
        
        # Prepare data
        confidences = np.array([s.predicted_confidence for s in calibration_samples])
        correctness = np.array([s.is_correct for s in calibration_samples])
        
        # Fit global calibrator
        if self.config.method == "isotonic":
            self._fit_isotonic(confidences, correctness)
        elif self.config.method == "platt":
            self._fit_platt(confidences, correctness)
        elif self.config.method == "temperature":
            self._fit_temperature(confidences, correctness)
        elif self.config.method == "binning":
            self._fit_binning(confidences, correctness)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.method}")
        
        # Fit per-intent calibrators if enough data
        self._fit_per_intent_calibrators(calibration_samples)
        
        # Calculate calibration statistics
        self._calculate_calibration_stats(calibration_samples)
        
        self.is_calibrated = True
        logger.info("Calibration fitting completed")
    
    def _fit_isotonic(self, confidences: np.ndarray, correctness: np.ndarray):
        """Fit isotonic regression calibrator."""
        # Sort by confidence
        sorted_idx = np.argsort(confidences)
        conf_sorted = confidences[sorted_idx]
        correct_sorted = correctness[sorted_idx]
        
        # Add boundary points for stability
        conf_sorted = np.concatenate([[0.0], conf_sorted, [1.0]])
        correct_sorted = np.concatenate([[0.0], correct_sorted, [1.0]])
        
        # Fit isotonic regression
        self.global_calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        self.global_calibrator.fit(conf_sorted, correct_sorted)
    
    def _fit_platt(self, confidences: np.ndarray, correctness: np.ndarray):
        """Fit Platt scaling (logistic regression)."""
        from sklearn.linear_model import LogisticRegression
        
        # Transform to logit space
        epsilon = self.config.smoothing
        logits = np.log((confidences + epsilon) / (1 - confidences + epsilon))
        
        # Fit logistic regression
        self.global_calibrator = LogisticRegression()
        self.global_calibrator.fit(logits.reshape(-1, 1), correctness)
    
    def _fit_temperature(self, confidences: np.ndarray, correctness: np.ndarray):
        """Fit temperature scaling."""
        # Find optimal temperature using grid search
        temperatures = np.linspace(0.5, 3.0, 26)
        best_temp = 1.0
        best_error = float('inf')
        
        for temp in temperatures:
            # Apply temperature scaling
            scaled_conf = self._apply_temperature(confidences, temp)
            
            # Calculate calibration error
            error = self._calculate_ece(scaled_conf, correctness)
            
            if error < best_error:
                best_error = error
                best_temp = temp
        
        self.temperature = best_temp
        logger.info(f"Optimal temperature: {self.temperature:.2f}")
    
    def _fit_binning(self, confidences: np.ndarray, correctness: np.ndarray):
        """Fit histogram binning calibration."""
        # Create bins
        bin_edges = np.linspace(0, 1, self.config.n_bins + 1)
        
        # Calculate accuracy per bin
        bin_accuracies = {}
        bin_counts = {}
        
        for i in range(self.config.n_bins):
            bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == self.config.n_bins - 1:  # Include right edge for last bin
                bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            
            bin_samples = correctness[bin_mask]
            
            if len(bin_samples) >= self.config.min_samples_per_bin:
                bin_accuracies[i] = np.mean(bin_samples)
                bin_counts[i] = len(bin_samples)
            else:
                # Use neighboring bins or default
                bin_accuracies[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
                bin_counts[i] = 0
        
        self.bin_edges = bin_edges
        self.bin_accuracies = bin_accuracies
        self.bin_counts = bin_counts
    
    def _fit_per_intent_calibrators(self, samples: List[CalibrationSample]):
        """Fit calibrators for each intent if enough data."""
        # Group samples by predicted intent
        intent_samples = {}
        for sample in samples:
            intent = sample.predicted_label
            if intent not in intent_samples:
                intent_samples[intent] = []
            intent_samples[intent].append(sample)
        
        # Fit calibrator for each intent with enough samples
        min_samples = 20
        for intent, intent_samples_list in intent_samples.items():
            if len(intent_samples_list) >= min_samples:
                confidences = np.array([s.predicted_confidence for s in intent_samples_list])
                correctness = np.array([s.is_correct for s in intent_samples_list])
                
                # Use isotonic regression for per-intent calibration
                calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                
                # Add boundary points
                conf_with_bounds = np.concatenate([[0.0], confidences, [1.0]])
                correct_with_bounds = np.concatenate([[0.0], correctness, [1.0]])
                
                calibrator.fit(conf_with_bounds, correct_with_bounds)
                self.calibrators[intent] = calibrator
                
                logger.info(f"Fitted calibrator for intent {intent} with {len(intent_samples_list)} samples")
    
    def calibrate(self, intent: str, raw_confidence: float) -> float:
        """
        Calibrate a confidence score.
        
        Args:
            intent: Predicted intent
            raw_confidence: Raw confidence score
            
        Returns:
            Calibrated confidence score
        """
        if not self.is_calibrated:
            return raw_confidence
        
        # Clip to valid range
        raw_confidence = np.clip(raw_confidence, 0.0, 1.0)
        
        # Try intent-specific calibrator first
        if intent in self.calibrators:
            calibrated = self.calibrators[intent].predict([raw_confidence])[0]
        elif self.global_calibrator is not None:
            # Use global calibrator
            if self.config.method == "isotonic":
                calibrated = self.global_calibrator.predict([raw_confidence])[0]
            elif self.config.method == "platt":
                epsilon = self.config.smoothing
                logit = np.log((raw_confidence + epsilon) / (1 - raw_confidence + epsilon))
                calibrated = self.global_calibrator.predict_proba([[logit]])[0, 1]
            elif self.config.method == "temperature":
                calibrated = self._apply_temperature(raw_confidence, self.temperature)
            elif self.config.method == "binning":
                calibrated = self._apply_binning(raw_confidence)
            else:
                calibrated = raw_confidence
        else:
            calibrated = raw_confidence
        
        # Ensure valid range
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def _apply_temperature(self, confidence: Union[float, np.ndarray], 
                         temperature: float) -> Union[float, np.ndarray]:
        """Apply temperature scaling."""
        # Convert to logit
        epsilon = self.config.smoothing
        if isinstance(confidence, np.ndarray):
            logit = np.log((confidence + epsilon) / (1 - confidence + epsilon))
        else:
            logit = np.log((confidence + epsilon) / (1 - confidence + epsilon))
        
        # Scale by temperature
        scaled_logit = logit / temperature
        
        # Convert back to probability
        return 1 / (1 + np.exp(-scaled_logit))
    
    def _apply_binning(self, confidence: float) -> float:
        """Apply histogram binning calibration."""
        # Find bin
        for i in range(self.config.n_bins):
            if i == self.config.n_bins - 1:  # Last bin
                if confidence >= self.bin_edges[i] and confidence <= self.bin_edges[i + 1]:
                    return self.bin_accuracies[i]
            else:
                if confidence >= self.bin_edges[i] and confidence < self.bin_edges[i + 1]:
                    return self.bin_accuracies[i]
        
        # Fallback
        return confidence
    
    def _calculate_ece(self, confidences: np.ndarray, correctness: np.ndarray) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            confidences: Predicted confidences
            correctness: Binary correctness array
            
        Returns:
            ECE score
        """
        # Create bins
        bin_edges = np.linspace(0, 1, self.config.n_bins + 1)
        ece = 0.0
        
        for i in range(self.config.n_bins):
            bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == self.config.n_bins - 1:
                bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            
            bin_samples = correctness[bin_mask]
            bin_confidences = confidences[bin_mask]
            
            if len(bin_samples) > 0:
                bin_accuracy = np.mean(bin_samples)
                bin_confidence = np.mean(bin_confidences)
                bin_weight = len(bin_samples) / len(confidences)
                
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _calculate_calibration_stats(self, samples: List[CalibrationSample]):
        """Calculate calibration statistics."""
        # Overall statistics
        confidences = np.array([s.predicted_confidence for s in samples])
        correctness = np.array([s.is_correct for s in samples])
        
        # Calculate ECE before calibration
        ece_before = self._calculate_ece(confidences, correctness)
        
        # Calculate ECE after calibration
        calibrated_confs = np.array([
            self.calibrate(s.predicted_label, s.predicted_confidence) 
            for s in samples
        ])
        ece_after = self._calculate_ece(calibrated_confs, correctness)
        
        # Reliability diagram data
        fraction_positive, mean_predicted = calibration_curve(
            correctness, confidences, n_bins=self.config.n_bins
        )
        
        self.calibration_stats = {
            'ece_before': float(ece_before),
            'ece_after': float(ece_after),
            'improvement': float(ece_before - ece_after),
            'reliability_diagram': {
                'fraction_positive': fraction_positive.tolist(),
                'mean_predicted': mean_predicted.tolist()
            },
            'n_samples': len(samples),
            'accuracy': float(np.mean(correctness)),
            'avg_confidence_before': float(np.mean(confidences)),
            'avg_confidence_after': float(np.mean(calibrated_confs))
        }
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level category.
        
        Args:
            confidence: Calibrated confidence score
            
        Returns:
            Confidence level: 'high', 'medium', 'low', or 'very_low'
        """
        if confidence >= self.config.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.config.confidence_thresholds['medium']:
            return 'medium'
        elif confidence >= self.config.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def should_use_fallback(self, confidence: float, intent: str) -> bool:
        """
        Determine if fallback (e.g., KNN) should be used.
        
        Args:
            confidence: Calibrated confidence score
            intent: Predicted intent
            
        Returns:
            True if fallback should be used
        """
        # Check intent-specific threshold if available
        intent_threshold = self.config.confidence_thresholds.get(
            intent, 
            self.config.confidence_thresholds['medium']
        )
        
        return confidence < intent_threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get calibration statistics.
        
        Returns:
            Dictionary with calibration statistics
        """
        stats = {
            'is_calibrated': self.is_calibrated,
            'method': self.config.method,
            'n_bins': self.config.n_bins,
            'temperature': self.temperature if self.config.method == 'temperature' else None,
            'per_intent_calibrators': list(self.calibrators.keys()),
            'calibration_stats': self.calibration_stats
        }
        
        return stats
    
    def save(self, path: Union[str, Path]):
        """
        Save the calibrator.
        
        Args:
            path: Path to save the calibrator
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'config': self.config,
            'calibrators': self.calibrators,
            'global_calibrator': self.global_calibrator,
            'temperature': self.temperature,
            'calibration_stats': self.calibration_stats,
            'is_calibrated': self.is_calibrated
        }
        
        # Handle binning-specific data
        if self.config.method == 'binning':
            save_data['bin_edges'] = self.bin_edges
            save_data['bin_accuracies'] = self.bin_accuracies
            save_data['bin_counts'] = self.bin_counts
        
        joblib.dump(save_data, path)
        logger.info(f"Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ConfidenceCalibrator':
        """
        Load a saved calibrator.
        
        Args:
            path: Path to the saved calibrator
            
        Returns:
            Loaded ConfidenceCalibrator instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {path}")
        
        # Load data
        data = joblib.load(path)
        
        # Create instance
        calibrator = cls(data['config'])
        
        # Restore state
        calibrator.calibrators = data['calibrators']
        calibrator.global_calibrator = data['global_calibrator']
        calibrator.temperature = data['temperature']
        calibrator.calibration_stats = data['calibration_stats']
        calibrator.is_calibrated = data['is_calibrated']
        
        # Restore binning-specific data
        if data['config'].method == 'binning':
            calibrator.bin_edges = data['bin_edges']
            calibrator.bin_accuracies = data['bin_accuracies']
            calibrator.bin_counts = data['bin_counts']
        
        logger.info(f"Calibrator loaded from {path}")
        
        return calibrator

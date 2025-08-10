"""
Tests for the confidence calibration module.
"""

import pytest
import numpy as np
from intent_classifier.core.confidence import (
    ConfidenceCalibrator, CalibrationConfig, CalibrationSample
)


class TestConfidenceCalibrator:
    """Test the ConfidenceCalibrator class."""
    
    @pytest.fixture
    def config(self):
        """Create calibration configuration."""
        return CalibrationConfig(
            method="isotonic",
            n_bins=10,
            temperature=1.5,
            confidence_thresholds={
                'high': 0.8,
                'medium': 0.5,
                'low': 0.3
            }
        )
    
    @pytest.fixture
    def calibration_samples(self):
        """Create sample calibration data."""
        samples = []
        
        # High confidence correct predictions
        for i in range(20):
            samples.append(CalibrationSample(
                predicted_confidence=0.85 + np.random.uniform(-0.05, 0.05),
                true_label="INTENT_A",
                predicted_label="INTENT_A",
                is_correct=True
            ))
        
        # High confidence incorrect predictions (overconfident)
        for i in range(10):
            samples.append(CalibrationSample(
                predicted_confidence=0.9 + np.random.uniform(-0.05, 0.05),
                true_label="INTENT_A",
                predicted_label="INTENT_B",
                is_correct=False
            ))
        
        # Medium confidence mixed
        for i in range(15):
            is_correct = i % 2 == 0
            samples.append(CalibrationSample(
                predicted_confidence=0.6 + np.random.uniform(-0.1, 0.1),
                true_label="INTENT_B",
                predicted_label="INTENT_B" if is_correct else "INTENT_C",
                is_correct=is_correct
            ))
        
        # Low confidence mostly incorrect
        for i in range(10):
            is_correct = i < 3
            samples.append(CalibrationSample(
                predicted_confidence=0.3 + np.random.uniform(-0.1, 0.1),
                true_label="INTENT_C",
                predicted_label="INTENT_C" if is_correct else "INTENT_A",
                is_correct=is_correct
            ))
        
        return samples
    
    def test_initialization(self, config):
        """Test calibrator initialization."""
        calibrator = ConfidenceCalibrator(config)
        
        assert calibrator.config == config
        assert not calibrator.is_calibrated
        assert len(calibrator.calibrators) == 0
        assert calibrator.temperature == config.temperature
    
    def test_collect_calibration_data(self, config):
        """Test collecting calibration samples."""
        calibrator = ConfidenceCalibrator(config)
        
        predictions = [
            ("INTENT_A", 0.9),
            ("INTENT_B", 0.7),
            ("INTENT_A", 0.6)
        ]
        true_labels = ["INTENT_A", "INTENT_B", "INTENT_C"]
        
        samples = calibrator.collect_calibration_data(predictions, true_labels)
        
        assert len(samples) == 3
        assert samples[0].is_correct == True
        assert samples[1].is_correct == True
        assert samples[2].is_correct == False
    
    def test_isotonic_calibration(self, config, calibration_samples):
        """Test isotonic regression calibration."""
        config.method = "isotonic"
        calibrator = ConfidenceCalibrator(config)
        
        calibrator.fit(calibration_samples)
        
        assert calibrator.is_calibrated
        assert calibrator.global_calibrator is not None
        
        # Test calibration
        # High confidence should be reduced (was overconfident)
        calibrated_high = calibrator.calibrate("INTENT_A", 0.95)
        assert calibrated_high < 0.95
        
        # Low confidence might increase slightly
        calibrated_low = calibrator.calibrate("INTENT_C", 0.25)
        assert 0.0 <= calibrated_low <= 1.0
    
    def test_temperature_calibration(self, config, calibration_samples):
        """Test temperature scaling calibration."""
        config.method = "temperature"
        calibrator = ConfidenceCalibrator(config)
        
        calibrator.fit(calibration_samples)
        
        assert calibrator.is_calibrated
        assert calibrator.temperature > 0
        
        # Test that temperature scaling works
        raw_conf = 0.8
        calibrated = calibrator.calibrate("INTENT_A", raw_conf)
        
        # Temperature > 1 should flatten confidence
        if calibrator.temperature > 1:
            assert abs(calibrated - 0.5) < abs(raw_conf - 0.5)
    
    def test_binning_calibration(self, config, calibration_samples):
        """Test histogram binning calibration."""
        config.method = "binning"
        config.n_bins = 5
        calibrator = ConfidenceCalibrator(config)
        
        calibrator.fit(calibration_samples)
        
        assert calibrator.is_calibrated
        assert hasattr(calibrator, 'bin_edges')
        assert hasattr(calibrator, 'bin_accuracies')
        
        # Test calibration
        calibrated = calibrator.calibrate("INTENT_A", 0.85)
        assert 0.0 <= calibrated <= 1.0
    
    def test_per_intent_calibration(self, config):
        """Test per-intent calibration."""
        calibrator = ConfidenceCalibrator(config)
        
        # Create samples with enough data for one intent
        samples = []
        for i in range(25):
            samples.append(CalibrationSample(
                predicted_confidence=0.8,
                true_label="INTENT_A",
                predicted_label="INTENT_A",
                is_correct=i < 20  # 80% accuracy
            ))
        
        for i in range(5):
            samples.append(CalibrationSample(
                predicted_confidence=0.7,
                true_label="INTENT_B",
                predicted_label="INTENT_B",
                is_correct=True
            ))
        
        calibrator.fit(samples)
        
        # Should have per-intent calibrator for INTENT_A
        assert "INTENT_A" in calibrator.calibrators
        assert "INTENT_B" not in calibrator.calibrators  # Not enough samples
    
    def test_confidence_levels(self, config):
        """Test confidence level categorization."""
        calibrator = ConfidenceCalibrator(config)
        
        assert calibrator.get_confidence_level(0.9) == "high"
        assert calibrator.get_confidence_level(0.6) == "medium"
        assert calibrator.get_confidence_level(0.4) == "low"
        assert calibrator.get_confidence_level(0.2) == "very_low"
    
    def test_fallback_decision(self, config):
        """Test fallback decision logic."""
        calibrator = ConfidenceCalibrator(config)
        
        # Below medium threshold
        assert calibrator.should_use_fallback(0.4, "INTENT_A") == True
        
        # Above medium threshold
        assert calibrator.should_use_fallback(0.7, "INTENT_A") == False
        
        # Custom intent threshold
        config.confidence_thresholds["INTENT_CRITICAL"] = 0.9
        assert calibrator.should_use_fallback(0.85, "INTENT_CRITICAL") == True
    
    def test_calibration_statistics(self, config, calibration_samples):
        """Test calibration statistics calculation."""
        calibrator = ConfidenceCalibrator(config)
        calibrator.fit(calibration_samples)
        
        stats = calibrator.get_statistics()
        
        assert stats['is_calibrated'] == True
        assert stats['method'] == config.method
        assert 'calibration_stats' in stats
        assert 'ece_before' in stats['calibration_stats']
        assert 'ece_after' in stats['calibration_stats']
        assert 'improvement' in stats['calibration_stats']
        
        # ECE should improve after calibration
        assert stats['calibration_stats']['improvement'] >= 0
    
    def test_save_load(self, config, calibration_samples, tmp_path):
        """Test saving and loading calibrator."""
        calibrator = ConfidenceCalibrator(config)
        calibrator.fit(calibration_samples)
        
        # Save
        save_path = tmp_path / "calibrator.pkl"
        calibrator.save(save_path)
        
        assert save_path.exists()
        
        # Load
        loaded_calibrator = ConfidenceCalibrator.load(save_path)
        
        assert loaded_calibrator.is_calibrated
        assert loaded_calibrator.config.method == config.method
        
        # Test that calibration works the same
        test_conf = 0.85
        original_calib = calibrator.calibrate("INTENT_A", test_conf)
        loaded_calib = loaded_calibrator.calibrate("INTENT_A", test_conf)
        
        assert abs(original_calib - loaded_calib) < 0.001
    
    def test_edge_cases(self, config):
        """Test edge cases."""
        calibrator = ConfidenceCalibrator(config)
        
        # No samples
        calibrator.fit([])
        assert not calibrator.is_calibrated
        
        # Uncalibrated calibration (should return original)
        assert calibrator.calibrate("INTENT", 0.7) == 0.7
        
        # Extreme confidences
        calibrator.fit([
            CalibrationSample(0.0, "A", "A", True),
            CalibrationSample(1.0, "A", "A", True),
            CalibrationSample(0.5, "A", "A", True)
        ])
        
        assert 0 <= calibrator.calibrate("A", 0.0) <= 1.0
        assert 0 <= calibrator.calibrate("A", 1.0) <= 1.0
        assert 0 <= calibrator.calibrate("A", 1.5) <= 1.0  # Out of range
    
    def test_platt_calibration(self, config, calibration_samples):
        """Test Platt scaling calibration."""
        config.method = "platt"
        calibrator = ConfidenceCalibrator(config)
        
        calibrator.fit(calibration_samples)
        
        assert calibrator.is_calibrated
        assert calibrator.global_calibrator is not None
        
        # Test calibration
        calibrated = calibrator.calibrate("INTENT_A", 0.8)
        assert 0.0 <= calibrated <= 1.0
    
    def test_reliability_diagram_data(self, config, calibration_samples):
        """Test reliability diagram data generation."""
        calibrator = ConfidenceCalibrator(config)
        calibrator.fit(calibration_samples)
        
        stats = calibrator.calibration_stats
        assert 'reliability_diagram' in stats
        assert 'fraction_positive' in stats['reliability_diagram']
        assert 'mean_predicted' in stats['reliability_diagram']
        
        # Should have data points for bins
        assert len(stats['reliability_diagram']['fraction_positive']) > 0
        assert len(stats['reliability_diagram']['mean_predicted']) > 0

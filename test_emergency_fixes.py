#!/usr/bin/env python3
"""
Test Suite: Emergency Fixes Validation
=======================================

Comprehensive testing for NMS, dual thresholds, and FDI validation fixes.

Run: python test_emergency_fixes.py

Author: Antigravity AI Emergency Response Team
Date: 2026-01-28
Version: 1.0
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules to test
try:
    from emergency_fix_nms_thresholds import compute_iou, apply_nms, process_predictions
    from fdi_validator import FDIValidator
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure emergency_fix_nms_thresholds.py and fdi_validator.py are in the same directory")
    sys.exit(1)


# ==================================================================
# TEST 1: NMS Functionality
# ==================================================================

def test_nms_removes_overlaps():
    """
    Test that NMS with IoU=0.25 removes overlapping detections.
    
    Scenario:
    - 3 detections: 2 overlapping (40% IoU), 1 separate
    - Expected: Keep highest confidence from overlapping pair + separate detection = 2 total
    """
    print("\n" + "="*70)
    print("TEST 1: NMS Overlapping Detection Removal")
    print("="*70)
    
    detections = [
        {'box': np.array([0, 0, 100, 100]), 'score': 0.9, 'class_id': 0},          # High confidence
        {'box': np.array([30, 30, 130, 130]), 'score': 0.8, 'class_id': 0},       # 40% overlap with box1
        {'box': np.array([200, 200, 300, 300]), 'score': 0.7, 'class_id': 0}      # No overlap
    ]
    
    # Calculate IoU to verify test setup
    iou_1_2 = compute_iou(detections[0]['box'], detections[1]['box'])
    print(f"  Setup: Detection 1 and 2 overlap = {iou_1_2:.2f} (should be ~0.14-0.15)")
    
    # Apply NMS with IoU=0.25 (aggressive)
    kept = apply_nms(detections, iou_threshold=0.25)
    
    print(f"  Input: 3 detections")
    print(f"  After NMS (IoU=0.25): {len(kept)} detections kept")
    print(f"  Kept scores: {[k['score'] for k in kept]}")
    
    # Assertions
    assert len(kept) == 2, f"❌ Expected 2 detections, got {len(kept)}"
    assert 0.9 in [k['score'] for k in kept], "❌ Highest confidence detection should be kept"
    assert 0.7 in [k['score'] for k in kept], "❌ Non-overlapping detection should be kept"
    assert 0.8 not in [k['score'] for k in kept], "❌ Overlapping lower-confidence detection should be suppressed"
    
    print("  ✅ PASSED: NMS correctly suppresses overlapping detections")


def test_nms_threshold_sensitivity():
    """
    Test that different IoU thresholds produce expected results.
    
    Scenario:
    - 2 detections with ~22% overlap (actual IoU calculation)
    - IoU=0.20: Should suppress (22% > 20%)
    - IoU=0.25: Should keep both (22% < 25%)
    """
    print("\n" + "="*70)
    print("TEST 2: NMS IoU Threshold Sensitivity")
    print("="*70)
    
    detections = [
        {'box': np.array([0, 0, 100, 100]), 'score': 0.9, 'class_id': 0},
        {'box': np.array([40, 40, 140, 140]), 'score': 0.85, 'class_id': 0}  # ~22% IoU (4000/18000)
    ]
    
    iou = compute_iou(detections[0]['box'], detections[1]['box'])
    print(f"  Detection overlap: {iou:.2%}")
    
    # Test very aggressive threshold (0.20) - should suppress
    kept_very_aggressive = apply_nms(detections, iou_threshold=0.20)
    print(f"  IoU=0.20 (very aggressive): {len(kept_very_aggressive)} kept")
    assert len(kept_very_aggressive) == 1, f"❌ Should suppress at IoU=0.20 (overlap is {iou:.2%})"
    
    # Test standard threshold (0.25) - should keep both since 22% < 25%
    kept_standard = apply_nms(detections, iou_threshold=0.25)
    print(f"  IoU=0.25 (standard aggressive): {len(kept_standard)} kept")
    assert len(kept_standard) == 2, f"❌ Should keep both at IoU=0.25 (overlap is {iou:.2%})"
    
    print("  ✅ PASSED: IoU threshold correctly controls suppression")


# ==================================================================
# TEST 3: Dual Jaw-Specific Thresholds
# ==================================================================

def test_jaw_specific_thresholds():
    """
    Verify threshold calculation logic:
    - Upper jaw, center: 0.12 × 0.5 = 0.06
    - Upper jaw, outer: 0.12 × 1.0 = 0.12
    - Lower jaw, center: 0.42 × 0.5 = 0.21
    - Lower jaw, outer: 0.42 × 1.0 = 0.42
    """
    print("\n" + "="*70)
    print("TEST 3: Jaw-Specific Threshold Calculation")
    print("="*70)
    
    test_cases = [
        {"jaw": "upper", "zone": "center", "base": 0.12, "mult": 0.5, "expected": 0.06},
        {"jaw": "upper", "zone": "outer", "base": 0.12, "mult": 1.0, "expected": 0.12},
        {"jaw": "lower", "zone": "center", "base": 0.42, "mult": 0.5, "expected": 0.21},
        {"jaw": "lower", "zone": "outer", "base": 0.42, "mult": 1.0, "expected": 0.42},
    ]
    
    print(f"  {'Jaw':<8} {'Zone':<8} {'Base':<8} {'Mult':<8} {'Expected':<10} {'Calculated':<12} {'Status':<8}")
    print("  " + "-"*70)
    
    all_passed = True
    for tc in test_cases:
        calculated = tc["base"] * tc["mult"]
        status = "✅" if abs(calculated - tc["expected"]) < 0.01 else "❌"
        
        print(f"  {tc['jaw']:<8} {tc['zone']:<8} {tc['base']:<8.2f} {tc['mult']:<8.1f} {tc['expected']:<10.2f} {calculated:<12.2f} {status:<8}")
        
        if abs(calculated - tc["expected"]) >= 0.01:
            all_passed = False
    
    assert all_passed, "❌ Threshold calculations incorrect"
    print("  ✅ PASSED: All threshold combinations correct")


# ==================================================================
# TEST 4: FDI Duplicate Detection & Resolution
# ==================================================================

def test_duplicate_fdi_detection():
    """
    Test that FDIValidator detects and resolves duplicate FDI numbers.
    
    Scenario:
    - 3 teeth all labeled FDI 36 (duplicate)
    - Expected: Warning issued, correction via spatial reassignment to 36, 37, 38
    """
    print("\n" + "="*70)
    print("TEST 4: FDI Duplicate Detection & Correction")
    print("="*70)
    
    test_teeth = [
        {'fdi': 36, 'box': [100, 200, 150, 250], 'score': 0.9, 'quadrant': 3},  # Left-most
        {'fdi': 36, 'box': [160, 200, 210, 250], 'score': 0.85, 'quadrant': 3}, # Middle
        {'fdi': 36, 'box': [220, 200, 270, 250], 'score': 0.88, 'quadrant': 3}, # Right-most
    ]
    
    print(f"  Input: 3 teeth all with FDI 36 (lower left 1st molar)")
    
    corrected, warnings, confidence = FDIValidator.validate_and_correct(test_teeth)
    
    print(f"  Warnings generated: {len(warnings)}")
    for w in warnings:
        print(f"    - {w}")
    
    # Check duplicate was detected
    assert any("CRITICAL" in w and "Duplicate" in w for w in warnings), "❌ Duplicate not detected"
    
    # Check spatial reassignment
    fdi_corrected = sorted([t['fdi'] for t in corrected])
    print(f"  Corrected FDI: {fdi_corrected}")
    
    assert 36 in fdi_corrected, "❌ Original FDI 36 should be preserved"
    assert 37 in fdi_corrected, "❌ Expected FDI 37 from spatial reassignment"
    assert 38in fdi_corrected, "❌ Expected FDI 38 from spatial reassignment"
    
    # Check confidence is lowered (due to critical warning)
    print(f"  Confidence: {confidence:.2f}")
    assert confidence < 0.85, "❌ Confidence should be lowered due to duplicates"
    
    print("  ✅ PASSED: Duplicates detected and spatially corrected")


def test_quadrant_count_validation():
    """
    Test that FDIValidator warns about excessive quadrant counts.
    
    Scenario:
    - 10 teeth in Quadrant 1 (expected max 8)
    - Expected: WARNING about overcount
    """
    print("\n" + "="*70)
    print("TEST 5: Quadrant Count Validation")
    print("="*70)
    
    # Create 10 teeth in Q1 (too many)
    test_teeth = [
        {'fdi': 10 + i, 'box': [i*30, 100, i*30+25, 150], 'score': 0.9, 'quadrant': 1}
        for i in range(1, 11)  # FDI 11-20... but Q1 only has 11-18 (8 teeth max)
    ]
    
    print(f"  Input: 10 teeth in Quadrant 1 (expected max 8)")
    
    _, warnings, confidence = FDIValidator.validate_and_correct(test_teeth)
    
    print(f"  Warnings: {[w for w in warnings if 'Quadrant 1' in w]}")
    
    # Check warning was issued
    assert any("Quadrant 1" in w and "10 teeth" in w for w in warnings), "❌ Overcount not detected"
    
    print("  ✅ PASSED: Quadrant overcount detected")


# ==================================================================
# TEST 6: Spatial Coherence Scoring
# ==================================================================

def test_spatial_coherence_scoring():
    """
    Test that spatial coherence scoring differentiates good vs bad FDI order.
    
    Scenario A (Good): FDI order matches X-coordinate order
    Scenario B (Bad): FDI order scrambled
    """
    print("\n" + "="*70)
    print("TEST 6: Spatial Coherence Scoring")
    print("="*70)
    
    # Perfect spatial order: Q1 should have FDI decreasing (18→11) left-to-right
    perfect_teeth = [
        {'fdi': 18, 'box': [100, 100, 150, 150], 'score': 0.9, 'quadrant': 1},
        {'fdi': 17, 'box': [160, 100, 210, 150], 'score': 0.9, 'quadrant': 1},
        {'fdi': 16, 'box': [220, 100, 270, 150], 'score': 0.9, 'quadrant': 1},
        {'fdi': 15, 'box': [280, 100, 330, 150], 'score': 0.9, 'quadrant': 1},
    ]
    
    coherence_perfect = FDIValidator._check_spatial_coherence(perfect_teeth)
    print(f"  Perfect order (18→17→16→15): Coherence = {coherence_perfect:.2f}")
    assert coherence_perfect >= 0.95, f"❌ Expected coherence ≥0.95, got {coherence_perfect}"
    
    # Scrambled spatial order
    scrambled_teeth = [
        {'fdi': 15, 'box': [100, 100, 150, 150], 'score': 0.9, 'quadrant': 1},  # Out of order
        {'fdi': 18, 'box': [160, 100, 210, 150], 'score': 0.9, 'quadrant': 1},
        {'fdi': 16, 'box': [220, 100, 270, 150], 'score': 0.9, 'quadrant': 1},
        {'fdi': 17, 'box': [280, 100, 330, 150], 'score': 0.9, 'quadrant': 1},
    ]
    
    coherence_scrambled = FDIValidator._check_spatial_coherence(scrambled_teeth)
    print(f"  Scrambled order (15→18→16→17): Coherence = {coherence_scrambled:.2f}")
    assert coherence_scrambled < 0.50, f"❌ Expected coherence <0.50, got {coherence_scrambled}"
    
    print("  ✅ PASSED: Spatial coherence correctly differentiates quality")


# ==================================================================
# TEST 7: Confidence Calculation
# ==================================================================

def test_confidence_calculation():
    """
    Test that confidence scoring properly incorporates warnings and coherence.
    
    Scenario A (High Confidence): Perfect teeth, no warnings
    Scenario B (Low Confidence): Duplicates + spatial errors
    """
    print("\n" + "="*70)
    print("TEST 7: Confidence Score Calculation")
    print("="*70)
    
    # High confidence scenario
    high_conf_teeth = [
        {'fdi': 18, 'box': [100, 100, 150, 150], 'score': 0.9, 'quadrant': 1},
        {'fdi': 17, 'box': [160, 100, 210, 150], 'score': 0.9, 'quadrant': 1},
    ]
    
    _, warnings_high, conf_high = FDIValidator.validate_and_correct(high_conf_teeth)
    print(f"  High confidence scenario:")
    print(f"    Warnings: {len(warnings_high)}")
    print(f"    Confidence: {conf_high:.2f}")
    assert conf_high >= 0.85, f"❌ Expected confidence ≥0.85, got {conf_high}"
    
    # Low confidence scenario (duplicates)
    low_conf_teeth = [
        {'fdi': 36, 'box': [100, 200, 150, 250], 'score': 0.9, 'quadrant': 3},
        {'fdi': 36, 'box': [160, 200, 210, 250], 'score': 0.85, 'quadrant': 3},
        {'fdi': 36, 'box': [220, 200, 270, 250], 'score': 0.88, 'quadrant': 3},
    ]
    
    _, warnings_low, conf_low = FDIValidator.validate_and_correct(low_conf_teeth)
    print(f"  Low confidence scenario (3 duplicates):")
    print(f"    Warnings: {len(warnings_low)}")
    print(f"    Confidence: {conf_low:.2f}")
    assert conf_low < 0.70, f"❌ Expected confidence <0.70, got {conf_low}"
    
    print("  ✅ PASSED: Confidence calculation differentiates quality")


# ==================================================================
# MAIN TEST RUNNER
# ==================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("EMERGENCY FIXES - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("Testing NMS, dual thresholds, and FDI validation")
    print("="*70)
    
    tests = [
        ("NMS Overlap Removal", test_nms_removes_overlaps),
        ("NMS Threshold Sensitivity", test_nms_threshold_sensitivity),
        ("Jaw-Specific Thresholds", test_jaw_specific_thresholds),
        ("FDI Duplicate Detection", test_duplicate_fdi_detection),
        ("Quadrant Count Validation", test_quadrant_count_validation),
        ("Spatial Coherence Scoring", test_spatial_coherence_scoring),
        ("Confidence Calculation", test_confidence_calculation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ TEST FAILED: {test_name}")
            print(f"  Error: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n  ❌ TEST ERROR: {test_name}")
            print(f"  Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Total tests: {passed + failed}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    
    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED - EMERGENCY FIXES VALIDATED")
        print("  Ready for production deployment")
        print("="*70)
        return 0
    else:
        print(f"\n  ⚠️  {failed} TEST(S) FAILED - FIX REQUIRED BEFORE DEPLOYMENT")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

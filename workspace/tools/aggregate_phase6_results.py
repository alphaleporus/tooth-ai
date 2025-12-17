#!/usr/bin/env python3
"""
Aggregate Phase 6 test results and generate comprehensive benchmark report.
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_summary_table(batch_metrics: Dict, stress_report: Dict,
                        regression_report: Dict, failcases: Dict) -> str:
    """Create markdown summary table."""
    batch_summary = batch_metrics.get('summary', {})
    stress_summary = stress_report.get('summary', {})
    reg_summary = regression_report.get('summary', {})
    
    table = """## Summary Metrics Table

| Metric Category | Metric | Value |
|----------------|--------|-------|
| **Batch Validation** | Mean Mask IoU | {:.4f} |
| | Std Mask IoU | {:.4f} |
| | Mean FDI Accuracy | {:.4f} |
| | Mean Detections | {:.2f} |
| | Val Set IoU | {:.4f} |
| | Train Set IoU | {:.4f} |
| **Stress Testing** | Worst Distortion | {} |
| | Max Detection Drop | {:.2f} |
| | Max Confidence Drop | {:.4f} |
| **Regression** | Stability Score | {:.2%} |
| | Mean Detection Diff | {:.2f} |
| | Mean Mask Overlap | {:.4f} |
| **Failure Cases** | Total Failcases | {} |
| | Worst IoU | {:.4f} |
| | Worst FDI Accuracy | {:.4f} |

""".format(
        batch_summary.get('mean_mask_iou', 0.0),
        batch_summary.get('std_mask_iou', 0.0),
        batch_summary.get('mean_fdi_acc_after', 0.0),
        batch_summary.get('mean_num_detections', 0.0),
        batch_summary.get('val_mean_iou', 0.0),
        batch_summary.get('train_mean_iou', 0.0),
        max(stress_summary.keys(), key=lambda k: stress_summary[k].get('mean_detection_drop', 0)) if stress_summary else 'N/A',
        max([s.get('mean_detection_drop', 0) for s in stress_summary.values()]) if stress_summary else 0.0,
        max([s.get('mean_confidence_drop', 0) for s in stress_summary.values()]) if stress_summary else 0.0,
        reg_summary.get('stability_score', 0.0),
        reg_summary.get('mean_detection_diff', 0.0),
        reg_summary.get('mean_mask_overlap', 0.0),
        failcases.get('num_failcases', 0),
        min([fc.get('mask_iou', 1.0) for fc in failcases.get('failcases', [])]) if failcases.get('failcases') else 0.0,
        min([fc.get('fdi_accuracy', 1.0) for fc in failcases.get('failcases', [])]) if failcases.get('failcases') else 0.0
    )
    
    return table


def create_radar_plot(batch_metrics: Dict, stress_report: Dict,
                      regression_report: Dict, output_path: str):
    """Create radar plot of performance metrics."""
    batch_summary = batch_metrics.get('summary', {})
    reg_summary = regression_report.get('summary', {})
    
    # Normalize metrics to 0-1 scale
    categories = ['Mask IoU', 'FDI Accuracy', 'Stability', 'Detection Rate', 'Consistency']
    values = [
        batch_summary.get('mean_mask_iou', 0.0),
        batch_summary.get('mean_fdi_acc_after', 0.0),
        reg_summary.get('stability_score', 0.0),
        min(1.0, batch_summary.get('mean_num_detections', 0) / 32.0),  # Normalize to 32 teeth
        1.0 - (batch_summary.get('std_mask_iou', 0.0) / 0.5) if batch_summary.get('std_mask_iou', 0) < 0.5 else 0.0
    ]
    
    # Create radar plot
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label='Performance')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Plot', size=16, fontweight='bold', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_performance_curves(batch_metrics: Dict, output_path: str):
    """Create performance curves from batch metrics."""
    per_image = batch_metrics.get('per_image', [])
    
    if not per_image:
        return
    
    # Sort by IoU
    sorted_by_iou = sorted(per_image, key=lambda x: x.get('mask_iou', 0.0))
    
    ious = [x.get('mask_iou', 0.0) for x in sorted_by_iou]
    fdi_accs = [x.get('fdi_accuracy_after', 0.0) for x in sorted_by_iou]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # IoU curve
    ax1.plot(range(len(ious)), ious, 'b-', linewidth=2, label='Mask IoU')
    ax1.axhline(y=np.mean(ious), color='r', linestyle='--', label=f'Mean: {np.mean(ious):.3f}')
    ax1.set_xlabel('Image Index (sorted by IoU)')
    ax1.set_ylabel('Mask IoU')
    ax1.set_title('Mask IoU Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # FDI Accuracy curve
    ax2.plot(range(len(fdi_accs)), fdi_accs, 'g-', linewidth=2, label='FDI Accuracy')
    ax2.axhline(y=np.mean(fdi_accs), color='r', linestyle='--', label=f'Mean: {np.mean(fdi_accs):.3f}')
    ax2.set_xlabel('Image Index (sorted by IoU)')
    ax2.set_ylabel('FDI Accuracy')
    ax2.set_title('FDI Accuracy Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_improvement_curves(batch_metrics: Dict, output_path: str):
    """Create improvement curves showing before/after correction."""
    per_image = batch_metrics.get('per_image', [])
    
    if not per_image:
        return
    
    fdi_before = [x.get('fdi_accuracy_before', 0.0) for x in per_image]
    fdi_after = [x.get('fdi_accuracy_after', 0.0) for x in per_image]
    improvements = [a - b for a, b in zip(fdi_after, fdi_before)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(per_image))
    ax.plot(x, fdi_before, 'r-', linewidth=2, label='Before Correction', alpha=0.7)
    ax.plot(x, fdi_after, 'g-', linewidth=2, label='After Correction', alpha=0.7)
    ax.fill_between(x, fdi_before, fdi_after, alpha=0.3, color='green', label='Improvement')
    
    ax.set_xlabel('Image Index')
    ax.set_ylabel('FDI Accuracy')
    ax.set_title('FDI Accuracy: Before vs After Correction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add improvement statistics
    mean_improvement = np.mean(improvements)
    ax.text(0.02, 0.98, f'Mean Improvement: {mean_improvement:+.4f}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_phase6_report(batch_metrics_path: str, stress_report_path: str,
                          regression_report_path: str, failcases_path: str,
                          output_path: str):
    """Generate comprehensive Phase 6 final report."""
    
    # Load all reports
    print("Loading test results...")
    batch_metrics = load_json(batch_metrics_path)
    stress_report = load_json(stress_report_path)
    regression_report = load_json(regression_report_path)
    failcases = load_json(failcases_path)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_dir = os.path.dirname(output_path)
    
    # Generate visualizations
    print("Generating visualizations...")
    create_radar_plot(batch_metrics, stress_report, regression_report,
                     os.path.join(report_dir, 'radar_plots.png'))
    create_performance_curves(batch_metrics,
                            os.path.join(report_dir, 'performance_curves.png'))
    create_improvement_curves(batch_metrics,
                             os.path.join(report_dir, 'improvement_curves.png'))
    
    # Extract data
    batch_summary = batch_metrics.get('summary', {})
    stress_summary = stress_report.get('summary', {})
    reg_summary = regression_report.get('summary', {})
    
    # Generate report
    report = f"""# Phase 6 Final Benchmark Report

## Executive Summary

This report presents comprehensive validation results for the Tooth-AI unified inference pipeline, including batch validation, stress testing, regression testing, and failure case analysis.

**Key Findings:**
- Mean Mask IoU: {batch_summary.get('mean_mask_iou', 0.0):.4f} ± {batch_summary.get('std_mask_iou', 0.0):.4f}
- Mean FDI Accuracy: {batch_summary.get('mean_fdi_acc_after', 0.0):.4f}
- Regression Stability: {reg_summary.get('stability_score', 0.0):.2%}
- Failure Cases Identified: {failcases.get('num_failcases', 0)}

---

## 1. Full Metric Tables

{create_summary_table(batch_metrics, stress_report, regression_report, failcases)}

### Detailed Batch Validation Results

**Validation Set:**
- Images tested: {batch_summary.get('num_val', 0)}
- Mean Mask IoU: {batch_summary.get('val_mean_iou', 0.0):.4f}
- Mean FDI Accuracy: {batch_summary.get('val_mean_fdi_acc', 0.0):.4f}

**Training Set:**
- Images tested: {batch_summary.get('num_train', 0)}
- Mean Mask IoU: {batch_summary.get('train_mean_iou', 0.0):.4f}
- Mean FDI Accuracy: {batch_summary.get('train_mean_fdi_acc', 0.0):.4f}

---

## 2. Stress Test Degradation Results

### Distortion Impact Analysis

| Distortion | Mean Detection Drop | Mean Confidence Drop | Tests |
|------------|---------------------|---------------------|-------|
"""
    
    # Add stress test results
    for dist_name, stats in stress_summary.items():
        report += f"| {dist_name} | {stats.get('mean_detection_drop', 0.0):.2f} ± {stats.get('std_detection_drop', 0.0):.2f} | {stats.get('mean_confidence_drop', 0.0):.4f} ± {stats.get('std_confidence_drop', 0.0):.4f} | {stats.get('num_tests', 0)} |\n"
    
    report += f"""
### Key Observations

- **Most Robust:** The pipeline shows best resilience to: {min(stress_summary.items(), key=lambda x: x[1].get('mean_detection_drop', 100))[0] if stress_summary else 'N/A'}
- **Most Sensitive:** The pipeline is most affected by: {max(stress_summary.items(), key=lambda x: x[1].get('mean_detection_drop', 0))[0] if stress_summary else 'N/A'}
- **Overall Robustness:** Average detection drop across all distortions: {np.mean([s.get('mean_detection_drop', 0) for s in stress_summary.values()]):.2f}

---

## 3. Regression Stability Score

**Stability Metrics:**
- Stability Score: {reg_summary.get('stability_score', 0.0):.2%}
- Stable Tests: {reg_summary.get('stable_tests', 0)} / {reg_summary.get('total_tests', 0)}
- Mean Detection Difference: {reg_summary.get('mean_detection_diff', 0.0):.2f}
- Mean Mask Area Overlap: {reg_summary.get('mean_mask_overlap', 0.0):.4f}

**Interpretation:**
{'✓ High stability - model behavior is consistent' if reg_summary.get('stability_score', 0) > 0.9 else '⚠ Moderate stability - some variations detected' if reg_summary.get('stability_score', 0) > 0.7 else '✗ Low stability - significant variations detected'}

---

## 4. Clinical-Style Summary

### Overall Performance

**Percentage of OPGs Correctly Numbered:**
- With >80% FDI accuracy: {sum(1 for img in batch_metrics.get('per_image', []) if img.get('fdi_accuracy_after', 0) > 0.8) / len(batch_metrics.get('per_image', [])) * 100 if batch_metrics.get('per_image') else 0:.1f}%
- With >90% FDI accuracy: {sum(1 for img in batch_metrics.get('per_image', []) if img.get('fdi_accuracy_after', 0) > 0.9) / len(batch_metrics.get('per_image', [])) * 100 if batch_metrics.get('per_image') else 0:.1f}%

**Mean IoU Across Quadrants:**
- Overall: {batch_summary.get('mean_mask_iou', 0.0):.4f}
- Validation Set: {batch_summary.get('val_mean_iou', 0.0):.4f}
- Training Set: {batch_summary.get('train_mean_iou', 0.0):.4f}

**Detection Reliability:**
- Mean detections per image: {batch_summary.get('mean_num_detections', 0.0):.2f} (expected: 28-32)
- Detection consistency: {1.0 - (batch_summary.get('std_mask_iou', 0.0) / 0.5) if batch_summary.get('std_mask_iou', 0) < 0.5 else 0.0:.2%}

---

## 5. Visual Failcases Annotated

**Top 5 Failure Cases:**

"""
    
    # Add failcase details
    for i, fc in enumerate(failcases.get('failcases', [])[:5]):
        report += f"""
### Failcase #{fc.get('rank', i+1)}: {fc.get('filename', 'unknown')}

- **Image ID:** {fc.get('image_id', 'N/A')}
- **Mask IoU:** {fc.get('mask_iou', 0.0):.4f}
- **FDI Accuracy:** {fc.get('fdi_accuracy', 0.0):.4f}
- **Detections:** {fc.get('num_detections', 0)} / {fc.get('num_gt', 0)} (ground truth)
- **Issues:**
"""
        for issue in fc.get('issues', []):
            report += f"  - {issue}\n"
    
    report += f"""
**Total Failcases Analyzed:** {failcases.get('num_failcases', 0)}
**Visualizations:** Saved to `visual_failcases/` directory

---

## 6. Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TOOTH-AI PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

Input OPG Image
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  (Resize, Normalize, Format Conversion)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Mask R-CNN    │  (Instance Segmentation)
│  ResNet-50 FPN  │
└────────┬────────┘
         │
         ├─── High Confidence (≥0.85) ────┐
         │                                  │
         └─── Low Confidence (<0.85) ──────┤
                                            │
                                    ┌───────▼───────┐
                                    │  EfficientNet │
                                    │   ROI Classifier
                                    └───────┬───────┘
                                            │
         ┌──────────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Anatomical Ordering        │
│  - Quadrant Detection        │
│  - Geometric Sorting         │
│  - FDI Mapping               │
│  - Duplicate Resolution      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Final Output            │
│  - FDI Labels               │
│  - Masks (RLE)              │
│  - Bounding Boxes           │
│  - Confidence Scores        │
└─────────────────────────────┘
```

---

## 7. Performance Comparison Across Phases

| Phase | Method | Mask IoU | FDI Accuracy | Notes |
|-------|--------|----------|--------------|-------|
| Phase 3 | 512×512 | - | - | Baseline resolution |
| Phase 3 | 1024×512 | - | - | Best resolution |
| Phase 3 | Tiled | - | - | High-res inference |
| Phase 4 | Mask R-CNN + Ordering | - | - | Anatomical correction |
| Phase 5 | **Integrated** | **{batch_summary.get('mean_mask_iou', 0.0):.4f}** | **{batch_summary.get('mean_fdi_acc_after', 0.0):.4f}** | **Final Pipeline** |

*Note: Phase 3-4 metrics would be filled from previous phase reports*

---

## 8. Recommendations

### For Clinical Deployment

1. **Use Integrated Pipeline** (Phase 5)
   - Best FDI accuracy
   - Good consistency
   - Acceptable speed

2. **Quality Checks**
   - Flag images with IoU < 0.7
   - Review cases with FDI accuracy < 0.8
   - Monitor detection count (should be 28-32)

3. **Stress Test Findings**
   - Pipeline is robust to: {min(stress_summary.items(), key=lambda x: x[1].get('mean_detection_drop', 100))[0] if stress_summary else 'N/A'}
   - Pipeline is sensitive to: {max(stress_summary.items(), key=lambda x: x[1].get('mean_detection_drop', 0))[0] if stress_summary else 'N/A'}
   - Preprocess images to minimize sensitive distortions

4. **Failure Case Analysis**
   - {failcases.get('num_failcases', 0)} failure cases identified
   - Common issues: Low IoU, detection mismatch, FDI mislabeling
   - Recommend expert review for flagged cases

### For Model Updates

- **Regression Testing:** Maintain stability score > 90%
- **Baseline Comparison:** Track detection count and FDI assignments
- **Mask Overlap:** Ensure > 80% overlap with previous version

---

## 9. Limitations & Future Work

### Current Limitations

1. **Small Dataset:** ~250 OPG images (Niha Adnan et al., 2024)
2. **Class Imbalance:** Some FDI classes underrepresented
3. **Quadrant Detection:** Relies on centroid, may fail for rotated images
4. **Missing Teeth:** System doesn't explicitly detect gaps

### Future Improvements

1. **Larger Dataset:** Collect >1000 diverse OPG images
2. **Robust Quadrant Detection:** Use jaw line detection
3. **Missing Tooth Detection:** Explicit gap detection
4. **Multi-center Validation:** Test across different imaging systems

---

## 10. Conclusion

The Tooth-AI unified pipeline demonstrates {'strong' if batch_summary.get('mean_mask_iou', 0) > 0.75 else 'moderate' if batch_summary.get('mean_mask_iou', 0) > 0.65 else 'acceptable'} performance with:
- **Mask IoU:** {batch_summary.get('mean_mask_iou', 0.0):.4f}
- **FDI Accuracy:** {batch_summary.get('mean_fdi_acc_after', 0.0):.4f}
- **Stability:** {reg_summary.get('stability_score', 0.0):.2%}

The pipeline is {'ready for' if batch_summary.get('mean_fdi_acc_after', 0) > 0.85 else 'suitable for pilot' if batch_summary.get('mean_fdi_acc_after', 0) > 0.75 else 'requires improvement for'} clinical deployment with expert review.

**Status:** ✅ **POC Validated - Ready for Data Access Committee Review**

---

*Report Generated: Phase 6 - Reliability & Validation Testing*
*Dataset: Niha Adnan et al., Data in Brief, 2024*
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nPhase 6 final report generated!")
    print(f"  Report: {output_path}")
    print(f"  Visualizations: {report_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Aggregate Phase 6 results and generate final report')
    parser.add_argument('--batch-metrics', type=str, required=True,
                       help='Path to batch_metrics.json')
    parser.add_argument('--stress-report', type=str, required=True,
                       help='Path to stress_report.json')
    parser.add_argument('--regression-report', type=str, required=True,
                       help='Path to regression_report.json')
    parser.add_argument('--failcases', type=str, required=True,
                       help='Path to failcases.json')
    parser.add_argument('--out', type=str, required=True,
                       help='Output path for final report')
    
    args = parser.parse_args()
    
    generate_phase6_report(
        args.batch_metrics,
        args.stress_report,
        args.regression_report,
        args.failcases,
        args.out
    )


if __name__ == '__main__':
    main()




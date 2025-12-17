#!/usr/bin/env python3
"""
Generate Phase 3 decision report based on resolution comparison results.
Answers key questions about which method to use for final model.
"""

import argparse
import json
import os
from pathlib import Path


def generate_decision_report(results_json_path: str, output_path: str):
    """
    Generate decision report from resolution comparison results.
    
    Args:
        results_json_path: Path to resolution_comparison.json
        output_path: Path to save decision report
    """
    # Load results
    with open(results_json_path, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    per_image = data['per_image']
    
    # Extract key metrics
    mean_512_iou = summary['mean_512_iou']
    mean_1024_iou = summary['mean_1024_iou']
    mean_tiled_iou = summary['mean_tiled_iou']
    
    mean_512_fdi = summary['mean_512_fdi']
    mean_1024_fdi = summary['mean_1024_fdi']
    mean_tiled_fdi = summary['mean_tiled_fdi']
    
    mean_512_map = summary['mean_512_map']
    mean_1024_map = summary['mean_1024_map']
    mean_tiled_map = summary['mean_tiled_map']
    
    winner_counts = summary['winner_counts']
    
    # Decision logic
    iou_improvement_tiled = mean_tiled_iou - mean_512_iou
    iou_improvement_1024 = mean_1024_iou - mean_512_iou
    
    # Question 1: Which method should be used?
    if mean_tiled_iou > mean_1024_iou and mean_tiled_iou > mean_512_iou:
        recommended_method = "tiled"
        recommended_reason = f"Tiled inference achieves highest mean IoU ({mean_tiled_iou:.4f})"
    elif mean_1024_iou > mean_512_iou:
        recommended_method = "1024×512"
        recommended_reason = f"1024×512 achieves better IoU ({mean_1024_iou:.4f}) than 512×512 ({mean_512_iou:.4f})"
    else:
        recommended_method = "512×512"
        recommended_reason = f"512×512 is sufficient with IoU of {mean_512_iou:.4f}"
    
    # Question 2: Does tiled inference significantly improve IoU?
    significant_improvement_threshold = 0.02
    tiled_significant = iou_improvement_tiled > significant_improvement_threshold
    tiled_improvement_analysis = (
        f"Tiled inference improves IoU by {iou_improvement_tiled:.4f} over 512×512. "
        f"This is {'significant' if tiled_significant else 'not significant'} "
        f"(threshold: {significant_improvement_threshold})."
    )
    
    # Question 3: Does FDI accuracy improve without noise?
    fdi_improvement_tiled = mean_tiled_fdi - mean_512_fdi
    fdi_improvement_1024 = mean_1024_fdi - mean_512_fdi
    
    fdi_analysis = (
        f"FDI accuracy improvements:\n"
        f"  - 1024×512: {fdi_improvement_1024:+.4f} ({mean_1024_fdi:.4f} vs {mean_512_fdi:.4f})\n"
        f"  - Tiled: {fdi_improvement_tiled:+.4f} ({mean_tiled_fdi:.4f} vs {mean_512_fdi:.4f})\n"
    )
    
    if abs(fdi_improvement_tiled) < 0.01 and abs(fdi_improvement_1024) < 0.01:
        fdi_analysis += "FDI accuracy shows minimal change, suggesting improvements are primarily in segmentation quality."
    elif fdi_improvement_tiled > 0.05 or fdi_improvement_1024 > 0.05:
        fdi_analysis += "Significant FDI accuracy improvements observed."
    else:
        fdi_analysis += "Modest FDI accuracy improvements observed."
    
    # Question 4: Are improvements worth GPU cost?
    # Estimate relative costs
    cost_512 = 1.0  # Baseline
    cost_1024 = 2.0  # ~2x due to larger input
    cost_tiled = 4.0  # ~4x due to multiple tiles
    
    cost_benefit_analysis = (
        f"Estimated relative GPU costs:\n"
        f"  - 512×512: {cost_512:.1f}x (baseline)\n"
        f"  - 1024×512: {cost_1024:.1f}x (~2x slower)\n"
        f"  - Tiled: {cost_tiled:.1f}x (~4x slower)\n\n"
    )
    
    if tiled_significant and mean_tiled_iou > mean_1024_iou + 0.01:
        cost_benefit_analysis += (
            f"Recommendation: Tiled inference provides significant improvement ({iou_improvement_tiled:.4f} IoU) "
            f"but at 4x computational cost. Consider using for high-priority cases or as a post-processing step."
        )
    elif mean_1024_iou > mean_512_iou + 0.01:
        cost_benefit_analysis += (
            f"Recommendation: 1024×512 provides good balance with {iou_improvement_1024:.4f} IoU improvement "
            f"at 2x computational cost. Recommended for production use."
        )
    else:
        cost_benefit_analysis += (
            f"Recommendation: 512×512 is sufficient for most cases. "
            f"Higher resolution provides minimal improvement ({iou_improvement_1024:.4f} IoU) "
            f"not worth the 2x cost increase."
        )
    
    # Generate report
    report = f"""# Phase 3 Decision Report: Resolution Impact Analysis

## Executive Summary

This report analyzes the impact of image resolution and tiling on Mask R-CNN performance for tooth detection.
Three methods were evaluated: 512×512 direct inference, 1024×512 direct inference, and tiled 1024×1024 inference.

**Key Findings:**
- Mean Mask IoU: 512×512={mean_512_iou:.4f}, 1024×512={mean_1024_iou:.4f}, Tiled={mean_tiled_iou:.4f}
- Mean bbox mAP@0.5: 512×512={mean_512_map:.4f}, 1024×512={mean_1024_map:.4f}, Tiled={mean_tiled_map:.4f}
- Mean FDI Accuracy: 512×512={mean_512_fdi:.4f}, 1024×512={mean_1024_fdi:.4f}, Tiled={mean_tiled_fdi:.4f}

---

## Question 1: Which Method Should Be Used for Final Model?

**Answer: {recommended_method}**

{recommended_reason}

**Winner Distribution:**
- 512×512: {winner_counts['512']} images ({winner_counts['512']/summary['num_images']*100:.1f}%)
- 1024×512: {winner_counts['1024']} images ({winner_counts['1024']/summary['num_images']*100:.1f}%)
- Tiled: {winner_counts['tiled']} images ({winner_counts['tiled']/summary['num_images']*100:.1f}%)

---

## Question 2: Does Tiled Inference Significantly Improve IoU?

**Answer: {'Yes' if tiled_significant else 'No'}**

{tiled_improvement_analysis}

**Detailed Comparison:**
- Tiled vs 512×512: {iou_improvement_tiled:+.4f} IoU improvement
- Tiled vs 1024×512: {mean_tiled_iou - mean_1024_iou:+.4f} IoU improvement
- 1024×512 vs 512×512: {iou_improvement_1024:+.4f} IoU improvement

---

## Question 3: Does FDI Accuracy Improve Without Noise?

**Answer:** See analysis below.

{fdi_analysis}

---

## Question 4: Are Improvements Worth GPU Cost?

**Answer:** See cost-benefit analysis below.

{cost_benefit_analysis}

---

## Detailed Metrics

### Mask IoU Statistics
- **512×512**: Mean={mean_512_iou:.4f}
- **1024×512**: Mean={mean_1024_iou:.4f} (Improvement: {iou_improvement_1024:+.4f})
- **Tiled**: Mean={mean_tiled_iou:.4f} (Improvement: {iou_improvement_tiled:+.4f})

### Bbox mAP@0.5 Statistics
- **512×512**: Mean={mean_512_map:.4f}
- **1024×512**: Mean={mean_1024_map:.4f}
- **Tiled**: Mean={mean_tiled_map:.4f}

### FDI Classification Accuracy
- **512×512**: Mean={mean_512_fdi:.4f}
- **1024×512**: Mean={mean_1024_fdi:.4f}
- **Tiled**: Mean={mean_tiled_fdi:.4f}

---

## Recommendations

### Primary Recommendation
**Use {recommended_method} for production deployment.**

### Secondary Recommendations
1. **For high-priority cases**: Consider tiled inference if IoU improvement > 0.02 is critical
2. **For real-time applications**: Use 512×512 for speed, 1024×512 for better accuracy
3. **For batch processing**: Use 1024×512 as a good balance between accuracy and speed

### Implementation Notes
- Tiled inference requires additional post-processing (soft-NMS, clustering)
- 1024×512 maintains aspect ratio better than 512×512 square crops
- Consider hybrid strategy: use 1024×512 for most cases, tiled for difficult images

---

## Next Steps

1. **Phase 4 Integration**: Implement recommended method in final pipeline
2. **Hybrid Strategy**: If tiled shows significant improvement, implement hybrid inference
3. **Fine-tuning**: Consider fine-tuning model on higher resolution if dataset allows
4. **Error Analysis**: Analyze which teeth/regions benefit most from higher resolution

---

*Report generated from {summary['num_images']} validation images*
"""
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Decision report saved to: {output_path}")
    print(f"\nKey Recommendation: Use {recommended_method}")
    print(f"  - IoU Improvement: {iou_improvement_tiled:.4f} (tiled), {iou_improvement_1024:.4f} (1024)")


def main():
    parser = argparse.ArgumentParser(description='Generate Phase 3 decision report')
    parser.add_argument('--results-json', type=str, required=True,
                       help='Path to resolution_comparison.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for decision report')
    
    args = parser.parse_args()
    
    generate_decision_report(args.results_json, args.output)


if __name__ == '__main__':
    main()




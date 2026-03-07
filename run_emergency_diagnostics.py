#!/usr/bin/env python3
"""
Emergency Diagnostic Batch Runner
==================================
Runs all critical diagnostics in sequence and generates comprehensive report.

This script automates the entire diagnostic workflow for medical-grade validation.

Usage:
    python run_emergency_diagnostics.py path/to/failing_image.jpg
"""

import argparse
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time


class EmergencyDiagnostics:
    """Orchestrate all diagnostic tests and compile results."""
    
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.results = {
            'image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'diagnostics': {}
        }
        
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    
    def run_diagnostic(self, name, command, timeout=300):
        """Run a diagnostic script and capture results."""
        print(f"\n{'='*70}")
        print(f"🔍 Running: {name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            
            self.results['diagnostics'][name] = {
                'status': 'success' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed_time': elapsed
            }
            
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            
            if result.returncode == 0:
                print(f"✅ {name} completed in {elapsed:.1f}s")
            else:
                print(f"⚠️  {name} failed with code {result.returncode}")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"❌ {name} timed out after {timeout}s")
            self.results['diagnostics'][name] = {
                'status': 'timeout',
                'timeout': timeout
            }
            return False
        except Exception as e:
            print(f"❌ {name} crashed: {str(e)}")
            self.results['diagnostics'][name] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def generate_summary_report(self):
        """Generate medical-grade summary report."""
        print(f"\n{'='*70}")
        print("📋 EMERGENCY DIAGNOSTIC SUMMARY REPORT")
        print(f"{'='*70}")
        print(f"Image: {self.image_path.name}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"\n{'='*70}")
        
        # Load individual diagnostic results
        raw_pred_path = Path("diagnostic_report.json")
        sweep_path = Path("threshold_sweep.json")
        model_comp_path = Path("model_comparison.json")
        
        # Parse key findings
        findings = {
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 1. Model Comparison Analysis
        if model_comp_path.exists():
            with open(model_comp_path) as f:
                model_results = json.load(f)
            
            print("\n🏆 MODEL COMPARISON RESULTS:")
            print(f"{'Model':<30} {'Teeth':<8} {'Anomalies':<12} {'Status':<10}")
            print("-" * 70)
            
            best_model = None
            best_teeth = -1
            
            for model_name, data in model_results.items():
                if isinstance(data, dict) and 'teeth_detected' in data:
                    teeth = data['teeth_detected']
                    anomalies = data.get('anomalies_detected', 0)
                    
                    if teeth > best_teeth:
                        best_teeth = teeth
                        best_model = model_name
                    
                    status = "✅ GOOD" if teeth >= 20 else ("⚠️  POOR" if teeth > 0 else "❌ FAIL")
                    print(f"{model_name:<30} {teeth:<8} {anomalies:<12} {status:<10}")
            
            if best_model:
                findings['recommendations'].append(f"DEPLOY {best_model} (detected {best_teeth} teeth)")
            else:
                findings['critical_issues'].append("ALL MODELS FAILED - Fine-tuning required")
        
        # 2. Raw Predictions Analysis
        if raw_pred_path.exists():
            with open(raw_pred_path) as f:
                raw_data = json.load(f)
            
            total = raw_data.get('total_predictions', 0)
            class_dist = raw_data.get('class_distribution', {})
            tooth_count = class_dist.get('Tooth', 0)
            
            print(f"\n🔬 RAW PREDICTIONS:")
            print(f"   Total predictions: {total}")
            print(f"   Tooth predictions: {tooth_count}")
            
            if tooth_count == 0:
                findings['critical_issues'].append("Model produces ZERO tooth predictions")
            elif tooth_count < 20:
                findings['warnings'].append(f"Low tooth count ({tooth_count}) - threshold or domain shift issue")
            
            # Score analysis
            if 'score_statistics' in raw_data and raw_data['score_statistics']:
                score_max = raw_data['score_statistics'].get('max', 0)
                print(f"   Max score: {score_max:.3f}")
                
                if score_max < 0.15:
                    findings['critical_issues'].append(f"Max tooth score very low ({score_max:.3f}) - severe domain shift")
                elif score_max < 0.35:
                    findings['warnings'].append(f"Tooth scores below threshold ({score_max:.3f}) - lower threshold needed")
        
        # 3. Threshold Sweep Analysis
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            
            optimal_thresh = sweep_data.get('optimal_threshold', 0)
            optimal_count = sweep_data.get('optimal_count', 0)
            
            print(f"\n📊 THRESHOLD SWEEP:")
            print(f"   Optimal threshold: {optimal_thresh:.3f}")
            print(f"   Teeth at optimal: {optimal_count}")
            
            if optimal_count >= 24:
                findings['recommendations'].append(f"Use threshold {optimal_thresh:.3f} (yields {optimal_count} teeth)")
            elif optimal_count > 0:
                findings['warnings'].append(f"Threshold {optimal_thresh:.3f} only yields {optimal_count} teeth")
        
        # Final Assessment
        print(f"\n{'='*70}")
        print("🎯 CLINICAL ASSESSMENT")
        print(f"{'='*70}")
        
        if findings['critical_issues']:
            print("\n❌ CRITICAL ISSUES:")
            for issue in findings['critical_issues']:
                print(f"   • {issue}")
        
        if findings['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in findings['warnings']:
                print(f"   • {warning}")
        
        if findings['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in findings['recommendations']:
                print(f"   • {rec}")
        
        # Medical Decision
        print(f"\n{'='*70}")
        print("🏥 MEDICAL DECISION")
        print(f"{'='*70}")
        
        if best_model and best_teeth >= 20:
            print(f"\n✅ RESOLUTION IDENTIFIED:")
            print(f"   Deploy {best_model} immediately")
            print(f"   Expected performance: {best_teeth}/32 teeth detected")
            print(f"\n   ACTION REQUIRED:")
            print(f"   1. Update app.py: MODEL_DIR = Path('output/{best_model}')")
            print(f"   2. Restart Streamlit app")
            print(f"   3. Validate on 10 test images")
            print(f"\n   ESTIMATED TIME TO RESOLUTION: 15 minutes")
        elif optimal_count >= 20 and optimal_thresh > 0:
            print(f"\n⚡ THRESHOLD ADJUSTMENT NEEDED:")
            print(f"   Lower threshold to {optimal_thresh:.3f}")
            print(f"   Expected: {optimal_count} teeth detected")
            print(f"\n   ACTION REQUIRED:")
            print(f"   1. Update app.py line 637: threshold = {optimal_thresh:.3f}")
            print(f"   2. Restart app and test")
            print(f"\n   ESTIMATED TIME TO RESOLUTION: 10 minutes")
        else:
            print(f"\n🚨 SYSTEM OFFLINE - FINE-TUNING REQUIRED:")
            print(f"   All models failed on this dataset")
            print(f"   Domain shift confirmed (severe)")
            print(f"\n   ACTION REQUIRED:")
            print(f"   1. Annotate 100-150 images from new dataset")
            print(f"   2. Fine-tune best model for 5K iterations")
            print(f"   3. Validate to medical-grade standards")
            print(f"\n   ESTIMATED TIME TO RESOLUTION: 3-5 days")
            print(f"\n   ⚠️  PATIENT SAFETY: SUSPEND clinical use until resolved")
        
        print(f"\n{'='*70}")
        
        # Save summary
        self.results['findings'] = findings
        self.results['best_model'] = best_model if best_model else None
        self.results['best_teeth_count'] = best_teeth if best_teeth > 0 else 0
        
        return self.results
    
    def run_all(self):
        """Execute full diagnostic suite."""
        print(f"{'='*70}")
        print("🚨 EMERGENCY DIAGNOSTIC SUITE")
        print(f"{'='*70}")
        print(f"Patient Safety: CRITICAL")
        print(f"Image: {self.image_path}")
        print(f"System Status: PRODUCTION FAILURE (0 teeth detected)")
        print(f"{'='*70}")
        
        # Test 1: Model Comparison (HIGHEST PRIORITY)
        success = self.run_diagnostic(
            "Model Comparison",
            f'python model_comparison.py "{self.image_path}" --models resnet50_9class_20k resnext101_cascade_60k rtx4060_48k'
        )
        
        if not success:
            print("\n⚠️  Model comparison failed, trying individual models...")
        
        # Test 2: Raw Predictions Analysis
        self.run_diagnostic(
            "Raw Predictions Analysis",
            f'python diagnostic_raw_predictions.py "{self.image_path}" --model resnet50_9class_20k'
        )
        
        # Test 3: Threshold Sweep
        self.run_diagnostic(
            "Threshold Sweep",
            f'python diagnostic_threshold_sweep.py "{self.image_path}" --output threshold_sweep.png'
        )
        
        # Test 4: Preprocessing Validation
        self.run_diagnostic(
            "Preprocessing Validation",
            f'python diagnostic_preprocessing.py "{self.image_path}" --save-intermediates'
        )
        
        # Generate final report
        final_results = self.generate_summary_report()
        
        # Save comprehensive results
        output_file = Path(f"emergency_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📄 Full diagnostic report saved to: {output_file}")
        
        return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Emergency diagnostic suite for critical system failure"
    )
    parser.add_argument("image", type=str, help="Path to failing panoramic X-ray image")
    
    args = parser.parse_args()
    
    try:
        diagnostics = EmergencyDiagnostics(args.image)
        results = diagnostics.run_all()
        
        # Exit code based on severity
        if results.get('best_teeth_count', 0) >= 20:
            print("\n✅ Solution identified - fix available")
            sys.exit(0)
        elif results.get('best_teeth_count', 0) > 0:
            print("\n⚠️  Partial solution - manual intervention needed")
            sys.exit(1)
        else:
            print("\n❌ Critical failure - fine-tuning required")
            sys.exit(2)
            
    except Exception as e:
        print(f"\n❌ DIAGNOSTIC SUITE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()

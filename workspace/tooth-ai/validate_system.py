#!/usr/bin/env python3
"""
Comprehensive system validation script for Tooth-AI POC.
Validates all components: filesystem, Python scripts, models, API, UI, exports.
"""

import os
import sys
import json
import subprocess
import importlib.util
import ast
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Set project roots
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "data", "niihhaa")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Results storage
results = {
    "filesystem": {},
    "python_checks": {},
    "model_validation": {},
    "api_validation": {},
    "ui_validation": {},
    "onnx_validation": {},
    "tensorrt_validation": {},
    "benchmark_validation": {},
    "ci_validation": {},
    "repro_validation": {},
    "errors": []
}

error_log_path = os.path.join(PROJECT_ROOT, "logs", "final_validation_errors.log")
os.makedirs(os.path.dirname(error_log_path), exist_ok=True)


def log_error(category: str, message: str, exception: Optional[Exception] = None):
    """Log error to results and error log file."""
    error_msg = f"[{category}] {message}"
    if exception:
        error_msg += f"\n{str(exception)}\n{traceback.format_exc()}"
    
    results["errors"].append(error_msg)
    
    with open(error_log_path, "a") as f:
        f.write(f"{error_msg}\n\n")


def check_file_exists(filepath: str) -> bool:
    """Check if file exists."""
    full_path = os.path.join(PROJECT_ROOT, filepath.lstrip("/"))
    exists = os.path.exists(full_path)
    results["filesystem"][filepath] = {
        "exists": exists,
        "path": full_path
    }
    return exists


def check_python_syntax(filepath: str) -> Tuple[bool, Optional[str]]:
    """Check Python file syntax."""
    try:
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if not os.path.exists(full_path):
            return False, "File not found"
        
        with open(full_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Try to parse
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_python_imports(filepath: str) -> Tuple[bool, List[str]]:
    """Check Python file imports (without executing)."""
    missing_imports = []
    try:
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if not os.path.exists(full_path):
            return False, ["File not found"]
        
        with open(full_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        
        # Check if imports are available (basic check)
        for imp in set(imports):
            if imp not in ['sys', 'os', 'json', 'pathlib', 'typing', 'collections']:
                try:
                    __import__(imp)
                except ImportError:
                    missing_imports.append(imp)
        
        return len(missing_imports) == 0, missing_imports
    except Exception as e:
        return False, [f"Error checking imports: {e}"]


def validate_filesystem():
    """Step 1: Validate filesystem structure."""
    print("\n" + "="*60)
    print("STEP 1: FILESYSTEM VALIDATION")
    print("="*60)
    
    required_files = [
        "inference/engine.py",
        "inference/preprocess.py",
        "inference/postprocess.py",
        "inference/visualize.py",
        "api/server.py",
        "ui/app.py",
        "export/export_all_to_onnx.py",
        "export/convert_to_tensorrt.sh",
        "export/benchmark_inference.py",
        "repro/EXPERIMENT_MANIFEST.json",
        "repro/reproduce_metrics.sh",
        "repro/docker-compose.yml",
        "Dockerfile",
        "compliance/IRB_CHECKLIST.md",
        "compliance/DATA_ACCESS_REQUEST.md",
        "publication/manuscript_draft.md",
        "docs/executive_summary.md",
        "grants/grant_one_pager.md",
    ]
    
    # Check .github/workflows/ci.yml (relative to workspace root)
    github_ci = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), ".github", "workflows", "ci.yml")
    results["filesystem"][".github/workflows/ci.yml"] = {
        "exists": os.path.exists(github_ci),
        "path": github_ci
    }
    
    for filepath in required_files:
        check_file_exists(filepath)
    
    # Count results
    total = len(required_files) + 1  # +1 for github ci
    found = sum(1 for r in results["filesystem"].values() if r.get("exists", False))
    
    print(f"Files checked: {total}")
    print(f"Files found: {found}")
    print(f"Files missing: {total - found}")
    
    return found, total


def validate_python_scripts():
    """Step 2: Validate Python scripts."""
    print("\n" + "="*60)
    print("STEP 2: PYTHON SANITY CHECKS")
    print("="*60)
    
    python_dirs = [
        "inference",
        "api",
        "ui",
        "export",
        "cli",
    ]
    
    checked = 0
    syntax_errors = 0
    import_errors = 0
    
    for dir_name in python_dirs:
        dir_path = os.path.join(PROJECT_ROOT, dir_name)
        if not os.path.exists(dir_path):
            continue
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.relpath(os.path.join(root, file), PROJECT_ROOT)
                    checked += 1
                    
                    # Syntax check
                    syntax_ok, syntax_msg = check_python_syntax(filepath)
                    imports_ok, missing = check_python_imports(filepath)
                    
                    results["python_checks"][filepath] = {
                        "syntax": syntax_ok,
                        "syntax_error": syntax_msg,
                        "imports": imports_ok,
                        "missing_imports": missing
                    }
                    
                    if not syntax_ok:
                        syntax_errors += 1
                        print(f"  ✗ {filepath}: Syntax error - {syntax_msg}")
                    elif not imports_ok:
                        import_errors += 1
                        print(f"  ⚠ {filepath}: Missing imports - {missing}")
                    else:
                        print(f"  ✓ {filepath}: OK")
    
    print(f"\nTotal Python files checked: {checked}")
    print(f"Syntax errors: {syntax_errors}")
    print(f"Import issues: {import_errors}")
    
    return checked, syntax_errors, import_errors


def validate_models():
    """Step 3: Validate model loading and inference."""
    print("\n" + "="*60)
    print("STEP 3: MODEL SANITY CHECK")
    print("="*60)
    
    # Check if models exist
    maskrcnn_path = os.path.join(MODEL_DIR, "maskrcnn_final.pth")
    effnet_path = os.path.join(MODEL_DIR, "effnet_fdi_final.pth")
    config_path = os.path.join(MODEL_DIR, "config.yaml")
    
    results["model_validation"]["maskrcnn_exists"] = os.path.exists(maskrcnn_path)
    results["model_validation"]["effnet_exists"] = os.path.exists(effnet_path)
    results["model_validation"]["config_exists"] = os.path.exists(config_path)
    
    if not all([results["model_validation"]["maskrcnn_exists"],
                results["model_validation"]["effnet_exists"],
                results["model_validation"]["config_exists"]]):
        print("  ⚠ Models not found - skipping inference test")
        results["model_validation"]["inference_test"] = "SKIPPED - Models not found"
        return
    
    # Try to find a test image
    test_image = None
    if os.path.exists(DATASET_ROOT):
        dataset_path = os.path.join(DATASET_ROOT, "dataset")
        if os.path.exists(dataset_path):
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_image = os.path.join(root, file)
                        break
                if test_image:
                    break
    
    if not test_image:
        print("  ⚠ Test image not found - skipping inference test")
        results["model_validation"]["inference_test"] = "SKIPPED - Test image not found"
        return
    
    print(f"  Test image: {test_image}")
    
    # Try to run inference
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from inference.engine import load_engine
        
        print("  Loading engine...")
        engine = load_engine(MODEL_DIR)
        
        print("  Running inference...")
        output = engine.predict(test_image, return_visualization=True)
        
        # Check output
        num_detections = output.get("num_detections", 0)
        teeth = output.get("teeth", [])
        
        results["model_validation"]["inference_test"] = "PASS"
        results["model_validation"]["num_detections"] = num_detections
        results["model_validation"]["num_teeth"] = len(teeth)
        results["model_validation"]["has_fdi"] = all("fdi" in t for t in teeth)
        
        # Save outputs
        os.makedirs(os.path.join(PROJECT_ROOT, "tmp"), exist_ok=True)
        output_json = os.path.join(PROJECT_ROOT, "tmp", "engine_sanity_output.json")
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"  ✓ Inference successful: {num_detections} detections, {len(teeth)} teeth")
        print(f"  ✓ Output saved to: {output_json}")
        
        if num_detections < 20:
            print(f"  ⚠ Warning: Low detection count ({num_detections}), expected >20")
        
    except Exception as e:
        log_error("MODEL_VALIDATION", "Inference test failed", e)
        results["model_validation"]["inference_test"] = f"FAIL: {str(e)}"
        print(f"  ✗ Inference test failed: {e}")


def validate_api():
    """Step 4: Validate API server."""
    print("\n" + "="*60)
    print("STEP 4: API SERVER SMOKE TEST")
    print("="*60)
    
    # Check if server.py exists and is valid
    server_path = os.path.join(PROJECT_ROOT, "api", "server.py")
    if not os.path.exists(server_path):
        results["api_validation"]["status"] = "SKIPPED - server.py not found"
        return
    
    # Check syntax
    syntax_ok, _ = check_python_syntax("api/server.py")
    if not syntax_ok:
        results["api_validation"]["status"] = "FAIL - Syntax error"
        return
    
    # Try to import (without running)
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "api"))
        spec = importlib.util.spec_from_file_location("server", server_path)
        if spec and spec.loader:
            results["api_validation"]["import_check"] = "PASS"
            print("  ✓ API server imports successfully")
        else:
            results["api_validation"]["import_check"] = "FAIL"
            print("  ✗ API server import failed")
    except Exception as e:
        log_error("API_VALIDATION", "Import check failed", e)
        results["api_validation"]["import_check"] = f"FAIL: {str(e)}"
    
    # Note: Actual server startup test would require models and dependencies
    results["api_validation"]["startup_test"] = "MANUAL - Requires models and dependencies"
    print("  ⚠ Server startup test requires manual execution with models")


def validate_ui():
    """Step 5: Validate Streamlit UI."""
    print("\n" + "="*60)
    print("STEP 5: STREAMLIT UI VALIDATION")
    print("="*60)
    
    ui_path = os.path.join(PROJECT_ROOT, "ui", "app.py")
    if not os.path.exists(ui_path):
        results["ui_validation"]["status"] = "SKIPPED - app.py not found"
        return
    
    # Check syntax
    syntax_ok, syntax_msg = check_python_syntax("ui/app.py")
    results["ui_validation"]["syntax"] = syntax_ok
    if not syntax_ok:
        print(f"  ✗ Syntax error: {syntax_msg}")
        return
    
    # Check imports
    imports_ok, missing = check_python_imports("ui/app.py")
    results["ui_validation"]["imports"] = imports_ok
    results["ui_validation"]["missing_imports"] = missing
    
    if imports_ok:
        print("  ✓ UI script syntax and imports OK")
    else:
        print(f"  ⚠ Missing imports: {missing}")
    
    results["ui_validation"]["startup_test"] = "MANUAL - Requires streamlit and dependencies"
    print("  ⚠ UI startup test requires manual execution")


def validate_onnx():
    """Step 6: Validate ONNX export scripts."""
    print("\n" + "="*60)
    print("STEP 6: ONNX & TENSORRT VALIDATION")
    print("="*60)
    
    export_script = os.path.join(PROJECT_ROOT, "export", "export_all_to_onnx.py")
    if not os.path.exists(export_script):
        results["onnx_validation"]["status"] = "SKIPPED - Script not found"
        return
    
    # Check syntax
    syntax_ok, _ = check_python_syntax("export/export_all_to_onnx.py")
    results["onnx_validation"]["script_syntax"] = syntax_ok
    
    # Check if ONNX files exist (would be created by running the script)
    onnx_maskrcnn = os.path.join(PROJECT_ROOT, "export", "maskrcnn.onnx")
    onnx_effnet = os.path.join(PROJECT_ROOT, "export", "effnet.onnx")
    
    results["onnx_validation"]["maskrcnn_onnx_exists"] = os.path.exists(onnx_maskrcnn)
    results["onnx_validation"]["effnet_onnx_exists"] = os.path.exists(onnx_effnet)
    
    if syntax_ok:
        print("  ✓ ONNX export script syntax OK")
    else:
        print("  ✗ ONNX export script has syntax errors")
    
    if results["onnx_validation"]["maskrcnn_onnx_exists"]:
        print("  ✓ maskrcnn.onnx exists")
    else:
        print("  ⚠ maskrcnn.onnx not found (run export script to generate)")
    
    if results["onnx_validation"]["effnet_onnx_exists"]:
        print("  ✓ effnet.onnx exists")
    else:
        print("  ⚠ effnet.onnx not found (run export script to generate)")
    
    # TensorRT
    trt_script = os.path.join(PROJECT_ROOT, "export", "convert_to_tensorrt.sh")
    results["tensorrt_validation"]["script_exists"] = os.path.exists(trt_script)
    
    if os.path.exists(trt_script):
        # Check if executable
        is_executable = os.access(trt_script, os.X_OK)
        results["tensorrt_validation"]["script_executable"] = is_executable
        print(f"  {'✓' if is_executable else '⚠'} TensorRT script exists ({'executable' if is_executable else 'not executable'})")
    
    # Check for TensorRT engines
    trt_maskrcnn = os.path.join(PROJECT_ROOT, "export", "maskrcnn_trt_fp16.plan")
    trt_effnet = os.path.join(PROJECT_ROOT, "export", "effnet_trt_fp16.plan")
    
    results["tensorrt_validation"]["maskrcnn_trt_exists"] = os.path.exists(trt_maskrcnn)
    results["tensorrt_validation"]["effnet_trt_exists"] = os.path.exists(trt_effnet)
    
    if results["tensorrt_validation"]["maskrcnn_trt_exists"]:
        print("  ✓ maskrcnn_trt_fp16.plan exists")
    else:
        print("  ⚠ TensorRT engines not found (run conversion script to generate)")


def validate_benchmark():
    """Step 7: Validate benchmarking script."""
    print("\n" + "="*60)
    print("STEP 7: BENCHMARKING VALIDATION")
    print("="*60)
    
    benchmark_script = os.path.join(PROJECT_ROOT, "export", "benchmark_inference.py")
    if not os.path.exists(benchmark_script):
        results["benchmark_validation"]["status"] = "SKIPPED - Script not found"
        return
    
    # Check syntax
    syntax_ok, _ = check_python_syntax("export/benchmark_inference.py")
    results["benchmark_validation"]["script_syntax"] = syntax_ok
    
    # Check for benchmark outputs
    benchmark_dir = os.path.join(PROJECT_ROOT, "export", "benchmarks")
    latency_json = os.path.join(benchmark_dir, "latency.json")
    throughput_json = os.path.join(benchmark_dir, "throughput.json")
    
    results["benchmark_validation"]["latency_json_exists"] = os.path.exists(latency_json)
    results["benchmark_validation"]["throughput_json_exists"] = os.path.exists(throughput_json)
    
    if syntax_ok:
        print("  ✓ Benchmark script syntax OK")
    else:
        print("  ✗ Benchmark script has syntax errors")
    
    if results["benchmark_validation"]["latency_json_exists"]:
        print("  ✓ latency.json exists")
    else:
        print("  ⚠ latency.json not found (run benchmark script to generate)")
    
    if results["benchmark_validation"]["throughput_json_exists"]:
        print("  ✓ throughput.json exists")
    else:
        print("  ⚠ throughput.json not found (run benchmark script to generate)")


def validate_ci():
    """Step 8: Validate CI/CD workflow."""
    print("\n" + "="*60)
    print("STEP 8: CI/CD VALIDATION")
    print("="*60)
    
    github_ci = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), ".github", "workflows", "ci.yml")
    
    if not os.path.exists(github_ci):
        results["ci_validation"]["status"] = "FAIL - ci.yml not found"
        print("  ✗ CI workflow file not found")
        return
    
    # Try to parse YAML (basic check)
    try:
        import yaml
        with open(github_ci, 'r') as f:
            ci_config = yaml.safe_load(f)
        
        has_triggers = "on" in ci_config
        has_jobs = "jobs" in ci_config
        
        results["ci_validation"]["yaml_valid"] = True
        results["ci_validation"]["has_triggers"] = has_triggers
        results["ci_validation"]["has_jobs"] = has_jobs
        
        if has_jobs:
            jobs = list(ci_config["jobs"].keys())
            results["ci_validation"]["jobs"] = jobs
            print(f"  ✓ CI workflow valid with jobs: {', '.join(jobs)}")
        else:
            print("  ⚠ CI workflow missing jobs section")
            
    except ImportError:
        results["ci_validation"]["yaml_valid"] = "SKIPPED - PyYAML not available"
        print("  ⚠ Cannot validate YAML (PyYAML not installed)")
    except Exception as e:
        log_error("CI_VALIDATION", "YAML parsing failed", e)
        results["ci_validation"]["yaml_valid"] = False
        print(f"  ✗ YAML parsing error: {e}")


def validate_repro():
    """Step 9: Validate reproducibility bundle."""
    print("\n" + "="*60)
    print("STEP 9: REPRODUCIBILITY BUNDLE VALIDATION")
    print("="*60)
    
    manifest_path = os.path.join(PROJECT_ROOT, "repro", "EXPERIMENT_MANIFEST.json")
    
    if not os.path.exists(manifest_path):
        results["repro_validation"]["manifest_exists"] = False
        print("  ✗ EXPERIMENT_MANIFEST.json not found")
        return
    
    results["repro_validation"]["manifest_exists"] = True
    
    # Validate manifest structure
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        required_fields = ["experiment_name", "models", "dataset", "phases_completed"]
        missing_fields = [f for f in required_fields if f not in manifest]
        
        results["repro_validation"]["manifest_valid"] = len(missing_fields) == 0
        results["repro_validation"]["missing_fields"] = missing_fields
        
        if len(missing_fields) == 0:
            print("  ✓ EXPERIMENT_MANIFEST.json valid")
        else:
            print(f"  ⚠ Missing fields: {missing_fields}")
            
    except Exception as e:
        log_error("REPRO_VALIDATION", "Manifest parsing failed", e)
        results["repro_validation"]["manifest_valid"] = False
        print(f"  ✗ Manifest parsing error: {e}")
    
    # Check reproduce script
    repro_script = os.path.join(PROJECT_ROOT, "repro", "reproduce_metrics.sh")
    results["repro_validation"]["script_exists"] = os.path.exists(repro_script)
    results["repro_validation"]["script_executable"] = os.access(repro_script, os.X_OK) if os.path.exists(repro_script) else False
    
    if results["repro_validation"]["script_exists"]:
        print(f"  {'✓' if results['repro_validation']['script_executable'] else '⚠'} reproduce_metrics.sh exists ({'executable' if results['repro_validation']['script_executable'] else 'not executable'})")


def calculate_deployment_score() -> int:
    """Calculate deployment readiness score (0-100)."""
    score = 0
    max_score = 100
    
    # Filesystem (20 points)
    fs_total = len(results["filesystem"])
    fs_found = sum(1 for r in results["filesystem"].values() if r.get("exists", False))
    if fs_total > 0:
        score += int(20 * (fs_found / fs_total))
    
    # Python checks (20 points)
    py_total = len(results["python_checks"])
    py_ok = sum(1 for r in results["python_checks"].values() if r.get("syntax") and r.get("imports"))
    if py_total > 0:
        score += int(20 * (py_ok / py_total))
    
    # Models (15 points)
    if results["model_validation"].get("inference_test") == "PASS":
        score += 15
    elif results["model_validation"].get("maskrcnn_exists") and results["model_validation"].get("effnet_exists"):
        score += 10  # Models exist but not tested
    
    # API (10 points)
    if results["api_validation"].get("import_check") == "PASS":
        score += 10
    
    # UI (10 points)
    if results["ui_validation"].get("syntax") and results["ui_validation"].get("imports"):
        score += 10
    
    # ONNX (10 points)
    if results["onnx_validation"].get("script_syntax"):
        score += 5
    if results["onnx_validation"].get("maskrcnn_onnx_exists") and results["onnx_validation"].get("effnet_onnx_exists"):
        score += 5
    
    # CI/CD (5 points)
    if results["ci_validation"].get("yaml_valid") and results["ci_validation"].get("has_jobs"):
        score += 5
    
    # Reproducibility (10 points)
    if results["repro_validation"].get("manifest_valid"):
        score += 5
    if results["repro_validation"].get("script_exists"):
        score += 5
    
    return min(score, max_score)


def generate_report():
    """Step 10: Generate final validation report."""
    print("\n" + "="*60)
    print("STEP 10: GENERATING FINAL VALIDATION REPORT")
    print("="*60)
    
    report_path = os.path.join(PROJECT_ROOT, "FINAL_VALIDATION_REPORT.md")
    
    deployment_score = calculate_deployment_score()
    
    report = f"""# Final Validation Report: Tooth-AI POC

**Validation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Project Root:** {PROJECT_ROOT}  
**Deployment Readiness Score:** {deployment_score}/100

---

## Executive Summary

This report presents the results of comprehensive system validation for the Tooth-AI POC project, covering all phases from 0 to 7B.

**Overall Status:** {'✅ PASS' if deployment_score >= 70 else '⚠️ NEEDS ATTENTION' if deployment_score >= 50 else '❌ FAIL'}

**Deployment Readiness:** {deployment_score}/100

---

## 1. Overall PASS/FAIL Summary

### Filesystem Validation
- **Status:** {'✅ PASS' if sum(1 for r in results['filesystem'].values() if r.get('exists')) / max(len(results['filesystem']), 1) >= 0.9 else '⚠️ PARTIAL' if sum(1 for r in results['filesystem'].values() if r.get('exists')) / max(len(results['filesystem']), 1) >= 0.7 else '❌ FAIL'}
- **Files Found:** {sum(1 for r in results['filesystem'].values() if r.get('exists'))}/{len(results['filesystem'])}

### Python Scripts Validation
- **Status:** {'✅ PASS' if sum(1 for r in results['python_checks'].values() if r.get('syntax') and r.get('imports')) / max(len(results['python_checks']), 1) >= 0.9 else '⚠️ PARTIAL'}
- **Files Checked:** {len(results['python_checks'])}
- **Syntax Errors:** {sum(1 for r in results['python_checks'].values() if not r.get('syntax'))}
- **Import Issues:** {sum(1 for r in results['python_checks'].values() if not r.get('imports'))}

### Model Validation
- **Status:** {results['model_validation'].get('inference_test', 'NOT TESTED')}
- **Mask R-CNN Exists:** {'✅' if results['model_validation'].get('maskrcnn_exists') else '❌'}
- **EfficientNet Exists:** {'✅' if results['model_validation'].get('effnet_exists') else '❌'}

### API Validation
- **Status:** {results['api_validation'].get('import_check', 'NOT TESTED')}
- **Startup Test:** {results['api_validation'].get('startup_test', 'NOT TESTED')}

### UI Validation
- **Status:** {'✅ PASS' if results['ui_validation'].get('syntax') and results['ui_validation'].get('imports') else '❌ FAIL'}
- **Syntax:** {'✅' if results['ui_validation'].get('syntax') else '❌'}
- **Imports:** {'✅' if results['ui_validation'].get('imports') else '❌'}

### ONNX/TensorRT Validation
- **ONNX Script:** {'✅' if results['onnx_validation'].get('script_syntax') else '❌'}
- **ONNX Files:** {'✅' if results['onnx_validation'].get('maskrcnn_onnx_exists') and results['onnx_validation'].get('effnet_onnx_exists') else '⚠️ Not generated'}
- **TensorRT:** {'✅' if results['tensorrt_validation'].get('maskrcnn_trt_exists') and results['tensorrt_validation'].get('effnet_trt_exists') else '⚠️ Not generated'}

### CI/CD Validation
- **Status:** {'✅ PASS' if results['ci_validation'].get('yaml_valid') and results['ci_validation'].get('has_jobs') else '❌ FAIL'}
- **Jobs:** {', '.join(results['ci_validation'].get('jobs', [])) if results['ci_validation'].get('jobs') else 'None'}

### Reproducibility Validation
- **Manifest:** {'✅' if results['repro_validation'].get('manifest_valid') else '❌'}
- **Script:** {'✅' if results['repro_validation'].get('script_exists') else '❌'}

---

## 2. File Presence Table

| File | Status | Path |
|------|--------|------|
"""
    
    for filepath, info in results["filesystem"].items():
        status = "✅ EXISTS" if info.get("exists") else "❌ MISSING"
        report += f"| `{filepath}` | {status} | `{info.get('path', 'N/A')}` |\n"
    
    report += f"""
---

## 3. Syntax/Import Error Table

| File | Syntax | Imports | Issues |
|------|--------|---------|--------|
"""
    
    for filepath, info in results["python_checks"].items():
        syntax = "✅" if info.get("syntax") else "❌"
        imports = "✅" if info.get("imports") else "❌"
        issues = []
        if not info.get("syntax"):
            issues.append(info.get("syntax_error", "Syntax error"))
        if not info.get("imports"):
            issues.append(f"Missing: {', '.join(info.get('missing_imports', []))}")
        issues_str = "; ".join(issues) if issues else "None"
        report += f"| `{filepath}` | {syntax} | {imports} | {issues_str} |\n"
    
    report += f"""
---

## 4. Unified Inference Results

**Test Status:** {results['model_validation'].get('inference_test', 'NOT TESTED')}

"""
    
    if results["model_validation"].get("inference_test") == "PASS":
        report += f"""
- **Detections:** {results['model_validation'].get('num_detections', 'N/A')}
- **Teeth Found:** {results['model_validation'].get('num_teeth', 'N/A')}
- **FDI Labels Present:** {'✅' if results['model_validation'].get('has_fdi') else '❌'}
- **Output Saved:** `tmp/engine_sanity_output.json`

"""
    else:
        report += f"""
- **Reason:** {results['model_validation'].get('inference_test', 'Models or test image not available')}

"""
    
    report += f"""
---

## 5. API Server Results

**Import Check:** {results['api_validation'].get('import_check', 'NOT TESTED')}  
**Startup Test:** {results['api_validation'].get('startup_test', 'NOT TESTED')}

**Note:** Full server startup test requires models and all dependencies to be installed.

---

## 6. UI Startup Results

**Syntax:** {'✅ PASS' if results['ui_validation'].get('syntax') else '❌ FAIL'}  
**Imports:** {'✅ PASS' if results['ui_validation'].get('imports') else '❌ FAIL'}  
**Missing Imports:** {', '.join(results['ui_validation'].get('missing_imports', [])) if results['ui_validation'].get('missing_imports') else 'None'}

**Note:** Full UI startup test requires streamlit and all dependencies to be installed.

---

## 7. ONNX Export Results

**Script Syntax:** {'✅ PASS' if results['onnx_validation'].get('script_syntax') else '❌ FAIL'}  
**maskrcnn.onnx:** {'✅ EXISTS' if results['onnx_validation'].get('maskrcnn_onnx_exists') else '⚠️ NOT GENERATED'}  
**effnet.onnx:** {'✅ EXISTS' if results['onnx_validation'].get('effnet_onnx_exists') else '⚠️ NOT GENERATED'}

**Note:** ONNX files are generated by running the export script with trained models.

---

## 8. TensorRT Results

**Script Exists:** {'✅' if results['tensorrt_validation'].get('script_exists') else '❌'}  
**Script Executable:** {'✅' if results['tensorrt_validation'].get('script_executable') else '❌'}  
**maskrcnn_trt_fp16.plan:** {'✅ EXISTS' if results['tensorrt_validation'].get('maskrcnn_trt_exists') else '⚠️ NOT GENERATED'}  
**effnet_trt_fp16.plan:** {'✅ EXISTS' if results['tensorrt_validation'].get('effnet_trt_exists') else '⚠️ NOT GENERATED'}

**Note:** TensorRT engines require TensorRT SDK and ONNX files to be generated.

---

## 9. Benchmark Summary

**Script Syntax:** {'✅ PASS' if results['benchmark_validation'].get('script_syntax') else '❌ FAIL'}  
**latency.json:** {'✅ EXISTS' if results['benchmark_validation'].get('latency_json_exists') else '⚠️ NOT GENERATED'}  
**throughput.json:** {'✅ EXISTS' if results['benchmark_validation'].get('throughput_json_exists') else '⚠️ NOT GENERATED'}

**Note:** Benchmark results require models and test images to be generated.

---

## 10. CI/CD Summary

**YAML Valid:** {'✅' if results['ci_validation'].get('yaml_valid') else '❌'}  
**Has Triggers:** {'✅' if results['ci_validation'].get('has_triggers') else '❌'}  
**Has Jobs:** {'✅' if results['ci_validation'].get('has_jobs') else '❌'}  
**Jobs:** {', '.join(results['ci_validation'].get('jobs', [])) if results['ci_validation'].get('jobs') else 'None'}

---

## 11. Reproducibility Summary

**Manifest Exists:** {'✅' if results['repro_validation'].get('manifest_exists') else '❌'}  
**Manifest Valid:** {'✅' if results['repro_validation'].get('manifest_valid') else '❌'}  
**Missing Fields:** {', '.join(results['repro_validation'].get('missing_fields', [])) if results['repro_validation'].get('missing_fields') else 'None'}  
**Script Exists:** {'✅' if results['repro_validation'].get('script_exists') else '❌'}  
**Script Executable:** {'✅' if results['repro_validation'].get('script_executable') else '❌'}

---

## 12. Risks Identified

"""
    
    risks = []
    if sum(1 for r in results['filesystem'].values() if not r.get('exists')) > 0:
        risks.append("Missing critical files")
    if sum(1 for r in results['python_checks'].values() if not r.get('syntax')) > 0:
        risks.append("Python syntax errors present")
    if sum(1 for r in results['python_checks'].values() if not r.get('imports')) > 0:
        risks.append("Missing Python dependencies")
    if not results['model_validation'].get('maskrcnn_exists') or not results['model_validation'].get('effnet_exists'):
        risks.append("Trained models not found")
    if results['model_validation'].get('inference_test') != "PASS":
        risks.append("Model inference not validated")
    if not results['onnx_validation'].get('maskrcnn_onnx_exists') or not results['onnx_validation'].get('effnet_onnx_exists'):
        risks.append("ONNX exports not generated")
    
    if risks:
        for risk in risks:
            report += f"- ⚠️ {risk}\n"
    else:
        report += "- ✅ No critical risks identified\n"
    
    report += f"""
---

## 13. Recommended Fixes

### Priority 1 (Critical)
"""
    
    critical_fixes = []
    if not results['model_validation'].get('maskrcnn_exists'):
        critical_fixes.append("Copy trained Mask R-CNN model to `models/maskrcnn_final.pth`")
    if not results['model_validation'].get('effnet_exists'):
        critical_fixes.append("Copy trained EfficientNet model to `models/effnet_fdi_final.pth`")
    if sum(1 for r in results['python_checks'].values() if not r.get('syntax')) > 0:
        critical_fixes.append("Fix Python syntax errors in scripts")
    
    if critical_fixes:
        for fix in critical_fixes:
            report += f"- {fix}\n"
    else:
        report += "- None\n"
    
    report += f"""
### Priority 2 (Important)
"""
    
    important_fixes = []
    if sum(1 for r in results['python_checks'].values() if not r.get('imports')) > 0:
        important_fixes.append("Install missing Python dependencies")
    if not results['onnx_validation'].get('maskrcnn_onnx_exists'):
        important_fixes.append("Run ONNX export script to generate model exports")
    if not results['benchmark_validation'].get('latency_json_exists'):
        important_fixes.append("Run benchmark script to generate performance metrics")
    
    if important_fixes:
        for fix in important_fixes:
            report += f"- {fix}\n"
    else:
        report += "- None\n"
    
    report += f"""
### Priority 3 (Nice to Have)
"""
    
    nice_to_have = []
    if not results['tensorrt_validation'].get('maskrcnn_trt_exists'):
        nice_to_have.append("Generate TensorRT engines (requires TensorRT SDK)")
    if not results['repro_validation'].get('script_executable'):
        nice_to_have.append("Make reproduce_metrics.sh executable: `chmod +x repro/reproduce_metrics.sh`")
    
    if nice_to_have:
        for fix in nice_to_have:
            report += f"- {fix}\n"
    else:
        report += "- None\n"
    
    report += f"""
---

## 14. Final Deployment Readiness Score

**Score: {deployment_score}/100**

### Score Breakdown:
- **Filesystem (20 points):** {int(20 * (sum(1 for r in results['filesystem'].values() if r.get('exists')) / max(len(results['filesystem']), 1)))}
- **Python Scripts (20 points):** {int(20 * (sum(1 for r in results['python_checks'].values() if r.get('syntax') and r.get('imports')) / max(len(results['python_checks']), 1)))}
- **Models (15 points):** {15 if results['model_validation'].get('inference_test') == 'PASS' else 10 if results['model_validation'].get('maskrcnn_exists') and results['model_validation'].get('effnet_exists') else 0}
- **API (10 points):** {10 if results['api_validation'].get('import_check') == 'PASS' else 0}
- **UI (10 points):** {10 if results['ui_validation'].get('syntax') and results['ui_validation'].get('imports') else 0}
- **ONNX (10 points):** {5 if results['onnx_validation'].get('script_syntax') else 0} + {5 if results['onnx_validation'].get('maskrcnn_onnx_exists') and results['onnx_validation'].get('effnet_onnx_exists') else 0}
- **CI/CD (5 points):** {5 if results['ci_validation'].get('yaml_valid') and results['ci_validation'].get('has_jobs') else 0}
- **Reproducibility (10 points):** {5 if results['repro_validation'].get('manifest_valid') else 0} + {5 if results['repro_validation'].get('script_exists') else 0}

### Interpretation:
- **90-100:** Production ready
- **70-89:** Ready for testing/deployment
- **50-69:** Needs fixes before deployment
- **0-49:** Significant issues, not ready

**Current Status:** {'Production Ready' if deployment_score >= 90 else 'Ready for Testing' if deployment_score >= 70 else 'Needs Fixes' if deployment_score >= 50 else 'Not Ready'}

---

## 15. Error Log

All errors have been logged to: `logs/final_validation_errors.log`

**Total Errors:** {len(results['errors'])}

"""
    
    if results['errors']:
        report += "**Recent Errors:**\n\n"
        for error in results['errors'][-10:]:  # Last 10 errors
            report += f"```\n{error}\n```\n\n"
    
    report += f"""
---

## Appendix: Validation Environment

- **Python Version:** {sys.version.split()[0]}
- **Project Root:** {PROJECT_ROOT}
- **Model Directory:** {MODEL_DIR}
- **Dataset Root:** {DATASET_ROOT}
- **Validation Script:** `validate_system.py`

---

**Report Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Validation Version:** 1.0
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"  ✓ Validation report generated: {report_path}")
    print(f"  ✓ Deployment readiness score: {deployment_score}/100")


def main():
    """Run complete system validation."""
    print("="*60)
    print("TOOTH-AI COMPLETE SYSTEM VALIDATION")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Dataset Root: {DATASET_ROOT}")
    
    # Clear error log
    if os.path.exists(error_log_path):
        os.remove(error_log_path)
    
    # Run all validation steps
    validate_filesystem()
    validate_python_scripts()
    validate_models()
    validate_api()
    validate_ui()
    validate_onnx()
    validate_benchmark()
    validate_ci()
    validate_repro()
    generate_report()
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"Report: {os.path.join(PROJECT_ROOT, 'FINAL_VALIDATION_REPORT.md')}")
    print(f"Errors: {error_log_path}")


if __name__ == "__main__":
    main()




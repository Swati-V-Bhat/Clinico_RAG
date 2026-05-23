# ============================================================
# CELL 1 — Install dependencies
# ============================================================
import subprocess, sys

pkgs = [
    'torchxrayvision',
    'segmentation-models-pytorch',
    'faiss-cpu',
    'timm',
    'grad-cam',
    'sentence-transformers',
]
for pkg in pkgs:
    print(f"Installing {pkg}...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                             capture_output=True, text=True)
    print(f"  {'✅' if result.returncode == 0 else '❌'} {pkg}")
print("\n✅ All done — run Cell 2")

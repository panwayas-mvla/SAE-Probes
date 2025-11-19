#!/usr/bin/env python3
"""
Script to apply the sae_bench import patch to the virtual environment.
Run this after recreating your virtual environment or when the patch is missing.
"""
import os
import sys
from pathlib import Path

def apply_patch():
    # Find the virtual environment directory
    venv_dir = Path("probing")
    if not venv_dir.exists():
        print("Error: 'probing' virtual environment directory not found.")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Find the target file
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    target_file = venv_dir / "lib" / python_version / "site-packages" / "sae_bench" / "sae_bench_utils" / "general_utils.py"
    
    if not target_file.exists():
        print(f"Error: {target_file} not found.")
        print("Make sure sae_bench is installed in your virtual environment.")
        sys.exit(1)
    
    # Read the file
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Check if patch is already applied
    if 'try:' in content and 'from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory' in content:
        print(f"Patch already applied to {target_file}")
        return
    
    # Old import pattern to replace
    old_import = 'from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory'
    
    # New import with try/except
    new_import = '''try:
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
except ImportError:
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory'''
    
    # Apply the patch
    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(target_file, 'w') as f:
            f.write(content)
        print(f"Successfully applied patch to {target_file}")
    else:
        print(f"Warning: Could not find the expected import pattern in {target_file}")
        print("The file may have already been modified or has a different structure.")
        sys.exit(1)

if __name__ == "__main__":
    apply_patch()


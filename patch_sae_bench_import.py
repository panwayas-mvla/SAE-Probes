"""
Monkey-patch for sae_bench to handle the sae_lens import path change.

This file should be imported before any sae_bench imports to apply the fix.
Usage: Add 'import patch_sae_bench_import' at the top of your scripts.
"""
import sys

def patch_sae_bench_general_utils():
    """Patch the sae_bench general_utils module to handle import path changes."""
    try:
        # Try to import the module
        from sae_bench import sae_bench_utils
        general_utils = sae_bench_utils.general_utils
        
        # Check if the patch is already applied
        import inspect
        source = inspect.getsource(general_utils)
        if 'try:' in source and 'sae_lens.loading.pretrained_saes_directory' in source:
            # Patch already applied or already has try/except
            return
        
        # Read the original file
        import importlib.util
        spec = general_utils.__spec__
        if spec and spec.origin:
            with open(spec.origin, 'r') as f:
                content = f.read()
            
            # Check if we need to patch
            if 'from sae_lens.toolkit.pretrained_saes_directory import' in content:
                # Replace the import with try/except
                old_import = 'from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory'
                new_import = '''try:
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
except ImportError:
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory'''
                
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    # Write back (this modifies the venv file)
                    with open(spec.origin, 'w') as f:
                        f.write(content)
                    # Reload the module
                    importlib.reload(general_utils)
    except Exception as e:
        # If patching fails, that's okay - the module might already be patched
        # or the import might work with the new path
        pass

# Auto-apply patch when imported
patch_sae_bench_general_utils()


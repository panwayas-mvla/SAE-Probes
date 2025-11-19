# SAE Bench Patch Notes

## Issue
The `sae_bench` package imports from `sae_lens.toolkit.pretrained_saes_directory`, but newer versions of `sae_lens` have moved this to `sae_lens.loading.pretrained_saes_directory`.

## Solution
A patch has been applied to `probing/lib/python3.12/site-packages/sae_bench/sae_bench_utils/general_utils.py` to handle both import paths using a try/except block.

## To Reapply After Recreating Virtual Environment

1. **Option 1: Use the Python patch script (recommended)**
   ```bash
   python apply_sae_bench_patch.py
   ```
   or
   ```bash
   ./apply_sae_bench_patch.py
   ```

2. **Option 2: Use the shell script**
   ```bash
   ./apply_sae_bench_patch.sh
   ```

3. **Option 3: Manual patch**
   Edit `probing/lib/python3.12/site-packages/sae_bench/sae_bench_utils/general_utils.py` and replace:
   ```python
   from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
   ```
   with:
   ```python
   try:
       from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
   except ImportError:
       from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
   ```

## Files Modified
- `probing/lib/python3.12/site-packages/sae_bench/sae_bench_utils/general_utils.py` (in venv, not tracked by git)

## Files Created (tracked by git)
- `apply_sae_bench_patch.py` - Python script to reapply the patch (recommended)
- `apply_sae_bench_patch.sh` - Shell script to reapply the patch (alternative)
- `patch_sae_bench_import.py` - Runtime monkey-patch (alternative approach)
- `PATCH_NOTES.md` - This file


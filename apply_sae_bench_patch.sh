#!/bin/bash
# Script to apply the sae_bench import patch to the virtual environment
# Run this after recreating your virtual environment

VENV_DIR="probing"
SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages"
TARGET_FILE="$SITE_PACKAGES/sae_bench/sae_bench_utils/general_utils.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: $TARGET_FILE not found. Make sure sae_bench is installed in your venv."
    exit 1
fi

# Check if patch is already applied
if grep -q "from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory" "$TARGET_FILE"; then
    echo "Patch already applied to $TARGET_FILE"
    exit 0
fi

# Apply the patch: replace the old import with try/except
sed -i 's/from sae_lens\.toolkit\.pretrained_saes_directory import get_pretrained_saes_directory/try:\n    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory\nexcept ImportError:\n    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory/' "$TARGET_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully applied patch to $TARGET_FILE"
else
    echo "Error: Failed to apply patch"
    exit 1
fi


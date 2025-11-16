#!/usr/bin/env python3
"""Trace import order to understand circular imports."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Trace all imports
original_import = __builtins__.__import__
indent_level = 0

def traced_import(name, *args, **kwargs):
    global indent_level
    if 'proxai' in name:
        print('  ' * indent_level + f'→ {name}')
        indent_level += 1
    result = original_import(name, *args, **kwargs)
    if 'proxai' in name:
        indent_level -= 1
    return result

__builtins__.__import__ = traced_import

print("Starting import proxai.types...\n")
try:
    import proxai.types
    print("\n✓ Success!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

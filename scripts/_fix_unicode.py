"""Fix unicode symbols in script files for Windows CP1252 compatibility."""
import os
import glob

files = glob.glob(r'c:\Users\simi\Desktop\protium-credit-risk\scripts\*.py')
replacements = [
    ('\u2713', '[OK]'),
    ('\u2192', '->'),
    ('\u2190', '<-'),
    ('\u2014', '--'),
]
for fp in files:
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)
    if new_content != content:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed: {os.path.basename(fp)}')
print('Done')

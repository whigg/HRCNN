from pathlib import Path
from subprocess import run


for param_file in Path('.').glob('*.json'):
    print(f'Train {param_file.stem}')
    run(['python', 'train.py', str(param_file)])

from pathlib import Path
from subprocess import run


for param_file in Path('.').glob('*.json'):
    print(f'Test {param_file.stem}')
    run(['python', 'test_and_apply.py', str(param_file)])

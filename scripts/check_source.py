#!/usr/bin/env python3
"""Check syntax and published source hygiene without simulator dependencies."""
import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []
count = 0
secret = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
for p in ROOT.rglob('*'):
    if not p.is_file() or any(x in p.parts for x in ('.git', '__pycache__', '.venv', 'outputs')):
        continue
    if p.suffix == '.py':
        try:
            ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))
            count += 1
        except SyntaxError as error:
            errors.append(str(error))
    if p.suffix in {'.py', '.sh', '.md', '.yaml', '.yml', '.json', '.txt'}:
        if secret.search(p.read_text(errors='replace')):
            errors.append('Possible credential in ' + str(p.relative_to(ROOT)))
for error in errors:
    print(error, file=sys.stderr)
print(f'Parsed {count} Python files; {len(errors)} errors.')
raise SystemExit(bool(errors))

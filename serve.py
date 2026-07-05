"""Launch helper — sets cwd to okf_version/ so `import pipeline` resolves."""
import os, sys, io
from pathlib import Path

# force UTF-8 output on Windows so emoji in pipeline.py don't crash
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

HERE = Path(__file__).parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

import uvicorn
uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)

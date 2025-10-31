```bash
python -m venv .venv
source .venv/bin/activate
bash prepare.sh
bash download.sh
```

```py
# Terminal 1
cd game
python -m http.server 8000
# Terminal 2
python QWOPEnv.py
```

**MkDocs: Static Site Generator**

## Linux / MacOS

**Installation:** `conda` environment
```bash
conda install conda-forge::mkdocs-material
```
**Preview Page:** Start the live-reloading docs server.

⤷ inside root directory where`mkdoc.yml`resides:
```bash
mkdocs serve [OPTIONS]
```
Kill all processes listening to a given port. If you ran `mkdocs serve` without specifying a port:
```bash
(PolyMine) user@archlinux [~/PolyMine] $ sudo kill -9 $(sudo lsof -t -i:8000)
```

```bash
(PolyMine) user@archlinux [~/PolyMine] $ mkdocs build --clean
INFO    -  Cleaning site directory
INFO    -  Building documentation to directory: /home/user/PolyMine/site
INFO    -  Documentation built in 0.20 seconds
(PolyMine) user@archlinux [~/PolyMine] $ mkdocs serve
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  Documentation built in 0.19 seconds
INFO    -  [18:12:34] Serving on http://127.0.0.1:8000/
```

## Windows

Kill all processes listening to a given port.

```powershell
netstat -ano | findstr :8080
```
This might return a line like: `TCP 0.0.0.0:8080 0.0.0.0:0 LISTENING 12345`, where `12345` is the PID.

```powershell
taskkill /PID [PID] /F
```
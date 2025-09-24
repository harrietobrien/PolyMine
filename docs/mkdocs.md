**MkDocs: Static Site Generator**


```bash
conda install conda-forge::mkdocs-material
```

```bash
mkdocs serve [OPTIONS]
```

```bash
(PolyMine) ✿ harrie@archlinux [~/6840DM/PolyMine] $ lsof -i :8000
COMMAND     PID   USER FD   TYPE DEVICE SIZE/OFF NODE NAME
vivaldi-b 16217 harrie 24u  IPv4 700559      0t0  TCP localhost:60724->localhost:irdmi (ESTABLISHED)
mkdocs    16798 harrie  3u  IPv4 360275      0t0  TCP localhost:irdmi (LISTEN)
```
```bash
(PolyMine) ✿ harrie@archlinux [~/6840DM/PolyMine] $ kill -9 16217
(PolyMine) ✿ harrie@archlinux [~/6840DM/PolyMine] $ kill -9 16798
[1]+  Killed                     mkdocs serve
```

```bash
(PolyMine) ✿ harrie@archlinux [~/6840DM/PolyMine] $ mkdocs build --clean
INFO    -  Cleaning site directory
INFO    -  Building documentation to directory: /home/harrie/6840DM/PolyMine/site
INFO    -  Documentation built in 0.20 seconds
(PolyMine) ✿ harrie@archlinux [~/6840DM/PolyMine] $ mkdocs serve
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  Documentation built in 0.19 seconds
INFO    -  [18:12:34] Serving on http://127.0.0.1:8000/
```
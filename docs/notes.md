**Notes**

[HPC Application](https://eastcarolinauniversity.formstack.com/workflows/high_performance_computing_application?sso=6516dc5160386)

![COD SVG](img/cod.svg){ .cod-logo }

**Crystallography Open Database ([COD](https://www.crystallography.net/cod/))**

- [COD wiki](https://wiki.crystallography.net)
- [How to obtain all COD data](https://wiki.crystallography.net/howtoobtaincod/)

`rsync` address of the repository: `rsync://www.crystallography.net/cif`

```bash
# Crystallographic Information File (.cif)
rsync -av --delete rsync://www.crystallography.net/cif/ cif/
# Crystallographic reflection data - Miller indices (h, k, l)
rsync -av --delete rsync://www.crystallography.net/hkl/ hkl/
```
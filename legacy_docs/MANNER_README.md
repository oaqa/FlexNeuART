**This notebook is kept for historical purposes mainly: It uses an older framework version** 

The installation procedure is covered in a [separate notebook](INSTALL_LEGACY.md).

One should use the following mini-release:
```
git checkout tags/repr2020-12-06
```

This example covers a Manner subset of the Yahoo Answers Comprehensive.
However, a similar procedure can be applied to a bigger collection. All
experiments assume the variable `COLLECT_ROOT` in the script `scripts/config.sh` 
is set to `collections` and that all collections are stored in the `collections`
sub-directory (relative to the source code root)

A detailed step-by-step process is shown in [this notebook](manner.ipynb).

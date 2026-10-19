# sprotmetrics

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/sprotmetrics/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/sprotmetrics/badge/

Quality metrics for imaging-based spatial proteomics: segmentation, signal-to-background, normalization, marker distribution and technical-covariate diagnostics.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install sprotmetrics:

<!--
1. Add the latest release of `sprotmetrics` from [PyPI][] to your `uv` project:

   ```bash
   uv add sprotmetrics
   ```

1. Install the latest release into a [standard virtual environment][venv]:

   ```bash
   (after activating your venv)
   pip install sprotmetrics
   ```

-->

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/theislab/sprotmetrics.git  # (or `uv add`)
   ```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/sprotmetrics/issues
[tests]: https://github.com/theislab/sprotmetrics/actions/workflows/test.yaml
[documentation]: https://sprotmetrics.readthedocs.io
[changelog]: https://sprotmetrics.readthedocs.io/page/changelog.html
[api documentation]: https://sprotmetrics.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/sprotmetrics
[venv]: https://docs.python.org/3/tutorial/venv.html

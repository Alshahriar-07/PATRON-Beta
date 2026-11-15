# Third-party notices

PATRON source code is Apache-2.0. It depends on the packages below; each remains
under its own license. Versions are those used to build and test the
`1.0.0-beta.1` release. License identifiers were read from the installed package
metadata, not guessed.

## Runtime dependencies

| Package | Version | License | Project |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause (bundle also declares 0BSD, MIT, Zlib, CC0-1.0 components) | https://numpy.org |
| pandas | 3.1.1 | BSD-3-Clause | https://pandas.pydata.org |
| pyarrow | 26.0.1 | Apache-2.0 | https://arrow.apache.org |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| torch | 2.4.1+cpu | BSD-3-Clause | https://pytorch.org |
| PyYAML | 6.0.3 | MIT | https://pyyaml.org |
| pydantic | 2.14.3 | MIT | https://docs.pydantic.dev |
| lightgbm | 4.7.0 | MIT | https://lightgbm.readthedocs.io |
| matplotlib | 3.11.2 | PSF-based BSD-style license; bundles fonts under OFL-1.1 and other permissive licenses | https://matplotlib.org |
| pytest | 9.2.0 | MIT | https://pytest.org |
| tqdm | 4.70.1 | MPL-2.0 and MIT | https://tqdm.github.io |
| requests | 2.34.2 | Apache-2.0 | https://requests.readthedocs.io |
| jsonschema | 4.26.0 | MIT | https://github.com/python-jsonschema/jsonschema |

Transitive dependencies (for example `python-dateutil` Apache-2.0/BSD,
`pytz` MIT, `joblib` BSD-3-Clause, `threadpoolctl` BSD-3-Clause,
`certifi` MPL-2.0, `charset-normalizer` MIT, `urllib3` MIT, `idna` BSD-3-Clause,
`tzdata`, `narwhals`, `referencing`, `rpds-py`) keep their respective licenses.
Their installed metadata is the authoritative source; `pip-licenses` or
`importlib.metadata` will enumerate the exact set for your environment.

## Documentation and specification

The Contributor Covenant Code of Conduct v2.1 is licensed CC BY 4.0 and is
attributed in `CODE_OF_CONDUCT.md`.

The Apache License 2.0 text in `LICENSE` is the official, unmodified text from
<https://www.apache.org/licenses/LICENSE-2.0.txt>.

## Data

No market data is distributed with this repository or its release archives.
Binance public data is fetched at build time by the user and remains subject to
Binance's terms of use. See `DATASET_CARD.md`.

## Model weights

No pretrained weights are distributed. Checkpoints produced by this code are
derived works of the user's training run and may be distributed under Apache-2.0
together with this notice.

## Trademarks

"Binance", "PyTorch", "Python" and other names are trademarks of their respective
owners. PATRON is not affiliated with or endorsed by any of them.

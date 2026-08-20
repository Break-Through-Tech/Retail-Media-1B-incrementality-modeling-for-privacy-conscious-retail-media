# Project Data Guide

This folder contains a student-ready sample of the public Criteo Uplift Prediction Dataset,
unbiased version 2.1. The sample was prepared by Challenge Advisor Annamalai Annamalai for the
Break Through Tech AI Studio Fall 2026 project on privacy-conscious retail media incrementality
measurement.

Students can begin with the prepared files. Repeating the full source-data preparation is not
required for the core project.

## Files

| File | Purpose | Rows | Compressed size |
|---|---|---:|---:|
| `train.csv.gz` | Train models | 700,000 | 17.3 MB |
| `validation.csv.gz` | Compare approaches and make modeling decisions | 150,000 | 3.7 MB |
| `test.csv.gz` | Perform the final evaluation | 150,000 | 3.7 MB |
| `data_dictionary.md` | Explain every column and its permitted use | n/a | n/a |
| `sampling_method.md` | Document how the sample and splits were created | n/a | n/a |
| `validation_summary.md` | Summarize quality and representativeness checks | n/a | n/a |
| `prepare_criteo_sample.py` | Reproduce the sample from the public source | n/a | n/a |
| `checksums.sha256` | Verify the three prepared data files | n/a | n/a |
| `DATA_LICENSE.md` | Record source attribution, changes, and license terms | n/a | n/a |

## Quick Start In Google Colab

Start from a fresh Google Colab notebook by cloning the project repository. The files remain
compressed and pandas can read them directly.

```python
!git clone https://github.com/Break-Through-Tech/TBD_Retail-Media-1A-incrementality-modeling-for-privacy-conscious-retail-media.git
%cd TBD_Retail-Media-1A-incrementality-modeling-for-privacy-conscious-retail-media

import pandas as pd

train = pd.read_csv("data/train.csv.gz")
validation = pd.read_csv("data/validation.csv.gz")
test = pd.read_csv("data/test.csv.gz")

print(train.shape)
print(train.columns.tolist())
print(train[["treatment", "visit", "conversion", "exposure"]].mean())
```

Use only `f0` through `f11` as audience model features. The following columns have special roles:

- `treatment` identifies treatment and control assignment
- `visit` is the primary outcome
- `conversion` is an optional secondary outcome
- `exposure` is a post-assignment diagnostic field and must not be used as a normal audience feature

The primary analysis estimates the intention-to-treat effect of assignment on `visit`. Do not
filter the primary analysis to `exposure = 1`. The public files do not include the internal source
row numbers used during preparation and validation.

Read [`data_dictionary.md`](data_dictionary.md) before modeling.

## Source

- Original creator: Criteo AI Lab
- Dataset: Criteo Uplift Prediction Dataset, unbiased version 2.1
- [Criteo AI Lab documentation](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [Dataset download on Hugging Face](https://huggingface.co/datasets/criteo/criteo-uplift)
- [Exact source file](https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz)
- Original source rows: 13,979,592
- Original file size: approximately 311 MB compressed and 3.25 GB uncompressed
- Original compressed SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`

The source dataset page identifies the license as
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See
[`DATA_LICENSE.md`](DATA_LICENSE.md) for attribution and change details.

## Why A Prepared Sample Is Provided

The source file is ordered. The first control record does not appear until source row 5,571,800.
Taking the first million rows would therefore create a treatment-only sample and invalidate uplift
analysis. The provided script scanned the full source and gave every source record the same chance
of selection.

The resulting one-million-row package is small enough for reliable work in a free notebook
environment while closely preserving feature means and key treatment and outcome rates.

## Modeling Essentials

- Train the conventional response baseline on treatment records only
- Calculate transformed outcomes with `Z = visit * (treatment - p) / (p * (1 - p))`
- Calculate `p` from the training data and do not assume equal treatment and control allocation
- Evaluate every ranking using treatment visit rate minus control visit rate within each group
- Break tied model scores deterministically before creating ten approximately equal groups
- Report treatment and control counts, visit rates, uplift, and a 95 percent confidence interval
- Treat close model results as inconclusive when the evidence does not support a reliable winner

The fictional SolePeak and MarketBridge scenario is a teaching interpretation. The Criteo data
came from multiple online advertising experiments and is not documented as retail media or data
clean room output.

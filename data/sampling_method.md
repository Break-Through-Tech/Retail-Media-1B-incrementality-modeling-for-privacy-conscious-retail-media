# Reproducible Sampling Method

## Goal

Create a fixed, student-ready sample that preserves treatment, control, visit, and conversion
records while remaining practical in Google Colab or another free notebook environment.

## Source

- Dataset: Criteo Uplift Prediction Dataset, unbiased version 2.1
- Exact source file: https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz
- Source records: 13,979,592
- Source columns: `f0` through `f11`, `treatment`, `conversion`, `visit`, and `exposure`
- Compressed source SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- Uncompressed source SHA-256: `e4d7c710ca1f38e523309d0f8a0745d1b53e7392d51f20d1088b6cfeaef222ef`

## Method

1. Read the full source in 500,000-row portions.
2. Validate the expected columns and binary values.
3. Profile missing values, infinite values, outcome rates, and treatment-group rates.
4. Give every source record the same candidate-selection probability.
5. Use seed `2026` to select 1,030,450 candidates.
6. Randomly trim those candidates to exactly 1,000,000 records using a deterministic second seed.
7. Divide the sample using treatment, visit, and conversion strata.
8. Keep identical anonymous row patterns within the same split.
9. Use source row numbers internally to verify that records do not cross splits.
10. Remove source row numbers from the public student files.
11. Shuffle each split deterministically.
12. Save the results as reproducible gzip-compressed CSV files.

The final split contains 700,000 training records, 150,000 validation records, and 150,000 test
records. The split seed is `2028`.

## Why The Full Source Must Be Scanned

The source file is ordered. The first control record appears at source row 5,571,800. Selecting the
first million records would produce a treatment-only sample and make uplift analysis invalid. The
full-file random method avoids this ordering bias.

## Reproduce The Files

Run the following command from the repository root after downloading the original source archive:

```bash
python data/prepare_criteo_sample.py \
  --input /path/to/criteo-research-uplift-v2.1.csv.gz \
  --output-dir reproduced_data \
  --report-dir reproduced_reports
```

The script requires Python, pandas, and NumPy. It reads the compressed source directly and does not
load the entire uncompressed dataset into memory.

Before processing, the script verifies that the compressed input matches the expected source
SHA-256. Generated public reports use filenames rather than absolute local paths.

## Expected Output Checksums

| File | SHA-256 |
|---|---|
| `train.csv.gz` | `b2c9cf545e3a58f10d633f5c0f92f858f9b7590a2ddb9854a52c1b1cfd3b8aa6` |
| `validation.csv.gz` | `21bfe6fdbec21f7f308a4ba7e75d442575486118dd53320f01f59e9020035ad7` |
| `test.csv.gz` | `0d48a824580cd9a72bed6a358aa3336b7d52da4b1437f53faefb95cd166fe5d5` |

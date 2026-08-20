# Dataset License And Attribution

## Original Dataset

- Creator: Criteo AI Lab
- Title: Criteo Uplift Prediction Dataset, unbiased version 2.1
- Documentation: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Distribution page: https://huggingface.co/datasets/criteo/criteo-uplift
- Exact source file: https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz
- License identified by Criteo's current dataset page and distribution page: CC BY-NC-SA 4.0
- License text: https://creativecommons.org/licenses/by-nc-sa/4.0/

The source dataset was released with the research paper:

Eustache Diemert, Artem Betlei, Christophe Renaudin, Massih-Reza Amini, Theophane Gregoir, and
Thibaud Rahier. "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift
Modeling." https://arxiv.org/abs/2111.10106

The 2021 paper's dataset datasheet references CC BY-SA 4.0. The current Criteo dataset page and
verified Criteo Hugging Face distribution identify CC BY-NC-SA 4.0. This project follows the more
restrictive current CC BY-NC-SA 4.0 terms for the prepared educational files.

## Changes Made For This Project

The files in this folder are modified versions of the original dataset. On July 31, 2026, the
Challenge Advisor:

- Scanned all 13,979,592 source records
- Selected a reproducible random sample of 1,000,000 records using seed `2026`
- Used source row numbers internally for sampling verification and removed them from public files
- Divided the sample into training, validation, and test files
- Used treatment, visit, and conversion values to preserve rare outcome representation across splits
- Kept identical anonymous row patterns within a single split
- Saved the files as gzip-compressed CSV files

The feature values, treatment assignment, exposure indicator, and outcomes were not relabeled or
given new factual meanings.

## License For Prepared Files

The prepared dataset files are shared for this noncommercial educational project under
CC BY-NC-SA 4.0, the same license identified on the original distribution page.

When reusing or redistributing these files:

- Attribute Criteo AI Lab as the original dataset creator
- Link to the original dataset and license
- State that the data was sampled and split for this project
- Use the files only for noncommercial purposes permitted by the license
- Distribute adaptations under the same or a compatible ShareAlike license
- Do not imply that Criteo endorses this project or its conclusions

The license does not remove obligations arising from other applicable policies or laws.

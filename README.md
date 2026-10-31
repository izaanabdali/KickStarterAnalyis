# Kickstarter campaign success

Compares logistic regression, random forest and gradient boosting on campaign metadata. A separate text pipeline uses TF-IDF with logistic regression.

## Files

Start with `01_data_cleaning.ipynb, kubera.ipynb`. Existing notebooks retain the original project work.

## Running the work

Use Python with pandas, NumPy, scikit-learn, Matplotlib, Jupyter and openpyxl for the data notebooks; some also use seaborn or odfpy. Open notebooks from their repository root so relative data paths resolve. Check all required input files before execution. Dependencies are not yet version-pinned; results may vary across environments. For the Manim presentation, use its original rendering workflow instead.

## Results and limitations

Original results use an 80/20 holdout, not cross-validation. Category and country success-rate features were calculated before splitting and leak target information. Do not treat historical scores as leakage-free estimates. TF-IDF is not a neural embedding.

## Contributions

For group coursework, the repository contains team work. Individual responsibilities are not documented here; do not infer sole authorship.

## Corrected metadata baseline

Run `python train_corrected.py --data ks-projects-201801.csv --output corrected_results.json` from the repository root. Requires pandas, NumPy and scikit-learn 1.2 or later. The raw CSV is already in the original repository. This new workflow excludes target-derived category/country rates and post-outcome funding fields, fits preprocessing only on training data, and uses a stratified 80/20 split with seed 42. It is a metadata-only baseline, not a replacement for the separate TF-IDF experiment. Cancelled/live campaigns are excluded; comparisons with old scores therefore are not direct.

The corrected workflow completed on 331675 terminal-outcome campaigns. See corrected_results.json for freshly computed metrics, evaluation settings and the majority-class baseline. These results were produced during the CV review and are not the original project results.

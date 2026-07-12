# PBMI: Sex analysis of lung-cancer ML/AI datasets

Data and figure-generation code for a systematic review analysing how sex
is distributed and reported across the datasets used in lung-cancer
machine-learning and artificial intelligence studies (n = 34).

## Repository contents

| File | Description |
|------|-------------|
| `analyse_schema_final.csv` | Data extraction table — one row per included study |
| `generate_figures.py` | Reads the CSV and renders all figures |
| `requirements.txt` | Pinned Python dependencies |

## The data (`analyse_schema_final.csv`)

One row per included lung-cancer ML study, with the sex distribution (counts and
percentages) for the total cohort and each data split, plus flags for how sex was
used in the model.

| Column (CSV header) | Meaning |
|---------------------|---------|
| `paper_id` | DOI of the study |
| `cancer_type` | `L` = lung |
| `task` | `D` = diagnosis, `P` = prognosis |
| `total_chort_absolut_number` | total cohort size (N) |
| `total_cohort_absolut_number_male_female` | total cohort counts, `"male;female"` |
| `total_cohort_gender_distribution_percent_male_female` | total cohort percentages, `"male;female"` |
| `training_data_gender_absolute_number_male_female` | training-set counts, `"male;female"` |
| `training_data_gender_distribution_percent_male_female` | training-set percentages, `"male;female"` |
| `validation_data_gender_distribution_absolute_number` | validation-set counts, `"male;female"` |
| `validation_data_gender_distribution_percent_male_female` | validation-set percentages, `"male;female"` |
| `test_data_gender_distribution_absolute_number` | test-set counts, `"male;female"` |
| `test_data_gender_distribution_percent_male_female` | test-set percentages, `"male;female"` |
| `external_test_data_gender_distribution_absolute_number` | external-test counts, `"male;female"` |
| `external_test_data_gender_distribution_percent_male_female` | external-test percentages, `"male;female"` |
| `sex_specific_trained_model` | `1` = separate male/female models trained |
| `sex_in_model_development` | `1` = sex used in model development |
| `sex_in_model_evaluation` | `1` = sex used in model evaluation |
| `notes` | free-text extraction notes |

> **Formatting note:** the CSV uses European conventions — `;` as the field
> separator and a decimal comma — so it is read with `encoding="cp1252"`. Cells
> like `"108;77"` hold a male/female pair. (Column spellings such as
> `total_chort_absolut_number` are kept exactly as in the source file.)
>
> The analysis script reads the file positionally (`header=None`) and renames
> these columns to shorter internal names (e.g. `total_abs_mf`, `total_pct_mf`);
> see the `COLUMNS` list in `generate_figures.py`.

## Requirements

- Python 3.12+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python generate_figures.py
```

This renders each figure as PNG, SVG and PDF into:

- `figures/` — with on-figure titles
- `figures/no_title/` — the same figures without titles (for slides / captions)

It also prints data-quality ("reconciliation") warnings for any study whose
total-cohort counts or percentages do not add up.

## Figures

| File | Shows |
|------|-------|
| `fig1_per_study_split` | Sex split per study (total cohort), sorted by % male |
| `fig3_sex_handling` | How many studies use sex in development / evaluation / sex-specific models |
| `fig5_deviation_boxplot` | % female per data split, with one study traced across splits |
| `fig6_reporting_by_split` | How many studies report sex numbers for each split |
| `fig7_balance_by_split` | % female per data split (plain boxplot) |


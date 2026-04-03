"""Helper script with instructions for downloading NLSY data.

NLSY data must be downloaded manually from the NLS Investigator tool:
https://www.nlsinfo.org/investigator

Steps:
1. Create a free account at the NLS Investigator.
2. Select the cohort (NLSY79 or NLSY97).
3. Search for and add the variable groups listed below.
4. Go to Save/Download → Advanced Download.
5. Select "Comma-delimited datafile".
6. Download and extract the zip file.
7. Place the CSV file(s) in data/raw/

The pipeline expects files named:
  - data/raw/nlsy79.csv
  - data/raw/nlsy97.csv

Until real data is available, use --synthetic flag with all scripts:
  make preprocess-synthetic
  make train-synthetic
  make archetypes-synthetic

Required NLSY79 Variables (search by reference number in NLS Investigator):
────────────────────────────────────────────────────────────────────────────
  Category         | Search Term / Reference         | Notes
  ─────────────────┼─────────────────────────────────┼──────────────────────
  Identification   | CASEID (R0000100)               | Unique person ID
  Sex              | SEX (R0214800)                  | 1=Male, 2=Female
  Race/Ethnicity   | SAMPLE_RACE (R0214700)          | 1=Hisp, 2=Black, 3=Other
  Birth Year       | YEAR OF BIRTH (R0000500)        | 4-digit year
  Region at 14     | REGION (R0000300)               | 1=NE, 2=NC, 3=S, 4=W
  Parent Education | HGC_MOTHER, HGC_FATHER          | Highest grade completed
  Family Income    | TNFI_TRUNC (various rounds)     | Total net family income
  Employment       | EMP_STATUS (various rounds)     | Employment status per round
  Income           | TOTAL_INCOME / HOURLY_RATE      | Per round
  Job Satisfaction | JOB_SATISFACTION                | Per round (1-4 scale)
  AFQT Score       | AFQT_1 (R0618300)               | Armed Forces percentile
  Education        | HIGHEST_GRADE_COMPLETED         | Per round

Required NLSY97 Variables:
────────────────────────────────────────────────────────────────────────────
  Category         | Search Term / Reference         | Notes
  ─────────────────┼─────────────────────────────────┼──────────────────────
  Identification   | PUBID (R0000100)                | Unique person ID
  Sex              | KEY_SEX (R0536300)              | 1=Male, 2=Female
  Race/Ethnicity   | KEY_RACE_ETHNICITY (R0536401)   | 1=Black, 2=Hisp, ...
  Birth Year       | KEY_BDATE_Y (R0536402)          | 4-digit year
  Region at 14     | CENSUS REGION (various)         | 1=NE, 2=NC, 3=S, 4=W
  Parent Education | CV_HGC_BIO_MOM, CV_HGC_BIO_DAD | Bio parent education
  Household Income | CV_INCOME_FAMILY (various)      | Total family income
  Employment       | EMP_STATUS (various rounds)     | Employment status
  Income           | YINC-1700 / CV_HRLY_PAY         | Per round
  Job Satisfaction | JOB_SATISFACTION                | Per round
  ASVAB Score      | ASVAB_MATH_VERBAL_SCORE_PCT     | Composite percentile
  Education        | CV_HIGHEST_DEGREE_EVER          | Per round

Sentinel values: The pipeline automatically replaces -1 through -5 with NaN.
  -1 = Refused, -2 = Don't know, -3 = Invalid skip,
  -4 = Valid skip, -5 = Non-interview
"""

from __future__ import annotations


def main() -> None:
    """Print download instructions."""
    print(__doc__)


if __name__ == "__main__":
    main()

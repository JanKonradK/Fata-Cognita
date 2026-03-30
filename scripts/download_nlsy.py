"""Helper script with instructions for downloading NLSY data.

NLSY data must be downloaded manually from the NLS Investigator tool:
https://www.nlsinfo.org/investigator

Steps:
1. Create a free account at the NLS Investigator.
2. Select the cohort (NLSY79 or NLSY97).
3. Search for and add the following variable groups:
   - Employment: employment status, hours worked, class of worker
   - Income: total income, family income
   - Education: highest grade completed, enrollment status
   - Health: SF-12, CES-D, health limitations
   - Demographics: sex, race/ethnicity, birth year, region
   - Family: marital status, household composition
   - ASVAB/AFQT scores (NLSY79)
   - Job satisfaction measures
4. Go to Save/Download → Advanced Download.
5. Select "Comma-delimited datafile".
6. Download and extract the zip file.
7. Place the CSV file(s) in data/raw/

The pipeline expects files named:
  - data/raw/nlsy79.csv
  - data/raw/nlsy97.csv

Until real data is available, use --synthetic flag with all scripts.
"""

from __future__ import annotations


def main() -> None:
    """Print download instructions."""
    print(__doc__)


if __name__ == "__main__":
    main()

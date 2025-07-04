from pathlib import Path

# Based on the dataset description at , we can identify the features and their
# types. The dataset contains both categorical and numerical features, as well
# as binary features. We will define these features accordingly.
# https://archive.ics.uci.edu/dataset/222/bank+marketing
CATEGORICAL_FEATURES = [
    "contact",
    "day_of_week",
    "default",
    "education",
    "housing",
    "job",
    "loan",
    "marital",
    "month",
    "poutcome",
    "year",
]
NUMERICAL_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
BINARY_FEATURES = [
    "y",
]

DATA_DIR = Path("../data")
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATA_FILENAME = "bank-additional-full.csv"
APPROACHED_DATA_FILENAME = "approached_data.csv"
NOT_APPROACHED_DATA_FILENAME = "not_approached_data.csv"

HONOLULU_BLUE = "#1F77B4"
IMPERIAL_RED = "#F0534F"
PERSIAN_GREEN = "#27A69A"

SELECTED_YEAR = 2010

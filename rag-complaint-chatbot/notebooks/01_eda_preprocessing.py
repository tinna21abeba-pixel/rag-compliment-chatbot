# %% [markdown]
# # Task 1: Exploratory Data Analysis and Data Preprocessing
#
# **CrediTrust Financial — Intelligent Complaint Analysis**
#
# Objective: Understand the structure, content, and quality of the CFPB
# complaint data, then filter and clean it so it's ready for chunking and
# embedding in Task 2.
#
# **Input:**  `data/raw/complaints.csv` (full CFPB dataset — place your
# downloaded file here and update the filename below if different)
#
# **Output:** `data/filtered_complaints.csv`

# %%
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from text_cleaning import clean_narrative, word_count

sns.set_style("whitegrid")
pd.set_option("display.max_colwidth", 120)

RAW_DATA_PATH = "../data/raw/complaints.csv"
OUTPUT_PATH = "../data/filtered_complaints.csv"

TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer",
]

# %% [markdown]
# ## 1. Load the full CFPB dataset
#
# The official CFPB export uses these column names (rename map below handles
# common variants — adjust if your downloaded file differs).

# %%
df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
print(f"Raw shape: {df.shape}")
df.head(3)

# %%
# Standardize column names we rely on throughout this notebook.
# CFPB's public export typically uses these exact headers:
#   'Product', 'Sub-product', 'Issue', 'Sub-issue',
#   'Consumer complaint narrative', 'Company', 'State', 'Date received'
COLUMN_MAP = {
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "consumer_complaint_narrative",
    "Company": "company",
    "State": "state",
    "Date received": "date_received",
    "Complaint ID": "complaint_id",
}
df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

missing_expected = [
    v for v in COLUMN_MAP.values() if v not in df.columns
]
if missing_expected:
    print("WARNING - expected columns not found, check your CSV headers:")
    print(missing_expected)

df.columns.tolist()

# %% [markdown]
# ## 2. Initial EDA
#
# ### 2.1 Distribution of complaints across products

# %%
product_counts = df["product"].value_counts()
print(product_counts)

plt.figure(figsize=(10, 6))
product_counts.head(15).plot(kind="barh")
plt.title("Top 15 Products by Complaint Volume (Full Dataset)")
plt.xlabel("Number of Complaints")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../data/processed/product_distribution_full.png", dpi=120)
plt.show()

# %% [markdown]
# ### 2.2 Narrative presence: complaints with vs. without text

# %%
has_narrative = df["consumer_complaint_narrative"].notna() & (
    df["consumer_complaint_narrative"].str.strip() != ""
)
print(f"Complaints WITH narrative:    {has_narrative.sum():,}")
print(f"Complaints WITHOUT narrative: {(~has_narrative).sum():,}")
print(f"Percent with narrative:       {has_narrative.mean():.1%}")

plt.figure(figsize=(5, 5))
plt.pie(
    [has_narrative.sum(), (~has_narrative).sum()],
    labels=["Has narrative", "No narrative"],
    autopct="%1.1f%%",
    colors=["#4C72B0", "#DD8452"],
)
plt.title("Narrative Availability")
plt.tight_layout()
plt.savefig("../data/processed/narrative_availability.png", dpi=120)
plt.show()

# %% [markdown]
# ### 2.3 Narrative length distribution (word count)
#
# Computed only over complaints that *have* a narrative.

# %%
df_with_text = df[has_narrative].copy()
df_with_text["word_count"] = df_with_text["consumer_complaint_narrative"].apply(word_count)

print(df_with_text["word_count"].describe())

plt.figure(figsize=(10, 5))
sns.histplot(df_with_text["word_count"], bins=80, kde=False)
plt.xlim(0, df_with_text["word_count"].quantile(0.99))  # trim extreme outliers for readability
plt.title("Distribution of Narrative Word Count (up to 99th percentile)")
plt.xlabel("Word Count")
plt.ylabel("Number of Complaints")
plt.tight_layout()
plt.savefig("../data/processed/narrative_length_distribution.png", dpi=120)
plt.show()

# %%
# Flag very short / very long narratives for awareness
very_short = (df_with_text["word_count"] < 5).sum()
very_long = (df_with_text["word_count"] > 1000).sum()
print(f"Narratives under 5 words:     {very_short:,}")
print(f"Narratives over 1000 words:  {very_long:,}")

# %% [markdown]
# ## 3. Filter the dataset
#
# - Keep only the four target products (Credit Card, Personal Loan,
#   Savings Account, Money Transfer).
# - Drop rows with empty narratives.
#
# **Note:** CFPB's `product` field uses various labels across years
# (e.g. "Credit card or prepaid card", "Money transfer, virtual currency,
# or money service"). The mapping below is intentionally permissive —
# inspect `product_counts` above and adjust `PRODUCT_KEYWORDS` if your
# export uses different category strings.

# %%
PRODUCT_KEYWORDS = {
    "Credit Card": ["credit card"],
    "Personal Loan": ["personal loan", "consumer loan", "payday loan"],
    "Savings Account": ["savings account", "checking or savings"],
    "Money Transfer": ["money transfer", "money service", "virtual currency"],
}


def map_product(raw_product: str) -> str:
    """Map a raw CFPB product string to one of our four target categories,
    or None if it doesn't belong to any of them."""
    if not isinstance(raw_product, str):
        return None
    raw_lower = raw_product.lower()
    for target, keywords in PRODUCT_KEYWORDS.items():
        if any(kw in raw_lower for kw in keywords):
            return target
    return None


df["product_category"] = df["product"].apply(map_product)

filtered = df[
    df["product_category"].notna()
    & df["consumer_complaint_narrative"].notna()
    & (df["consumer_complaint_narrative"].str.strip() != "")
].copy()

print(f"Rows after product + narrative filtering: {len(filtered):,}")
print(filtered["product_category"].value_counts())

# %% [markdown]
# ## 4. Clean the text narratives
#
# Cleaning steps (implemented in `src/text_cleaning.py` so they are unit
# tested and reusable in Task 2):
# 1. Lowercase
# 2. Remove CFPB redaction placeholders (e.g. `XXXX`, `XX/XX/XXXX`)
# 3. Remove boilerplate complaint-opening phrases
# 4. Remove special characters
# 5. Collapse whitespace

# %%
filtered["cleaned_narrative"] = filtered["consumer_complaint_narrative"].apply(clean_narrative)
filtered["cleaned_word_count"] = filtered["cleaned_narrative"].apply(word_count)

# Drop any rows that became empty after cleaning (rare, but possible if a
# narrative was pure boilerplate/redaction).
before = len(filtered)
filtered = filtered[filtered["cleaned_word_count"] > 0]
print(f"Dropped {before - len(filtered)} rows that were empty after cleaning.")

filtered[["product_category", "consumer_complaint_narrative", "cleaned_narrative"]].head(3)

# %% [markdown]
# ## 5. Save the cleaned and filtered dataset

# %%
output_columns = [
    "complaint_id",
    "product_category",
    "product",
    "issue",
    "sub_issue",
    "company",
    "state",
    "date_received",
    "consumer_complaint_narrative",
    "cleaned_narrative",
    "cleaned_word_count",
]
output_columns = [c for c in output_columns if c in filtered.columns]

filtered[output_columns].to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(filtered):,} cleaned complaints to {OUTPUT_PATH}")

# %% [markdown]
# ## 6. Summary of Key EDA Findings
#
# *(Fill in the actual numbers after running this notebook on your data —
# placeholders below show the structure expected in the report.)*
#
# 1. **Product distribution:** Describe which products receive the most
#    complaints in the full dataset and how that compares to the filtered
#    four-product subset.
# 2. **Narrative coverage:** State what percentage of complaints include a
#    free-text narrative vs. metadata only, and what this means for the
#    available training/retrieval corpus size.
# 3. **Narrative length:** Summarize the word-count distribution (median,
#    typical range, and any extreme outliers), and note the chunking
#    implications (e.g. most narratives fit in 1-2 chunks at 500 chars).

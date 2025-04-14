import pandas as pd
import ast
import re
from rank_bm25 import BM25Okapi

# Function to lowercase and split
def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())  # remove punctuation
    return text.split()

df = pd.read_csv("raw_outputs.csv")

# tokenize each summary in the dataset
all_tokenized_summaries = []

for row in df["summaries"]:
    try:
        summaries_list = ast.literal_eval(row)  # convert string to list
        first_summary = summaries_list[0] if summaries_list else ""
        tokens = tokenize(first_summary)
    except:
        tokens = []
    all_tokenized_summaries.append(tokens)

# Build the BM25 model 
bm25 = BM25Okapi(all_tokenized_summaries)

# Input your search question here
query = "Where did Barack Obama's mother go to university?"
tokenized_query = tokenize(query)

# Score each row in the dataset 
scores = bm25.get_scores(tokenized_query)

# get the top 10 highest scoring rows
top_10 = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:10]

# Display the results
print("\nTop 10 Most Relevant Rows:\n")
for idx, score in top_10:
    try:
        summary = ast.literal_eval(df.loc[idx, "summaries"])[0]
        preview = summary[:200]  # show only the first 200 characters
    except:
        preview = "[summary could not be loaded]"

    print(f"Row {idx} — Score: {score:.2f}")
    print("Summary preview:", preview, "...\n")

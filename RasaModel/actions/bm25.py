import sys
import pandas as pd
import ast
import re
import numpy as np
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi


# tokenise and lowercase, remove punctuation
def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


# Load dataset
df = pd.read_csv("raw_outputs.csv")


all_summaries = []
tokenised_summaries = []
original_row_indices = []

# tokenise each summary
for idx, row in df.iterrows():
    try:
        summaries = ast.literal_eval(row["summaries"])
        for summary in summaries:
            if isinstance(summary, str):
                all_summaries.append(summary)
                tokenised_summaries.append(tokenize(summary))
                original_row_indices.append(idx)
    except:
        continue

# Load BM25 and embedding model
bm25 = BM25Okapi(tokenised_summaries)
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_summary(query, intent, bm25_top_n=5, confidence_threshold=0.40):
    # BM25 search
    tokenised_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenised_query)

    # Get the number of available summaries
    available_summaries_count = len(all_summaries)

    # Ensure bm25_top_n does not exceed the number of available summaries
    top_n = min(bm25_top_n, available_summaries_count)

    # Get indices of top n BM25 scores
    top_indices = np.argsort(bm25_scores)[-top_n:][::-1]
    top_summaries = [all_summaries[i] for i in top_indices]

    # Log the top results with their scores
    os.makedirs("logs", exist_ok=True)
    with open("logs/bm25_rankings.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- Query: {query} | Intent: {intent} | {datetime.now()} ---\n")
        for i, idx in enumerate(top_indices):
            bm25_score = bm25_scores[idx]
            f.write(f"[BM25: {bm25_score:.4f}] {all_summaries[idx]}\n")

    # Confidence check (optional; can adjust/remove as needed)
    if top_n > 0:  # Only proceed if we have top results
        top_score = bm25_scores[top_indices[0]]
        if top_score < confidence_threshold:
            return "not confident"

    # Return the highest ranked summary
    return top_summaries[0]  # Return the top result with the highest BM25 score




# Command-line
def main():
    query = " ".join(sys.argv[1:])
    result = retrieve_summary(query)
    print(result)


if __name__ == "__main__":
    main()
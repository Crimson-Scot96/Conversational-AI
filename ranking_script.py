import sys
import pandas as pd
import ast
import re
import numpy as np
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

#tokenise each summary
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

# retrieval function
def retrieve_summary(query, bm25_top_n=5, confidence_threshold=0.40):
    # BM25 search
    tokenised_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenised_query)
    
    # Get indices of top n BM25 scores
    top_indices = np.argsort(bm25_scores)[-bm25_top_n:][::-1]
    top_summaries = [all_summaries[i] for i in top_indices]

    #SentenceTransformer reranking
    query_embedding = model.encode(query, convert_to_tensor=True)
    top_embeddings = model.encode(top_summaries, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(query_embedding, top_embeddings)[0].cpu().numpy()

    # Check if top score is confident enough
    top_score = max(cosine_similarities)
    if top_score < confidence_threshold:
        return "not confident"

    best_index = np.argmax(cosine_similarities)
    return top_summaries[best_index]

# Command-line
def main():
    query = " ".join(sys.argv[1:])
    result = retrieve_summary(query)
    print(result)


if __name__ == "__main__":
    main()

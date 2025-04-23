#importing important libraries
import pandas as pd
import json
import requests
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import torch
import os

# loading data from github
def load_data():
    url = "https://raw.githubusercontent.com/Ansh0903/JSON/master/qulac_for_wiki.json"
    response = requests.get(url)
    data = json.loads(response.text)

    queries, clarifications = [], []
    for key, item in data.items():
        if len(item) > 1:
            queries.append(item[0]) # the ambigous query
            clarifications.append(item[1]) # the coresponding clarifiying question

    print(f"Extracted {len(queries)} query-clarification pairs.")
    return queries, clarifications

# training the sbert model
def train_sbert_model(queries, clarifications, model_name='all-MiniLM-L6-v2', output_path='output/fine_tuned_sbert_model'):
    df = pd.DataFrame({'User Query': queries, 'Clarifying Question': clarifications}).dropna()
    train_examples = [InputExample(texts=[q, a]) for q, a in zip(df['User Query'], df['Clarifying Question'])]

    model = SentenceTransformer(model_name)
    # data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # defining loss function for fine tuning
    train_loss = losses.MultipleNegativesRankingLoss(model)

# fine tuning the model using training data
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
    os.makedirs(output_path, exist_ok=True)

    # saving the model
    model.save(output_path)
    print(f"Model saved to {output_path}")
    return model

# testing the final model
def test_model(model):
    query = "obama family tree"
    clarifications = [
        "What aspect of the Obama family tree are you asking about?",
        "Are you asking about Michelle Obama?",
        "Would you like to know about the presidency?"
    ]
# calculating the cosine similarity
    query_emb = model.encode(query, convert_to_tensor=True)
    clarification_embs = model.encode(clarifications, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_emb, clarification_embs)

# printing input queries and the semantic similarity scores
    print(f"\n Query: {query}")
    for i, score in enumerate(similarities[0]):
        print(f"Clarification {i+1}: {clarifications[i]} (score: {score.item():.4f})")

if __name__ == "__main__":
    queries, clarifications = load_data()
    model = train_sbert_model(queries, clarifications)
    test_model(model)

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import tkinter as tk

# Load data and create service names
data = pd.read_excel('dataset.xlsx')
texts = data['Topic'].tolist()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = sbert_model.encode(texts, convert_to_tensor=True)
service_names = list(data['label'].astype('category').cat.categories)

# Load the trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("trained_model")
model = DistilBertForSequenceClassification.from_pretrained("trained_model")

# Function to analyze the problem
def analyze_problem(problem_text):
    # Tokenize and classify
    inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    outputs = model(**inputs)
    softmaxed = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_probs, top_indices = torch.topk(softmaxed, k=3)  # Top 3 predictions with probabilities
    
    # Retrieve top service names and probabilities
    try:
        top_services = [(service_names[idx], prob.item() * 100) for idx, prob in zip(top_indices[0], top_probs[0])]
    except IndexError:
        result_var.set("Error: Unable to retrieve service names from predictions.")
        return

    # Find the most similar problem
    query_embedding = sbert_model.encode(problem_text, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(query_embedding, text_embeddings)
    best_match_idx = torch.argmax(cos_sim).item()
    similar_problem = texts[best_match_idx]
    instruction = data['Solution'].iloc[best_match_idx] if pd.notnull(data['Solution'].iloc[best_match_idx]) else "No instruction found."

    # Display results
    result_text = f"Service suggestions:\n" + "\n".join(
        [f"{service}: {prob:.2f}%" for service, prob in top_services]
    ) + f"\n\nMost similar problem: {similar_problem}\nSuggested instruction: {instruction}"
    result_var.set(result_text)

# Tkinter GUI setup
window = tk.Tk()
window.title("Problem Analyzer")

label_problem = tk.Label(window, text="Enter your problem description:")
label_problem.pack()
entry_problem = tk.Entry(window, width=50)
entry_problem.pack()

button_analyze = tk.Button(window, text="Analyze", command=lambda: analyze_problem(entry_problem.get()))
button_analyze.pack()

label_result = tk.Label(window, text="Result:")
label_result.pack()
result_var = tk.StringVar()
entry_result = tk.Entry(window, textvariable=result_var, width=50)
entry_result.pack()

window.mainloop()

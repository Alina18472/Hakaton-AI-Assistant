import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import tkinter as tk
from tkinter import Toplevel, Text
import os
import docx  # Library to handle Word documents

# Load data and create service names
data = pd.read_excel('dataset.xlsx')
texts = data['Topic'].tolist()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = sbert_model.encode(texts, convert_to_tensor=True)
service_names = list(data['label'].astype('category').cat.categories)

# Load the trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("enhanced_trained_model")
model = DistilBertForSequenceClassification.from_pretrained("enhanced_trained_model")

# Instructions from dataset
instructions = data['Solution'].dropna().tolist()
instructions = [str(instruction) for instruction in instructions]
instruction_embeddings = sbert_model.encode(instructions, convert_to_tensor=True)

# Function to find and display matching Word content
def show_detailed_instruction(instruction_text):
    folder_path = 'Instructions'  # Path where Word files are stored
    best_file = None
    best_score = -1

    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            doc_path = os.path.join(folder_path, filename)
            doc = docx.Document(doc_path)
            full_text = ' '.join([p.text for p in doc.paragraphs])

            # Calculate similarity with the instruction text
            doc_embedding = sbert_model.encode(full_text, convert_to_tensor=True)
            instruction_embedding = sbert_model.encode(instruction_text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(instruction_embedding, doc_embedding).item()
            

            if score > best_score:
                best_score = score
                best_file = full_text

    if best_file:
        # Open a new window to display the content
        top = Toplevel(window)
        top.title("Detailed Instruction")
        text_widget = Text(top, wrap='word')
        text_widget.insert('1.0', best_file)
        text_widget.pack(expand=True, fill='both')

# Function to analyze the problem and display results
def analyze_problem(problem_text):
    inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    outputs = model(**inputs)
    softmaxed = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_probs, top_indices = torch.topk(softmaxed, k=3)

    top_services = [(service_names[idx], prob.item() * 100) for idx, prob in zip(top_indices[0], top_probs[0])]

    # Find the most similar problem and instruction
    query_embedding = sbert_model.encode(problem_text, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(query_embedding, text_embeddings)
    best_match_idx = torch.argmax(cos_sim).item()
    similar_problem = texts[best_match_idx]

    instruction_cos_sim = util.pytorch_cos_sim(query_embedding, instruction_embeddings)
    best_instruction_idx = torch.argmax(instruction_cos_sim).item()
    instruction = instructions[best_instruction_idx] if pd.notnull(instructions[best_instruction_idx]) else "No instruction found."

    clear_results()
    tk.Label(result_frame, text="Service suggestions:", font=("Arial", 14)).pack()
    for service, prob in top_services:
        tk.Label(result_frame, text=f"{service}: {prob:.2f}%", font=("Arial", 12)).pack()

    tk.Label(result_frame, text=f"\nMost similar problem:", font=("Arial", 14)).pack()
    tk.Label(result_frame, text=f" {similar_problem}", font=("Arial", 12)).pack()

    tk.Label(result_frame, text=f"\nSuggested instruction:", font=("Arial", 14)).pack()
    tk.Label(result_frame, text=f"{instruction}", font=("Arial", 12)).pack()

    # Button to view detailed instruction
    detailed_button = tk.Button(result_frame, text="View Detailed Instruction", command=lambda: show_detailed_instruction(instruction))
    detailed_button.pack()

def clear_results():
    for widget in result_frame.winfo_children():
        widget.destroy()

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

result_frame = tk.Frame(window)
result_frame.pack()

window.mainloop()

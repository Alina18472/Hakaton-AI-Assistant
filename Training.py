import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_excel('dataset.xlsx')
texts = data['Topic'].tolist()
services = data['label'].astype('category').cat.codes.tolist()
labels_to_services = data['label'].astype('category').cat.categories

# Load Tokenizer and Embedding Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Classify using Sentence-BERT embeddings with enhanced context
class ProblemDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        item = {key: val.flatten() for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Add additional weight to important context phrases
        item["weights"] = torch.ones(item['input_ids'].shape)  # Initialize weights as ones
        context_words = ["doesn't work", "won't start","wont start","won't starting", "not working", "broken", "dont work", "don't work","don't working"]
        for phrase in context_words:
            phrase_tokens = tokenizer(phrase, return_tensors="pt")["input_ids"][0][1:-1]  # Strip CLS and SEP tokens
            for i in range(len(item['input_ids']) - len(phrase_tokens)):
                if torch.equal(item['input_ids'][i:i+len(phrase_tokens)], phrase_tokens):
                    item["weights"][i:i+len(phrase_tokens)] = 1.5  # Higher weight for context phrases
        return item

# Train-Validation split
train_texts, val_texts, train_services, val_services = train_test_split(texts, services, test_size=0.2, random_state=42)
train_dataset = ProblemDataset(train_texts, train_services)
val_dataset = ProblemDataset(val_texts, val_services)

# Define Model with Multi-Label
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(set(services)))

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,  # Increasing epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()

# Save model
model.save_pretrained("enhanced_trained_model")
tokenizer.save_pretrained("enhanced_trained_model")

# # Analyzing Function for Contextual Predictions
# def analyze_problem(problem_text):
#     inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
#     outputs = model(**inputs)

#     # Extract probabilities and select top-k services
#     probs = torch.softmax(outputs.logits, dim=-1).flatten()
#     top_k_probs, top_k_indices = torch.topk(probs, k=3)  # Top 3 services

#     service_predictions = [
#         (labels_to_services[idx], round(prob.item() * 100, 2)) for idx, prob in zip(top_k_indices, top_k_probs)
#     ]
    
#     # Semantic similarity search for similar issues
#     query_embedding = sbert_model.encode(problem_text, convert_to_tensor=True)
#     cos_sim = util.pytorch_cos_sim(query_embedding, text_embeddings)
#     top_match_indices = torch.topk(cos_sim, k=3, dim=1).indices[0]  # Top 3 similar issues

#     similar_issues = [(texts[idx], data['Solution'].iloc[idx]) for idx in top_match_indices]
#     return service_predictions, similar_issues

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
# Загрузка данных
data = pd.read_excel('dataset.xlsx')
texts = data['Topic'].tolist()
services = data['label'].astype('category').cat.codes.tolist()

# Загрузка токенизатора и Sentence-BERT модели
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = sbert_model.encode(texts, convert_to_tensor=True)

# Класс Dataset
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
        return item

# Подготовка тренировочного и валидационного набора
train_texts, val_texts, train_services, val_services = train_test_split(texts, services, test_size=0.2, random_state=42)
train_dataset = ProblemDataset(train_texts, train_services)
val_dataset = ProblemDataset(val_texts, val_services)

# Определение модели
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(set(services)))

# Настройки обучения
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
)

# Тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Запуск обучения
trainer.train()

# # Функция для анализа проблемы
# def analyze_problem(problem_text):
#     inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
#     outputs = model(**inputs)
#     predicted_service_idx = torch.argmax(outputs.logits, dim=-1).item()
#     predicted_service = data['label'].astype('category').cat.categories[predicted_service_idx]

#     # Поиск по семантическому сходству
#     query_embedding = sbert_model.encode(problem_text, convert_to_tensor=True)
#     cos_sim = util.pytorch_cos_sim(query_embedding, text_embeddings)
#     best_match_idx = torch.argmax(cos_sim).item()
    
#     similar_problem = texts[best_match_idx]
#     instruction = data['Solution'].iloc[best_match_idx] if pd.notnull(data['Solution'].iloc[best_match_idx]) else "Инструкция отсутствует"

#     return f"{predicted_service}: {instruction} (похожая проблема: {similar_problem})"
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
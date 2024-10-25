import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import tkinter as tk

# Загрузка данных
data = pd.read_excel('dataset.xlsx')
texts = data['Topic'].tolist()
services = data['label'].astype('category').cat.codes.tolist()
has_instruction = data['Solution'].apply(lambda x: 1 if pd.notnull(x) else 0).tolist()

# Разделение на тренировочную и тестовую выборки
train_texts, val_texts, train_services, val_services, train_instructions, val_instructions = train_test_split(
    texts, services, has_instruction, test_size=0.2, random_state=42
)

# Загрузка предобученного токенизатора
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Создание кастомного класса Dataset
class ProblemDataset(Dataset):
    def __init__(self, texts, labels_service, labels_instruction):
        self.texts = texts
        self.labels_service = labels_service
        self.labels_instruction = labels_instruction

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        item = {key: val.flatten() for key, val in encoding.items()}
        item["labels_service"] = torch.tensor(self.labels_service[idx], dtype=torch.long)
        item["labels_instruction"] = torch.tensor(self.labels_instruction[idx], dtype=torch.float)
        return item

train_dataset = ProblemDataset(train_texts, train_services, train_instructions)
val_dataset = ProblemDataset(val_texts, val_services, val_instructions)

# Создание кастомной модели
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_service):
        super(MultiTaskModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.classifier_service = nn.Linear(self.bert.config.hidden_size, num_labels_service)
        self.classifier_instruction = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels_service=None, labels_instruction=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]

        service_logits = self.classifier_service(pooled_output)
        instruction_logits = self.classifier_instruction(pooled_output).squeeze()

        loss = None
        if labels_service is not None and labels_instruction is not None:
            loss_fct_service = nn.CrossEntropyLoss()
            loss_fct_instruction = nn.BCEWithLogitsLoss()
            loss_service = loss_fct_service(service_logits, labels_service)
            loss_instruction = loss_fct_instruction(instruction_logits, labels_instruction)
            loss = loss_service + loss_instruction

        return {"loss": loss, "logits": (service_logits, instruction_logits)}

# Инициализация модели
model = MultiTaskModel('distilbert-base-uncased', num_labels_service=len(set(services)))

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

# Определение тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Обучение модели
trainer.train()

# Функция для анализа текста проблемы
def analyze_problem():
    problem_text = entry_problem.get()
    inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    outputs = model(**inputs)
    service_logits, instruction_logits = outputs["logits"]
    predicted_service_idx = torch.argmax(service_logits, dim=-1).item()
    has_instruction = instruction_logits.item() > 0.5
    predicted_service = data['label'].astype('category').cat.categories[predicted_service_idx]
    result_var.set(predicted_service + (": есть инструкция" if has_instruction else ": инструкции нет"))

# Создание интерфейса tkinter
window = tk.Tk()
window.title("Анализатор проблем пользователей")

label_problem = tk.Label(window, text="Введите описание проблемы:")
label_problem.pack()
entry_problem = tk.Entry(window, width=50)
entry_problem.pack()

button_analyze = tk.Button(window, text="Анализировать", command=analyze_problem)
button_analyze.pack()

label_result = tk.Label(window, text="Результат - определенный сервис:")
label_result.pack()
result_var = tk.StringVar()
entry_result = tk.Entry(window, textvariable=result_var, width=50)
entry_result.pack()

window.mainloop()

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import tkinter as tk

# Загрузка данных для семантического поиска
data = pd.read_excel('dataset.xlsx')
texts = data['Topic'].tolist()

# Загрузка сервисов и преобразование в уникальные индексы для сопоставления
service_names = data['label'].unique().tolist()  # Создаем список уникальных названий сервисов
service_to_index = {service: idx for idx, service in enumerate(service_names)}  # Сопоставление сервиса с индексом

# Инициализируем модель для семантического поиска
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = sbert_model.encode(texts, convert_to_tensor=True)

# Загрузка сохраненной модели и токенизатора
tokenizer = DistilBertTokenizer.from_pretrained("trained_model")
model = DistilBertForSequenceClassification.from_pretrained("trained_model")

# Функция для анализа текста проблемы
def analyze_problem():
    problem_text = entry_problem.get()
    inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    outputs = model(**inputs)
    
    # Получаем вероятности для каждого класса
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top_k = 3  # Количество лучших сервисов для отображения
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
    
    # Определяем топ сервисов с их вероятностями, используя сопоставление индексов с сервисами
    top_services = [(service_names[idx], prob.item() * 100) for idx, prob in zip(top_k_indices, top_k_probs)]
    
    # Поиск по семантическому сходству
    query_embedding = sbert_model.encode(problem_text, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(query_embedding, text_embeddings)
    best_match_idx = torch.argmax(cos_sim).item()
    
    similar_problem = texts[best_match_idx]
    instruction = data['Solution'].iloc[best_match_idx] if pd.notnull(data['Solution'].iloc[best_match_idx]) else "Инструкция отсутствует"

    # Формирование ответа
    result = "Возможные сервисы:\n" + "\n".join([f"{service}: {prob:.2f}%" for service, prob in top_services])
    result += f"\n\nИнструкция: {instruction} (похожая проблема: {similar_problem})"
    result_var.set(result)

# Настройка интерфейса
window = tk.Tk()
window.title("Анализатор проблем пользователей")

label_problem = tk.Label(window, text="Введите описание проблемы:")
label_problem.pack()
entry_problem = tk.Entry(window, width=50)
entry_problem.pack()

button_analyze = tk.Button(window, text="Анализировать", command=analyze_problem)
button_analyze.pack()

label_result = tk.Label(window, text="Результат - возможные сервисы и инструкция:")
label_result.pack()
result_var = tk.StringVar()
entry_result = tk.Entry(window, textvariable=result_var, width=50)
entry_result.pack()

window.mainloop()

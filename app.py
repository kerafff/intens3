import tkinter as tk 
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import openpyxl
from statsmodels.tsa.seasonal import seasonal_decompose

# Пути к файлам
MODEL_PATH = "catboost_model.pkl"  # Проверь, что файл модели существует
DATA_PATH = "danie.csv"  # Данные для предсказания

# Загрузка обученной модели CatBoost
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    messagebox.showerror("Ошибка", "Файл модели не найден. Обучи модель и сохрани её в 'catboost_model.pkl'")
    exit()

# Функция для загрузки и предобработки данных
def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
        data['dt'] = pd.to_datetime(data['dt'])  # Преобразование столбца 'Дата' в datetime
        
        # Проверка необходимых столбцов
        required_columns = ['Цена на арматуру']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            messagebox.showerror("Ошибка", f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
            return None
        
        # Добавление тренда и сезонности
        decomposition = seasonal_decompose(data['Цена на арматуру'], model='additive', period=52, extrapolate_trend='freq')
        data['trend'] = decomposition.trend
        data['seasonal'] = decomposition.seasonal
        data.fillna(method='bfill', inplace=True)  # Заполнение пропусков
        
        return data
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл train.xlsx не найден.")
        return None
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка загрузки данных: {e}")
        return None

# Функция для предсказания
def predict_tender():
    selected_date = date_var.get()
    if not selected_date:
        messagebox.showerror("Ошибка", "Выберите дату.")
        return

    try:
        selected_data = data[data['dt'] == pd.to_datetime(selected_date)]
        if selected_data.empty:
            messagebox.showerror("Ошибка", "Данные для выбранной даты отсутствуют.")
            return

        # Подготовка фичей для модели
        feature_columns = ['Цена на арматуру', 'trend', 'seasonal']
        for col in feature_columns:
            if col not in selected_data.columns:
                messagebox.showerror("Ошибка", f"Отсутствует столбец: {col}")
                return

        features = selected_data[feature_columns]
        predicted_value = model.predict(features)[0]

        result_label.config(
            text=f"Дата: {selected_date}\n"
                 f"Рекомендуемый объём тендера: {predicted_value:.2f} тонн"
        )
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при прогнозе: {e}")

# Функция для построения графика
def plot_forecast():
    global data

    if data is None:
        messagebox.showerror("Ошибка", "Данные не загружены.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data['dt'], data['Цена на арматуру'], label='Фактические цены', color='blue', marker='o')
    
    predicted_values = model.predict(data[['Цена на арматуру', 'trend', 'seasonal']])
    ax.plot(data['dt'], predicted_values, label='Предсказанные цены', color='red', linestyle='--')

    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена на арматуру")
    ax.set_title("Прогноз цен на арматуру")
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

# Интерфейс приложения
def main():
    global data, date_var, result_label, graph_frame
    data = load_data()
    if data is None:
        return

    root = tk.Tk()
    root.title("Прогноз цен на арматуру")
    root.geometry("800x600")

    date_var = tk.StringVar()

    title_label = tk.Label(root, text="Прогноз цен на арматуру", font=("Arial", 16))
    title_label.pack(pady=10)

    date_label = tk.Label(root, text="Выберите дату:")
    date_label.pack(pady=5)

    dates = data['dt'].dt.strftime('%Y-%m-%d').tolist()
    date_dropdown = tk.OptionMenu(root, date_var, *dates)
    date_dropdown.pack(pady=5)
    if dates:
        date_var.set(dates[0])

    forecast_button = tk.Button(root, text="Предсказать", command=predict_tender)
    forecast_button.pack(pady=10)

    result_label = tk.Label(root, text="", font=("Arial", 12), fg="green", justify="left")
    result_label.pack(pady=10)

    graph_frame = tk.Frame(root)
    graph_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    plot_button = tk.Button(root, text="Построить график", command=plot_forecast)
    plot_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

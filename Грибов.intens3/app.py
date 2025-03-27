import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Функция для загрузки данных
def load_data():
    try:
        # Загрузка данных из CSV-файла
        data = pd.read_csv("catboost_prediction.csv")
        data['Дата'] = pd.to_datetime(data['Дата'])  # Преобразование столбца 'Дата' в формат datetime

        # Переименуем столбцы для удобства
        data.rename(columns={'Предсказанные значения': 'predicted_price'}, inplace=True)

        # Добавляем столбец "recommended_weeks"
        data["recommended_weeks"] = np.where(
            data["predicted_price"] > data["predicted_price"].shift(1),
            6,
            1
        )

        return data
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл 'catboost_predictions.csv' не найден.")
        return None
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")
        return None

# Функция для отображения прогноза
def show_forecast():
    selected_date = date_var.get()
    if not selected_date:
        messagebox.showerror("Ошибка", "Выберите дату.")
        return

    try:
        # Поиск данных для выбранной даты
        selected_data = data[data['Дата'] == pd.to_datetime(selected_date)]
        if selected_data.empty:
            messagebox.showerror("Ошибка", "Данные для выбранной даты отсутствуют.")
            return

        
        predicted_value = selected_data['predicted_price'].values[0]
        recommended_weeks = selected_data['recommended_weeks'].values[0]

        result_label.config(
            text=f"Дата: {selected_date}\n"
                 
                 f"Предсказанная цена: {predicted_value:.2f} руб./тонна\n"
                 f"Рекомендуемое количество недель для закупки: {recommended_weeks}"
        )
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при поиске прогноза: {e}")

# Функция для построения графика
def plot_forecast():
    global data

    if data is None:
        messagebox.showerror("Ошибка", "Данные не загружены.")
        return

    # Создание графика
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(data['Дата'], data['predicted_price'], label='Предсказанные значения', color='red', linestyle='--')

    # Выделение рекомендаций на графике
    for i, row in data.iterrows():
        if row['recommended_weeks'] == 6:
            ax.axvline(row['Дата'], color='green', linestyle='--', alpha=0.5, label='Рекомендация: 6 недель' if i == 0 else "")
        else:
            ax.axvline(row['Дата'], color='orange', linestyle='--', alpha=0.5, label='Рекомендация: 1 неделя' if i == 0 else "")

    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена на арматуру (руб./тонна)")
    ax.set_title("Сравнение реальных и предсказанных значений")
    ax.legend()

    # Отображение графика в интерфейсе
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

# Основная функция для создания интерфейса
def main():
    global data, date_var, result_label, graph_frame

    # Загрузка данных
    data = load_data()
    if data is None:
        return

    # Создание главного окна
    root = tk.Tk()
    root.title("Прогноз цен на арматуру")
    root.geometry("800x600")

    # Глобальные переменные
    date_var = tk.StringVar()

    # Заголовок
    title_label = tk.Label(root, text="Прогноз цен на арматуру", font=("Arial", 16))
    title_label.pack(pady=10)

    # Выбор даты
    date_label = tk.Label(root, text="Выберите дату:")
    date_label.pack(pady=5)

    # Заполнение выпадающего списка датами
    dates = data['Дата'].dt.strftime('%Y-%m-%d').tolist()
    date_dropdown = tk.OptionMenu(root, date_var, *dates)
    date_dropdown.pack(pady=5)
    if dates:
        date_var.set(dates[0])  # Устанавливаем первую дату по умолчанию

    # Кнопка для показа прогноза
    forecast_button = tk.Button(root, text="Показать прогноз", command=show_forecast)
    forecast_button.pack(pady=10)

    # Результат прогноза
    result_label = tk.Label(root, text="", font=("Arial", 12), fg="green", justify="left")
    result_label.pack(pady=10)

    # График
    graph_frame = tk.Frame(root)
    graph_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    # Кнопка для построения графика
    plot_button = tk.Button(root, text="Построить график", command=plot_forecast)
    plot_button.pack(pady=10)

    # Запуск главного цикла
    root.mainloop()

# Запуск приложения
if __name__ == "__main__":
    main()
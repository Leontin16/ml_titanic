# Titanic Dataset Analysis

## Описание проекта
Полный анализ датасета Titanic с использованием методов машинного обучения. Задача бинарной классификации - предсказать выживаемость пассажиров.

## Результаты анализа

### Метрики моделей:
- **K-NN (k=9)**: Accuracy = 73.33% (лучшая модель)
- **Random Forest**: Accuracy = 60.00%
- **Logistic Regression**: Accuracy = 53.33%
- **Decision Tree**: Accuracy = 53.33%

### Ключевые выводы:
1. **Лучшая модель**: K-Nearest Neighbors с k=9
2. **Точность**: 73.33% на тестовой выборке
3. **Важные признаки**: Возраст, стоимость билета, порт посадки
4. **Качество**: F1-score = 81.82% (хороший баланс precision/recall)

## Технологии
- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Запуск проекта
```bash
# Установка зависимостей
pip install -r requirements.txt

# Создание датасета (опционально)
python create_full_sample.py

# Запуск анализа
python titanic_analysis_perfect.py

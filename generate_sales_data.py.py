import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data():
    """Генерация данных продаж магазина электроники"""
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(180)]
    
    base_sales = 50
    trend = 0.2
    noise_std = 10
    
    sales_data = []
    for i, date in enumerate(dates):
        day_of_week = date.weekday()
        
        # Базовые продажи + тренд
        daily_sales = base_sales + (trend * i)
        
        # Сезонность
        if day_of_week == 4:  # Пятница
            daily_sales *= 1.2
        elif day_of_week == 6:  # Воскресенье
            daily_sales *= 0.7
        
        # Случайный шум
        daily_sales += np.random.normal(0, noise_std)
        daily_sales = max(0, daily_sales)  # Продажи не могут быть отрицательными
        
        sales_data.append(daily_sales)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales_data,
        'day_of_week': [date.weekday() for date in dates],
        'day_name': [date.strftime('%A') for date in dates],
        'week': [date.isocalendar().week for date in dates],
        'month': [date.month for date in dates]
    })
    
    return df

if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv('sales_data.csv', index=False)
    print("Данные продаж сохранены в sales_data.csv")
    print(f"Всего записей: {len(df)}")
    print(f"Период: {df['date'].min()} - {df['date'].max()}")
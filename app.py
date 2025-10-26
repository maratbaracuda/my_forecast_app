import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

# Настройка страницы
st.set_page_config(
    page_title="Прогнозирование временных рядов",
    page_icon="📈",
    layout="wide"
)

# Функции для прогнозирования
def moving_average_forecast(data, window=7, forecast_days=30):
    """Прогноз методом скользящего среднего"""
    ma = data.rolling(window=window).mean().iloc[-1]
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast = pd.Series([ma] * forecast_days, index=future_dates)
    return forecast

def linear_regression_forecast(data, forecast_days=30):
    """Прогноз линейной регрессией"""
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.array(range(len(data), len(data) + forecast_days)).reshape(-1, 1)
    forecast_values = model.predict(future_X)
    
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast = pd.Series(forecast_values, index=future_dates)
    
    return forecast

def polynomial_regression_forecast(data, degree=2, forecast_days=30):
    """Прогноз полиномиальной регрессией"""
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    future_X = np.array(range(len(data), len(data) + forecast_days)).reshape(-1, 1)
    future_X_poly = poly.transform(future_X)
    forecast_values = model.predict(future_X_poly)
    
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast = pd.Series(forecast_values, index=future_dates)
    
    return forecast

# Функции для анализа данных
def calculate_metrics(actual, predicted):
    """Расчет метрик качества"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def detect_anomalies(data, std_threshold=2):
    """Детектирование аномалий"""
    mean = np.mean(data)
    std = np.std(data)
    threshold = std_threshold * std
    anomalies = data[np.abs(data - mean) > threshold]
    return anomalies

# Главная функция приложения
def main():
    st.title("📈 Приложение для прогнозирования временных рядов")
    
    # Сайдбар для навигации
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите страницу:", [
        "Прогнозирование", 
        "Анализ продаж", 
        "Аналитика",
        "Метрики качества"
    ])
    
    # Страница 1: Прогнозирование
    if page == "Прогнозирование":
        st.header("📊 Прогнозирование временных рядов")
        
        # Загрузка файла
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Чтение данных
                df = pd.read_csv(uploaded_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # Проверка наличия необходимых колонок
                if 'value' not in df.columns:
                    st.error("Файл должен содержать колонку 'value'")
                    return
                
                # Отображение статистики
                st.subheader("📊 Статистика данных")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Среднее", f"{df['value'].mean():.2f}")
                with col2:
                    st.metric("Минимум", f"{df['value'].min():.2f}")
                with col3:
                    st.metric("Максимум", f"{df['value'].max():.2f}")
                with col4:
                    st.metric("Стандартное отклонение", f"{df['value'].std():.2f}")
                
                # Настройки прогноза
                st.subheader("⚙️ Настройки прогноза")
                col1, col2 = st.columns(2)
                with col1:
                    forecast_method = st.selectbox(
                        "Метод прогнозирования:",
                        ["Линейная регрессия", "Полиномиальная регрессия", "Скользящее среднее"]
                    )
                    forecast_days = st.slider("Дней для прогноза:", 7, 90, 30)
                
                with col2:
                    if forecast_method == "Полиномиальная регрессия":
                        degree = st.slider("Степень полинома:", 2, 5, 2)
                    elif forecast_method == "Скользящее среднее":
                        window = st.slider("Окно скользящего среднего:", 3, 30, 7)
                
                # Прогнозирование
                if st.button("Сгенерировать прогноз"):
                    with st.spinner("Генерация прогноза..."):
                        if forecast_method == "Линейная регрессия":
                            forecast = linear_regression_forecast(df['value'], forecast_days)
                        elif forecast_method == "Полиномиальная регрессия":
                            forecast = polynomial_regression_forecast(df['value'], degree, forecast_days)
                        else:
                            forecast = moving_average_forecast(df['value'], window, forecast_days)
                        
                        # Визуализация
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['value'],
                            name='Исходные данные',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast.index, y=forecast.values,
                            name='Прогноз',
                            line=dict(color='red', dash='dash')
                        ))
                        fig.update_layout(
                            title="Прогноз временного ряда",
                            xaxis_title="Дата",
                            yaxis_title="Значение"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Скачивание результатов
                        forecast_df = pd.DataFrame({
                            'date': forecast.index,
                            'value': forecast.values
                        })
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Скачать прогноз (CSV)",
                            data=csv,
                            file_name="forecast.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")
    
    # Страница 2: Анализ продаж
    elif page == "Анализ продаж":
        st.header("🏪 Анализ продаж магазина электроники")
        
        if st.button("Сгенерировать данные продаж"):
            # Генерация данных
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
            
            df_sales = pd.DataFrame({
                'date': dates,
                'sales': sales_data,
                'day_of_week': [date.weekday() for date in dates],
                'day_name': [date.strftime('%A') for date in dates]
            })
            
            # Сохранение в CSV
            df_sales.to_csv('sales_data.csv', index=False)
            st.success("Данные продаж сгенерированы и сохранены в sales_data.csv")
            
            # Визуализация по неделям
            df_sales['week'] = df_sales['date'].dt.isocalendar().week
            weekly_sales = df_sales.groupby('week')['sales'].sum().reset_index()
            
            fig_weekly = px.line(weekly_sales, x='week', y='sales', 
                               title="Продажи по неделям")
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Анализ по дням недели
            st.subheader("📈 Анализ по дням недели")
            daily_stats = df_sales.groupby('day_name')['sales'].agg(['mean', 'std']).reset_index()
            
            fig_daily = px.bar(daily_stats, x='day_name', y='mean',
                             error_y='std', title="Средние продажи по дням недели")
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Общий тренд
            st.subheader("📊 Общий тренд")
            X = np.array(range(len(df_sales))).reshape(-1, 1)
            y = df_sales['sales'].values
            model = LinearRegression()
            model.fit(X, y)
            trend_slope = model.coef_[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Тренд (продаж в день)", f"{trend_slope:.3f}")
            
            # Самые успешные и неуспешные дни
            best_day = df_sales.loc[df_sales['sales'].idxmax()]
            worst_day = df_sales.loc[df_sales['sales'].idxmin()]
            
            with col2:
                st.metric("Самый успешный день", 
                         f"{best_day['sales']:.1f} ({best_day['date'].strftime('%Y-%m-%d')})")
                st.metric("Самый неуспешный день", 
                         f"{worst_day['sales']:.1f} ({worst_day['date'].strftime('%Y-%m-%d')})")
            
            # Детектирование аномалий
            st.subheader("🚨 Детектирование аномалий")
            anomalies = detect_anomalies(df_sales['sales'])
            
            if len(anomalies) > 0:
                st.write(f"Найдено аномалий: {len(anomalies)}")
                anomaly_dates = df_sales.loc[anomalies.index, ['date', 'sales']]
                st.dataframe(anomaly_dates)
            else:
                st.info("Аномалии не обнаружены")
    
    # Страница 3: Аналитика
    elif page == "Аналитика":
        st.header("🔍 Аналитика временных рядов")
        
        uploaded_file = st.file_uploader("Загрузите CSV файл для анализа", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            if 'value' not in df.columns:
                st.error("Файл должен содержать колонку 'value'")
                return
            
            # Распределение данных
            st.subheader("📊 Распределение данных")
            fig_dist = px.histogram(df, x='value', title="Распределение значений")
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Автокорреляция
            st.subheader("🔄 Автокорреляция")
            autocorr = [df['value'].autocorr(lag=i) for i in range(1, 31)]
            fig_acf = px.line(x=range(1, 31), y=autocorr, 
                            title="Автокорреляционная функция")
            fig_acf.update_layout(xaxis_title="Лаг", yaxis_title="Автокорреляция")
            st.plotly_chart(fig_acf, use_container_width=True)
            
            # Сезонность (если данных достаточно)
            if len(df) > 30:
                st.subheader("📅 Сезонность")
                df['month'] = df.index.month
                df['day_of_week'] = df.index.dayofweek
                
                monthly_avg = df.groupby('month')['value'].mean()
                daily_avg = df.groupby('day_of_week')['value'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_monthly = px.bar(monthly_avg, title="Средние значения по месяцам")
                    st.plotly_chart(fig_monthly, use_container_width=True)
                
                with col2:
                    fig_daily = px.bar(daily_avg, title="Средние значения по дням недели")
                    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Страница 4: Метрики качества
    elif page == "Метрики качества":
        st.header("📐 Метрики качества прогнозирования")
        
        uploaded_file = st.file_uploader("Загрузите CSV файл с фактическими и прогнозными значениями", 
                                       type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Проверка необходимых колонок
            required_cols = ['actual', 'predicted']
            if all(col in df.columns for col in required_cols):
                # Расчет метрик
                mae, rmse, mape = calculate_metrics(df['actual'], df['predicted'])
                
                # Отображение метрик в карточках
                st.subheader("📊 Метрики качества")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Таблица с ошибками
                st.subheader("📋 Детализация ошибок")
                df_errors = df.copy()
                df_errors['error'] = df_errors['actual'] - df_errors['predicted']
                df_errors['abs_error'] = np.abs(df_errors['error'])
                df_errors['ape'] = np.abs(df_errors['error'] / df_errors['actual']) * 100
                
                st.dataframe(df_errors)
                
                # Визуализация остатков
                st.subheader("📈 Анализ остатков")
                fig_residuals = px.scatter(df_errors, x='predicted', y='error',
                                        title="Остатки vs Прогнозные значения")
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)
                
            else:
                st.error("Файл должен содержать колонки 'actual' и 'predicted'")

if __name__ == "__main__":
    main()
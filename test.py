# FINALNA PREDYKCJA NA 2017 ROK - PROPHET

print("=== FINALNA PREDYKCJA NA 2017 ROK (Prophet) ===")

from prophet import Prophet

# Przygotuj dane dla Prophet
df_prophet = monthly_sales[['Order Month', 'Sales']].rename(columns={'Order Month': 'ds', 'Sales': 'y'})

# Inicjalizacja i trenowanie modelu
model = Prophet(yearly_seasonality=True)
model.fit(df_prophet)

# Stwórz przyszłe daty do prognozy (12 miesięcy 2017)
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)

# Wyciągnij tylko prognozy na 2017
forecast_2017 = forecast[forecast['ds'].dt.year == 2017][['ds', 'yhat']]

print("\n=== PREDYKCJE NA 2017 ROK (Prophet) ===")
print(forecast_2017)

# Wykres: cała seria + predykcja na 2017
plt.figure(figsize=(15, 7))
plt.plot(monthly_sales['Order Month'], monthly_sales['Sales'], 'o-', label='Historyczne', color='blue')
plt.plot(forecast_2017['ds'], forecast_2017['yhat'], 's--', label='Predykcja 2017 (Prophet)', color='green', linewidth=3, markersize=8)
plt.xlabel('Data')
plt.ylabel('Sprzedaż')
plt.title('Predykcja Prophet na 2017 rok')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
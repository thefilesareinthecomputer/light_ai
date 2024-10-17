@staticmethod
def get_weather_forecast():
    '''get_weather_forecast gets a spoken weather forecast from openweathermap for the next 4 days by day part based on user defined home location'''
    appid = f'{open_weather_api_key}'

    # Fetching coordinates from environment variables
    lat = USER_SELECTED_HOME_LAT
    lon = USER_SELECTED_HOME_LON

    # OpenWeatherMap API endpoint for 4-day hourly forecast
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={appid}"

    response = requests.get(url)
    print("Response status:", response.status_code)
    if response.status_code != 200:
        return "Failed to retrieve weather data."

    data = response.json()
    print("Data received:", data)

    # Process forecast data
    forecast = ""
    timezone = pytz.timezone(USER_SELECTED_TIMEZONE)
    now = datetime.now(timezone)
    periods = [(now + timedelta(days=i)).replace(hour=h, minute=0, second=0, microsecond=0) for i in range(4) for h in [6, 12, 18, 0]]

    for i in range(0, len(periods), 4):
        day_forecasts = []
        for j in range(4):
            start, end = periods[i + j], periods[i + j + 1] if j < 3 else periods[i] + timedelta(days=1)
            period_forecast = [f for f in data['list'] if start <= datetime.fromtimestamp(f['dt'], tz=timezone) < end]
            
            if period_forecast:
                avg_temp_kelvin = sum(f['main']['temp'] for f in period_forecast) / len(period_forecast)
                avg_temp_fahrenheit = (avg_temp_kelvin - 273.15) * 9/5 + 32  # Convert from Kelvin to Fahrenheit
                descriptions = set(f['weather'][0]['description'] for f in period_forecast)
                time_label = ["morning", "afternoon", "evening", "night"][j]
                day_forecasts.append(f"{time_label}: average temperature {avg_temp_fahrenheit:.1f}°F, conditions: {', '.join(descriptions)}")

        if day_forecasts:
            forecast_date = periods[i].strftime('%Y-%m-%d')
            # Convert forecast_date to weekday format aka "Monday", etc.
            forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').strftime('%A')
            forecast += f"\n{forecast_date}: {'; '.join(day_forecasts)}."

            # print("Weather forecast:", forecast)
            # SpeechToTextTextToSpeechIO.speak_mainframe(f'Weather forecast for {USER_SELECTED_HOME_CITY}, {USER_SELECTED_HOME_STATE}: {forecast}')
            weather_forecast = f'Weather forecast for next 4 days, broken out by 6 hour day part: {forecast}'
            
        else:
            print("No weather forecast data available.")
            
    if weather_forecast:
        response = gemini_model.generate_content(f"""### SYSTEM MESSAGE START ### 
                                                    You are a weather report summarizer. 
                                                    Your output must be short and concise. Limit your response to just a few sentences. 
                                                    The report below is weather for the next 4 days, by 6 hour day part, and it's too verbose to be practical. 
                                                    Provide a concise (short) summary of this weather forecast with recommendations for how the user should navigate 
                                                    this weather on each day. Limit your reply to just 1-2 sentences per day.  
                                                    Be concise. Here is the report to summarize: {weather_forecast}
                                                    ### SYSTEM MESSAGE END ###""", stream=True)
        if response:
            response.resolve()
            print(f"Response from Gemini: {response.text}")
            SpeechToTextTextToSpeechIO.speak_mainframe(f'{response.text}')
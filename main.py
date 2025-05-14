# COVID-19 Global Data Analysis
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# 1. Data Collection & Loading
try:
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    print("✅ Dataset loaded successfully")
except Exception as e:
    print(f"🚨 Data loading failed: {e}")
    exit()

# 2. Data Exploration
print("\n🔍 Dataset Overview:")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False)[:10])

# 3. Data Cleaning
# Select key countries and columns
countries = ['Kenya', 'United States', 'India', 'Germany', 'Brazil']
cols = ['date', 'location', 'total_cases', 'new_cases', 
        'total_deaths', 'new_deaths', 'people_vaccinated', 
        'population', 'iso_code']

# Filter and clean data
covid = df[df['location'].isin(countries)][cols]
covid['date'] = pd.to_datetime(covid['date'])
covid.sort_values(['location', 'date'], inplace=True)

# Forward fill missing values within each country
covid = covid.groupby('location').apply(lambda x: x.ffill())
print("\n🧹 Cleaning Summary:")
print(f"Remaining missing values: {covid.isnull().sum().sum()}")

# 4. EDA: Time Trends Analysis
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Cases Over Time
plt.subplot(2, 2, 1)
for country in countries:
    subset = covid[covid['location'] == country]
    plt.plot(subset['date'], subset['total_cases'], label=country)
plt.title('Total COVID-19 Cases Over Time')
plt.xticks(rotation=45)

# Deaths Over Time
plt.subplot(2, 2, 2)
for country in countries:
    subset = covid[covid['location'] == country]
    plt.plot(subset['date'], subset['total_deaths'], label=country)
plt.title('Total Deaths Over Time')
plt.xticks(rotation=45)

plt.tight_layout()
plt.legend()
plt.show()

# 5. Vaccination Analysis
plt.figure(figsize=(12, 6))
for country in countries:
    subset = covid[covid['location'] == country].dropna(subset=['people_vaccinated'])
    vaccination_pct = (subset['people_vaccinated'] / subset['population']) * 100
    plt.plot(subset['date'], vaccination_pct, label=country, marker='o')

plt.title('Vaccination Progress (% Population)')
plt.ylabel('Percentage Vaccinated')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 6. Choropleth Map (Latest Data)
latest_date = covid['date'].max()
latest_data = covid[covid['date'] == latest_date]

fig = px.choropleth(latest_data,
                    locations="iso_code",
                    color="total_cases",
                    hover_name="location",
                    scope="world",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title=f"COVID-19 Case Distribution as of {latest_date.strftime('%Y-%m-%d')}")
fig.show()

# 7. Key Metrics Calculation
latest_data['death_rate'] = (latest_data['total_deaths'] / latest_data['total_cases']) * 100
latest_data['vaccination_rate'] = (latest_data['people_vaccinated'] / latest_data['population']) * 100

print("\n📊 Key Metrics Table:")
print(latest_data[['location', 'total_cases', 'total_deaths', 
                 'death_rate', 'vaccination_rate']].round(2))

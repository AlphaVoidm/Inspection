"""Creates examples/sample_energy_panel.csv - a small demo panel with deliberate data-quality defects.

The defects are intentional so the inspection notebook has something to find:
missing values, duplicated rows, a duplicated country-date key, a constant column,
an empty column, GDP stored as text with thousand separators, inconsistent country
identifiers, an invalid latitude, sentinel values, gaps in the monthly calendar and
unbalanced entity coverage.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

COUNTRIES = [
    ("Germany", "DEU", 51.2, 10.4, 83.2),
    ("France", "FRA", 46.2, 2.2, 67.8),
    ("Spain", "ESP", 40.5, -3.7, 47.4),
    ("Poland", "POL", 52.0, 19.1, 37.8),
    ("United States", "USA", 39.8, -98.6, 331.9),
    ("USA", "USA", 39.8, -98.6, 331.9),      # same country, second identifier
    ("Norway", "NOR", 60.5, 8.5, 5.4),       # short coverage
]

months = pd.period_range("2015-01", "2023-12", freq="M")
rows = []

for name, iso, lat, lon, pop in COUNTRIES:
    if name == "Norway":
        country_months = months[-30:]        # deliberately short coverage
    elif name == "USA":
        country_months = months[:24]         # the alias identifier only covers 2015-2016
    else:
        country_months = months

    base = {"Germany": 45000, "France": 42000, "Spain": 25000, "Poland": 15000,
            "United States": 380000, "USA": 380000, "Norway": 13000}[name]

    for i, m in enumerate(country_months):
        # deliberate calendar gaps
        if name == "Spain" and str(m) in {"2018-05", "2018-06", "2018-07"}:
            continue
        if name == "Poland" and m.year == 2020 and m.month in (3, 4):
            continue

        season = 1 + 0.18 * np.cos(2 * np.pi * (m.month - 1) / 12)
        trend = 1 + 0.004 * i
        demand = base * season * trend * rng.normal(1, 0.03)

        # a structural break in Poland: reporting switches from MWh to MW-ish scale
        if name == "Poland" and m >= pd.Period("2021-01", freq="M"):
            demand *= 12

        solar = max(0.0, base * 0.06 * season * rng.normal(1, 0.4))
        wind = max(0.0, base * 0.11 * rng.normal(1, 0.5))
        temp = 10 + 12 * np.cos(2 * np.pi * (m.month - 7) / 12) + rng.normal(0, 2)

        rows.append({
            "country_name": name,
            "iso3": iso,
            "date": f"{m}-01",
            "year": m.year,
            "month": m.month,
            "electricity_demand_mwh": round(demand, 1),
            "solar_generation_mwh": round(solar, 1),
            "wind_generation_mwh": round(wind, 1),
            "temperature_c": round(temp, 2),
            "population_millions": pop,
            "gdp_usd": f"{int(base * 90 + i * 1200):,}",     # numeric stored as text
            "latitude": lat,
            "longitude": lon,
            "data_source": "demo_v1",                        # constant column
            "notes": np.nan,                                 # 100% empty column
        })

df = pd.DataFrame(rows)

# missing values in two columns
df.loc[df.sample(frac=0.06, random_state=1).index, "temperature_c"] = np.nan
df.loc[df.sample(frac=0.31, random_state=2).index, "solar_generation_mwh"] = np.nan

# text placeholders that pandas will not read as NaN
df.loc[df.sample(frac=0.02, random_state=3).index, "gdp_usd"] = "N/A"

# a handful of sentinel / suspicious values
df.loc[df.sample(n=5, random_state=4).index, "electricity_demand_mwh"] = -9999
df.loc[df.sample(n=3, random_state=5).index, "wind_generation_mwh"] = 0.0

# one invalid latitude
df.loc[df.index[10], "latitude"] = 95.4

# duplicated rows
df = pd.concat([df, df.sample(n=12, random_state=6)], ignore_index=True)

# a duplicated country-date key with a different value (not an identical row)
extra = df[df["country_name"] == "France"].head(4).copy()
extra["electricity_demand_mwh"] = extra["electricity_demand_mwh"] * 1.01
df = pd.concat([df, extra], ignore_index=True)

df = df.sample(frac=1.0, random_state=8).reset_index(drop=True)   # unsorted on purpose
df.to_csv("examples/sample_energy_panel.csv", index=False)
print("Wrote examples/sample_energy_panel.csv", df.shape)

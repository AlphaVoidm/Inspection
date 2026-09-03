"""
Bundled static reference data used for inference.

Everything here is used for *inference* only and is always reported with a
confidence level.  Nothing here modifies source data.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Aggregate / non-country entities that must NOT be silently treated as a
# single country (spec section 10/11).
# ---------------------------------------------------------------------------
AGGREGATE_ENTITIES = {
    "world", "global", "world bank", "global total", "international",
    "africa", "asia", "europe", "north america", "south america",
    "oceania", "antarctica", "european union", "eu", "euro area", "eurozone",
    "oecd", "opec", "g7", "g20", "brics", "latin america", "caribbean",
    "middle east", "mena", "sub-saharan africa", "south asia", "east asia",
    "high income", "upper middle income", "lower middle income", "low income",
    "high-income", "upper-middle-income", "lower-middle-income", "low-income",
    "developed", "developing", "emerging markets", "least developed",
    "arab world", "central america", "southern africa", "northern africa",
    "western europe", "eastern europe", "central europe", "southeast asia",
    "central asia", "pacific islands", "small states", "other small states",
    "fragile states", "low & middle income", "low and middle income",
    "aggregate", "total", "sum", "all countries", "rest of world", "row",
    "world total", "wld", "ida total", "ibrd total", "euro area (19)",
}

# ---------------------------------------------------------------------------
# Common country name aliases -> ISO3.  Used only when pycountry is absent or
# as an extra hint.  A curated set is sufficient because the audit always
# marks low-confidence mappings as AMBIGUOUS / UNCERTAIN.
# ---------------------------------------------------------------------------
COUNTRY_ALIASES = {
    # name variants -> ISO3
    "united states": "USA", "united states of america": "USA", "usa": "USA",
    "u.s.": "USA", "u.s.a.": "USA", "us": "USA",
    "united kingdom": "GBR", "uk": "GBR", "great britain": "GBR",
    "england": "GBR", "britain": "GBR",
    "russia": "RUS", "russian federation": "RUS",
    "south korea": "KOR", "korea, rep.": "KOR", "korea republic": "KOR",
    "republic of korea": "KOR",
    "north korea": "PRK", "democratic people's republic of korea": "PRK",
    "czech republic": "CZE", "czechia": "CZE",
    "türkiye": "TUR", "turkiye": "TUR", "turkey": "TUR",
    "iran": "IRN", "iran, islamic rep.": "IRN", "islamic republic of iran": "IRN",
    "viet nam": "VNM", "vietnam": "VNM",
    "venezuela": "VEN", "venezuela, rb": "VEN",
    "egypt": "EGY", "egypt, arab rep.": "EGY", "arab republic of egypt": "EGY",
    "syria": "SYR", "syrian arab republic": "SYR",
    "laos": "LAO", "lao pdr": "LAO",
    "congo, dem. rep.": "COD", "dr congo": "COD", "democratic republic of the congo": "COD",
    "congo, rep.": "COG", "republic of the congo": "COG", "congo": "COG",
    "côte d'ivoire": "CIV", "cote d'ivoire": "CIV", "ivory coast": "CIV",
    "cape verde": "CPV", "cabo verde": "CPV",
    "the gambia": "GMB", "gambia": "GMB",
    "bolivia": "BOL", "bolivia (plurinational state of)": "BOL",
    "moldova": "MDA", "republic of moldova": "MDA",
    "tanzania": "TZA", "united republic of tanzania": "TZA",
    "brunei": "BRN", "brunei darussalam": "BRN",
    "myanmar": "MMR", "burma": "MMR",
    "north macedonia": "MKD", "macedonia": "MKD",
    "e swatini": "SWZ", "eswatini": "SWZ", "swaziland": "SWZ",
    "timor-leste": "TLS", "east timor": "TLS",
    "palestine": "PSE", "state of palestine": "PSE", "west bank and gaza": "PSE",
    "kosovo": "XKX",
    "taiwan": "TWN", "taiwan, china": "TWN",
    "hong kong": "HKG", "hong kong sar, china": "HKG",
    "macao": "MAC", "macau": "MAC",
    "china": "CHN", "people's republic of china": "CHN", "prc": "CHN",
    "india": "IND", "republic of india": "IND",
    "brazil": "BRA", "federative republic of brazil": "BRA",
    "germany": "DEU", "federal republic of germany": "DEU",
    "france": "FRA", "french republic": "FRA",
    "italy": "ITA", "italian republic": "ITA",
    "japan": "JPN", "spain": "ESP", "canada": "CAN", "australia": "AUS",
    "netherlands": "NLD", "holland": "NLD",
    "belgium": "BEL", "switzerland": "CHE", "sweden": "SWE", "norway": "NOR",
    "denmark": "DNK", "finland": "FIN", "austria": "AUT", "poland": "POL",
    "portugal": "PRT", "greece": "GRC", "ireland": "IRL", "mexico": "MEX",
    "chile": "CHL", "argentina": "ARG", "colombia": "COL", "peru": "PER",
    "south africa": "ZAF", "nigeria": "NGA", "kenya": "KEN", "ethiopia": "ETH",
    "ghana": "GHA", "morocco": "MAR", "algeria": "DZA", "tunisia": "TUN",
    "libya": "LBY", "sudan": "SDN", "iraq": "IRQ", "saudi arabia": "SAU",
    "united arab emirates": "ARE", "uae": "ARE", "qatar": "QAT", "kuwait": "KWT",
    "israel": "ISR", "jordan": "JOR", "lebanon": "LBN", "pakistan": "PAK",
    "bangladesh": "BGD", "indonesia": "IDN", "malaysia": "MYS",
    "thailand": "THA", "philippines": "PHL", "singapore": "SGP",
    "new zealand": "NZL", "ukraine": "UKR", "belarus": "BLR", "kazakhstan": "KAZ",
    "uzbekistan": "UZB", "azerbaijan": "AZE", "georgia": "GEO", "armenia": "ARM",
    "serbia": "SRB", "croatia": "HRV", "slovenia": "SVN", "slovakia": "SVK",
    "hungary": "HUN", "czechia": "CZE", "romania": "ROU", "bulgaria": "BGR",
    "estonia": "EST", "latvia": "LVA", "lithuania": "LTU", "iceland": "ISL",
    "luxembourg": "LUX", "malta": "MLT", "cyprus": "CYP", "albania": "ALB",
    "bosnia and herzegovina": "BIH", "montenegro": "MNE", "yemen": "YEM",
    "oman": "OMN", "bahrain": "BHR", "afghanistan": "AFG", "nepal": "NPL",
    "sri lanka": "LKA", "mongolia": "MNG", "cambodia": "KHM", "cuba": "CUB",
    "dominican republic": "DOM", "haiti": "HTI", "jamaica": "JAM",
    "trinidad and tobago": "TTO", "ecuador": "ECU", "uruguay": "URY",
    "paraguay": "PRY", "guatemala": "GTM", "honduras": "HND", "el salvador": "SLV",
    "nicaragua": "NIC", "costa rica": "CRI", "panama": "PAN",
}

# ---------------------------------------------------------------------------
# Unit tokens for detection (spec section 15).  key -> (canonical, kind)
# ---------------------------------------------------------------------------
UNIT_TOKENS = [
    # power
    ("megawatt", "MW", "power"), ("mw", "MW", "power"),
    ("gigawatt", "GW", "power"), ("gw", "GW", "power"),
    ("kilowatt", "kW", "power"), ("kw", "kW", "power"),
    ("watt", "W", "power"), (" w", "W", "power"),
    # energy
    ("terawatt-hour", "TWh", "energy"), ("twh", "TWh", "energy"),
    ("gigawatt-hour", "GWh", "energy"), ("gwh", "GWh", "energy"),
    ("megawatt-hour", "MWh", "energy"), ("mwh", "MWh", "energy"),
    ("kilowatt-hour", "kWh", "energy"), ("kwh", "kWh", "energy"),
    ("watt-hour", "Wh", "energy"), ("wh", "Wh", "energy"),
    # temperature
    ("celsius", "°C", "temperature"), ("°c", "°C", "temperature"), ("deg c", "°C", "temperature"),
    ("kelvin", "K", "temperature"), (" k", "K", "temperature"),
    ("fahrenheit", "°F", "temperature"), ("°f", "°F", "temperature"),
    # ratios / money
    ("percent", "%", "ratio"), ("percentage", "%", "ratio"), (" pct", "%", "ratio"),
    ("usd", "USD", "money"), ("$", "USD", "money"), ("us dollar", "USD", "money"),
    ("eur", "EUR", "money"), ("euro", "EUR", "money"),
    ("local currency", "local currency", "money"),
    ("persons", "persons", "count"), ("people", "persons", "count"),
    ("population", "persons", "count"), ("inhabitants", "persons", "count"),
    ("tonnes", "tonnes", "mass"), ("tons", "tons", "mass"), ("kg", "kg", "mass"),
    ("co2", "tCO2", "emissions"), ("emissions", "tCO2", "emissions"),
    ("usd/capita", "USD/capita", "money_per_capita"),
    ("index", "index", "index"), ("share", "%", "ratio"),
    ("gdp", "currency units", "money"), ("per capita", "per capita", "per_capita"),
]

# ---------------------------------------------------------------------------
# Dataset purpose keyword lexicon (spec section 19).  Ordered; first match wins
# is NOT used — every keyword contributes evidence and the strongest wins.
# ---------------------------------------------------------------------------
PURPOSE_KEYWORDS = [
    ("electricity demand", ["demand", "load", "electricity demand", "power demand", "load_mw", "demand_mw"]),
    ("electricity generation", ["generation", "gen_mw", "power generation", "electricity generation", "output_mw"]),
    ("electricity consumption", ["consumption", "electricity consumption", "final consumption"]),
    ("renewable energy", ["renewable", "renewables", "renewable share", "solar", "wind", "hydro", "pv"]),
    ("energy mix", ["energy mix", "fuel mix", "generation mix", "share of", "mix"]),
    ("prices", ["price", "prices", "cost", "tariff", "usd", "eur"]),
    ("weather", ["weather", "precipitation", "humidity", "wind speed", "solar irradiance", "cloud"]),
    ("temperature", ["temperature", "temp", "°c", "deg c", "tmin", "tmax", "tas", "surface temperature"]),
    ("gdp", ["gdp", "gross domestic product", "gross value added"]),
    ("population", ["population", "pop", "inhabitants", "persons"]),
    ("transport", ["transport", "traffic", "vehicle", "mobility", "fuel consumption"]),
    ("demographics", ["demographic", "age", "birth", "death", "mortality", "fertility", "urban"]),
    ("emissions", ["emissions", "co2", "ghg", "carbon"]),
    ("electricity capacity", ["capacity", "installed capacity", "capacity_mw"]),
    ("renewable capacity", ["renewable capacity", "installed renewable"]),
    ("trade", ["trade", "import", "export"]),
    ("finance", ["gdp", "inflation", "interest rate", "exchange rate", "cpi"]),
]

# ---------------------------------------------------------------------------
# Column-name keyword groups used for semantic role inference.
# ---------------------------------------------------------------------------
ROLE_KEYWORDS = {
    "DATE": ["date", "day", "datum", "fecha"],
    "DATETIME": ["datetime", "timestamp", "ts", "time_stamp", "created_at", "updated_at", "dt"],
    "YEAR": ["year", "yr", "annee", "anno", "jahr", "ano"],
    "MONTH": ["month", "monat", "mes"],
    "TIME": ["time", "hour", "minute", "second", "heure", "hora"],
    "COUNTRY": ["country", "country_name", "country name", "nation", "pays", "land", "countryname"],
    "COUNTRY_CODE": ["country_code", "country code", "cc", "countrycode"],
    "ISO2": ["iso2", "iso_2", "iso 2", "iso-2", "alpha2", "alpha_2", "alpha-2", "cc2"],
    "ISO3": ["iso3", "iso_3", "iso 3", "iso-3", "alpha3", "alpha_3", "alpha-3", "iso_code", "iso code", "isocode"],
    "REGION": ["region", "region_name", "regione", "région"],
    "CONTINENT": ["continent", "kontinent"],
    "LOCATION": ["location", "place", "loc", "site", "city", "town", "village"],
    "STATION": ["station", "plant", "facility", "unit", "asset", "meter", "sensor", "site_id"],
    "LATITUDE": ["latitude", "lat", "north", "y_coord", "ycoord"],
    "LONGITUDE": ["longitude", "lon", "lng", "long", "east", "x_coord", "xcoord"],
    "UNIT": ["unit", "units", "uom", "measurement_unit", "unit_of_measure"],
    "VALUE": ["value", "values", "amount", "measure", "observation", "reading"],
    "TARGET": ["target", "y", "label", "dependent", "outcome", "response"],
    "IDENTIFIER": ["id", "code", "key", "uuid", "guid", "identifier", "ref", "nr", "number_id"],
    "INDEX": ["index", "idx", "row_id", "rowid"],
    "TEXT": ["text", "description", "note", "comment", "remarks", "name", "title", "label"],
    "FORECAST": ["forecast", "projection", "predicted", "prediction", "scenario", "future"],
    "ACTUAL": ["actual", "observed", "realized", "real", "historic", "historical"],
}

# Value bands used to identify likely year-like / month-like / coordinate columns.
YEAR_RANGE = (1700, 2200)
MONTH_RANGE = (1, 12)
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)

# Known frequency labels (spec section 8).
FREQUENCY_LABELS = [
    "SUB-HOURLY", "HOURLY", "DAILY", "WEEKLY", "MONTHLY",
    "QUARTERLY", "ANNUAL", "IRREGULAR", "UNKNOWN",
]

"""
Weather functions

Sample functions to demonstrate function calling capabilities.
"""

import requests
import urllib.parse


# From here: https://open-meteo.com/en/docs
def interpret_WMO_current_weather_code(wmo_code: int):
    match wmo_code:
        case 0:
            return "Clear"
        case 1:
            return "Mainly Clear"
        case 2:
            return "Partly Cloudy"
        case 3:
            return "Overcast"
        case 45:
            return "Fog"
        case 48:
            return "Ice Fog"
        case 51:
            return "Light Drizzle"
        case 53:
            return "Moderate Drizzle"
        case 55:
            return "Dense Drizzle"
        case 56:
            return "Light Freezing Drizzle"
        case 57:
            return "Dense Freezing Drizzle"
        case 61:
            return "Light Rain"
        case 63:
            return "Moderate Rain"
        case 65:
            return "Heavy Rain"
        case 66:
            return "Light Freezing Rain"
        case 67:
            return "Heavy Freezing Rain"
        case 71:
            return "Light Snow"
        case 73:
            return "Moderate Snow"
        case 75:
            return "Heavy Snow"
        case 77:
            return "Snow Grains"
        case 80:
            return "Light Rain Showers"
        case 81:
            return "Moderate Rain Showers"
        case 82:
            return "Heavy Rain Showers"
        case 85:
            return "Light Snow Showers"
        case 86:
            return "Heavy Snow Showers"
        case 95:
            return "Thunderstorm"
        case 96:
            return "Thunderstorm with Hail"
        case 86:
            return "Thunderstorm with Heavy Hail"
    return "Unknown"


# Turn wind direction in to human readable string
def convert_degrees_to_compass_dir(degrees: int | None):
    if not degrees:
        return ""
    val = int((degrees / 22.5) + 0.5)
    direction_names = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    return direction_names[(val % 16)]


def get_weather(location: str):
    """
    Get the current weather at a specified location.

    Args:
        location: The location to get the temperature for without a country name.
    Returns:
        A text description of the weather at the given location.
    """
    if not location:
        return "Invalid location."
    print(f"Gettign weather for {location}:")

    # STEP 1: Call the open-meteo geocoding API to figure out latititude and longitude coordinates
    # Docs: https://open-meteo.com/en/docs/geocoding-api
    # The open-meteo geocoding API doesn't like if you include country names.
    # SO we're going to strip this info, which may cause the result to come from the wrong city.
    location = location.split(",")[0]

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
    response = requests.get(url)

    data = None
    if response.status_code == 200:
        try:
            data = response.json()["results"][0]
        except (KeyError, IndexError):
            return f"Invalid location data returned for location: {location}."
    else:
        return f"Failed getting details for location: {location}. (HTTP {response.status_code})"

    if not data or "latitude" not in data or "longitude" not in data:
        print("Unable to read location data for location: {location}")
        print(data)
        return f"Invalid data returned for location: {location}"

    # Parse location from city details
    latitude = data["latitude"]
    longitude = data["longitude"]
    tmz = data.get("timezone", "GMT")
    print(f"Latitude: {latitude}, Longitude: {longitude}, Timezone: {tmz}")

    # STEP 2: Call the open-meteo weather API with geogeraphic coordinates determined in STEP 1
    # Docs: https://open-meteo.com/en/docs
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
        ],
        "timezone": tmz,
        "forecast_days": 1,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            return f"Error fetching weather data: {data['message']}"
    except requests.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

    # Current conditions comes back as a "WMO code" integer
    description = interpret_WMO_current_weather_code(int(data["current"]["weather_code"]))

    # Format the result
    values = data["current"]
    units = data["current_units"]
    wind_direction = convert_degrees_to_compass_dir(int(values.get("wind_direction_10m", None)))

    return f"""Description: {description}
Temperature: {values.get("temperature_2m", "")}{units.get("temperature_2m", "")}
Feels Like: {values.get("apparent_temperature", "")}{units.get("apparent_temperature", "")}
Wind Speed: {values.get("wind_speed_10m", "")}{units.get("wind_speed_10m", "")} {wind_direction}
Humidity: {data["current"]["relative_humidity_2m"]}%
"""

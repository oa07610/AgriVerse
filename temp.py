import requests
import folium
import pandas as pd
from folium.plugins import HeatMap
from datetime import datetime
from branca.colormap import LinearColormap
# API Key and Base URL
WEATHER_API_KEY = "1ca81ccfdab94fa68ba181828252002"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1/current.json"

# Major cities in Pakistan
cities = [
    # Major Metropolitan Cities
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479},
    {"name": "Rawalpindi", "lat": 33.5973, "lon": 73.0479},
    {"name": "Faisalabad", "lat": 31.4167, "lon": 73.0833},
    
    # Punjab Region
    {"name": "Multan", "lat": 30.1575, "lon": 71.5249},
    {"name": "Gujranwala", "lat": 32.1617, "lon": 74.1883},
    {"name": "Sialkot", "lat": 32.4927, "lon": 74.5319},
    {"name": "Bahawalpur", "lat": 29.3956, "lon": 71.6836},
    {"name": "Sargodha", "lat": 32.0836, "lon": 72.6711},
    {"name": "Sheikhupura", "lat": 31.7167, "lon": 73.9850},
    {"name": "Sahiwal", "lat": 30.6706, "lon": 73.1064},
    {"name": "Okara", "lat": 30.8100, "lon": 73.4597},
    {"name": "Gujrat", "lat": 32.5731, "lon": 74.0750},
    {"name": "Jhang", "lat": 31.2681, "lon": 72.3181},
    {"name": "Chiniot", "lat": 31.7200, "lon": 72.9789},
    {"name": "Kasur", "lat": 31.1167, "lon": 74.4500},
    
    # Sindh Region
    {"name": "Hyderabad", "lat": 25.3792, "lon": 68.3683},
    {"name": "Sukkur", "lat": 27.7052, "lon": 68.8570},
    {"name": "Larkana", "lat": 27.5598, "lon": 68.2264},
    {"name": "Mirpur Khas", "lat": 25.5276, "lon": 69.0159},
    {"name": "Nawabshah", "lat": 26.2442, "lon": 68.4100},
    {"name": "Jacobabad", "lat": 28.2769, "lon": 68.4514},
    {"name": "Shikarpur", "lat": 27.9556, "lon": 68.6382},
    {"name": "Dadu", "lat": 26.7319, "lon": 67.7747},
    
    # Khyber Pakhtunkhwa Region
    {"name": "Peshawar", "lat": 34.0151, "lon": 71.5249},
    {"name": "Mardan", "lat": 34.1951, "lon": 72.0447},
    {"name": "Mingora", "lat": 34.7717, "lon": 72.3600},
    {"name": "Kohat", "lat": 33.5869, "lon": 71.4414},
    {"name": "Abbottabad", "lat": 34.1688, "lon": 73.2215},
    {"name": "Bannu", "lat": 32.9889, "lon": 70.6056},
    {"name": "Dera Ismail Khan", "lat": 31.8314, "lon": 70.9019},
    {"name": "Swabi", "lat": 34.1167, "lon": 72.4667},
    {"name": "Nowshera", "lat": 34.0153, "lon": 71.9747},
    
    # Balochistan Region
    {"name": "Quetta", "lat": 30.1798, "lon": 66.9750},
    {"name": "Turbat", "lat": 26.0031, "lon": 63.0544},
    {"name": "Khuzdar", "lat": 27.8119, "lon": 66.6164},
    {"name": "Gwadar", "lat": 25.1264, "lon": 62.3225},
    {"name": "Zhob", "lat": 31.3417, "lon": 69.4486},
    {"name": "Chaman", "lat": 30.9210, "lon": 66.4597},
    {"name": "Hub", "lat": 24.9180, "lon": 66.8972},
    {"name": "Sibi", "lat": 29.5430, "lon": 67.8772},
    
    # Gilgit-Baltistan Region
    {"name": "Gilgit", "lat": 35.9208, "lon": 74.3089},
    {"name": "Skardu", "lat": 35.3354, "lon": 75.5519},
    {"name": "Chilas", "lat": 35.4167, "lon": 74.1000},
    {"name": "Hunza", "lat": 36.3167, "lon": 74.6500},
    {"name": "Ghizer", "lat": 36.1500, "lon": 73.8667},
    
    # Azad Kashmir Region
    {"name": "Muzaffarabad", "lat": 34.3597, "lon": 73.4714},
    {"name": "Mirpur", "lat": 33.1516, "lon": 73.7510},
    {"name": "Kotli", "lat": 33.5156, "lon": 73.9019},
    {"name": "Bhimber", "lat": 32.9747, "lon": 74.0747},
    {"name": "Rawalakot", "lat": 33.8511, "lon": 73.7600},
    
    # Additional Small Cities/Towns
    {"name": "Taxila", "lat": 33.7456, "lon": 72.7989},
    {"name": "Mansehra", "lat": 34.3333, "lon": 73.2000},
    {"name": "Nankana Sahib", "lat": 31.4500, "lon": 73.7000},
    {"name": "Chakwal", "lat": 32.9300, "lon": 72.8500},
    {"name": "Khanewal", "lat": 30.3017, "lon": 71.9321},
    {"name": "Jhelum", "lat": 32.9333, "lon": 73.7333},
    {"name": "Sadiqabad", "lat": 28.3006, "lon": 70.1302},
    {"name": "Kamalia", "lat": 30.7258, "lon": 72.6447},
    {"name": "Mandi Bahauddin", "lat": 32.5861, "lon": 73.4917},
    {"name": "Attock", "lat": 33.7667, "lon": 72.3667}
]

def get_temperature(city):
    try:
        params = {"key": WEATHER_API_KEY, "q": f"{city['lat']},{city['lon']}"}
        response = requests.get(WEATHER_BASE_URL, params=params)
        data = response.json()

        if "error" in data:
            print(f"Error fetching weather for {city['name']}")
            return None

        temp_c = data["current"]["temp_c"]
        return temp_c
    except Exception as e:
        print(f"API Error for {city['name']}: {e}")
        return None

# Fetch temperatures for all cities
weather_data = []
for city in cities:
    temp = get_temperature(city)
    if temp is not None:
        weather_data.append({"name": city["name"], "lat": city["lat"], "lon": city["lon"], "temp": temp})

# Create a DataFrame
df = pd.DataFrame(weather_data)

# Create a map centered at Pakistan with a light background
pakistan_map = folium.Map(location=[30.3753, 69.3451], zoom_start=5, 
                         tiles='cartodbpositron')

# Create custom colormap
colors = ['#4BA375', '#9BB85D', '#CFB756', '#D6934E', '#BC6844']
gradient = LinearColormap(
    colors=colors,
    vmin=df['temp'].min(),
    vmax=df['temp'].max(),
    caption='Temperature (°C)'
)
gradient.add_to(pakistan_map)

# Modify HeatMap settings with string keys for gradient
heat_data = [[row["lat"], row["lon"], row["temp"]] for index, row in df.iterrows()]
HeatMap(
    heat_data,
    radius=35,
    blur=25,
    max_zoom=10,
    gradient={
        '0.2': '#4BA375',    # Cool green
        '0.4': '#9BB85D',    # Light green
        '0.6': '#CFB756',    # Yellow
        '0.8': '#D6934E',    # Light brown
        '1.0': '#BC6844'     # Darker brown
    },
    min_opacity=0.5
).add_to(pakistan_map)

# Add temperature labels
for index, row in df.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.DivIcon(html=f"""
            <div style='font-size: 12px; 
                       font-weight: bold; 
                       color: #2D2D2D; 
                       background-color: rgba(255,255,255,0.8);
                       padding: 2px 4px;
                       border-radius: 3px;
                       box-shadow: 1px 1px 1px rgba(0,0,0,0.3);'
            >{int(row['temp'])}°</div>
        """)
    ).add_to(pakistan_map)

# Updated legend
legend_html = """
<div style="position: fixed; 
            bottom: 50px; 
            left: 50px; 
            width: 150px;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            font-size: 12px;
            font-family: Arial;">
    <p style="margin: 0 0 5px 0;"><strong>Temperature (°C)</strong></p>
    <div style="display: flex; align-items: center; margin: 2px 0;">
        <div style="background: #4BA375; width: 20px; height: 15px; margin-right: 5px;"></div>
        <span>< 10°C</span>
    </div>
    <div style="display: flex; align-items: center; margin: 2px 0;">
        <div style="background: #9BB85D; width: 20px; height: 15px; margin-right: 5px;"></div>
        <span>10-15°C</span>
    </div>
    <div style="display: flex; align-items: center; margin: 2px 0;">
        <div style="background: #CFB756; width: 20px; height: 15px; margin-right: 5px;"></div>
        <span>15-20°C</span>
    </div>
    <div style="display: flex; align-items: center; margin: 2px 0;">
        <div style="background: #D6934E; width: 20px; height: 15px; margin-right: 5px;"></div>
        <span>20-25°C</span>
    </div>
    <div style="display: flex; align-items: center; margin: 2px 0;">
        <div style="background: #BC6844; width: 20px; height: 15px; margin-right: 5px;"></div>
        <span>> 25°C</span>
    </div>
</div>
"""
pakistan_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map
map_filename = f"pakistan_temperature_map_{datetime.today().strftime('%Y-%m-%d')}.html"
pakistan_map.save(map_filename)

print(f"Heatmap with natural color scheme saved as {map_filename}. Open it in a browser to view.")
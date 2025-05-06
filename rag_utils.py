from groq import Groq
from dotenv import load_dotenv
import os
from supabase_client import supabase
import requests
# import whisper

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def sql_from_nl(question: str) -> str:
    # Translate natural language to SQL for your specific tables and columns
    prompt = (
        "Translate this user question into a single Supabase SQL query. "
        "Use the crop tables cottonmaster, wheatmaster, maizemaster, sugarmaster (columns: date, product_id, product_en, by_product_id, by_product_en, province_id, province_en, district_id, district_en, station_id, station_en, minimum, maximum). "
        "Also you can reference external factors tables: pakistan_inflation_rate_cpi (Date, GDP (Billions of US$), Per Capita (US $)), crude_oil (Date, Price, Open, High, Low, Vol., Change %), usd_pkr_historical_data (Date, Price, Open, High, Low, Vol., Change %), petrol_prices (Date, Petrol Price (PKR)). "
        "Only return a valid SQL SELECT statement, selecting relevant fields and filtering by user intent."f"Q: {question}SQL:" )
    
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=256
    )
    return resp.choices[0].message.content.strip()

def run_sql(query: str) -> list[dict]:
    # Use Edge Function or RPC to execute arbitrary SQL securely
    result = supabase.rpc('execute_sql', {'sql': query}).execute()
    return result.data

def summarize_data(data: list[dict], question: str) -> str:
    # Convert rows to JSON-like string
    data_str = "".join(str(row) for row in data)
    prompt = (f"Given this data:{data_str}Answer the question: {question}" )
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=int(512)
    )
    return resp.choices[0].message.content

WEATHER_BASE = 'http://api.weatherapi.com/v1/forecast.json'

def get_weather_for_date(lat: float, lon: float, date: str) -> float:
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date,
        'end_date': date,
        'daily': 'temperature_2m_max,precipitation_sum',
        'timezone': 'UTC'
    }
    r = requests.get(WEATHER_BASE, params=params)
    j = r.json()
    # Example: return max temperature
    return j['daily']['temperature_2m_max'][0]

def upsert_weather(lat, lon, date, province):
    temp = get_weather_for_date(lat, lon, date)
    supabase.table('external_factors').upsert({
        'date': date,
        'factor_name': province + '_temp_max',
        'value': temp
    }).execute()

# temp_model = whisper.load_model('base')

# def transcribe_audio(audio_bytes: bytes) -> str:
#     # save to temp file or stream
#     with open('temp.mp3','wb') as f: f.write(audio_bytes)
#     result = temp_model.transcribe('temp.mp3')
#     return result['text']

def ask_sql_rag(question: str) -> str:
    sql = sql_from_nl(question)
    rows = run_sql(sql)
    if not rows:
        return "No data found for your query."
    return summarize_data(rows, question)
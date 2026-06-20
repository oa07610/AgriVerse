import os
import traceback
from groq import Groq
from supabase_client import supabase
from dotenv import load_dotenv
import json  # Added for JSON handling
from typing import List, Dict, Tuple  # For type hints
import re


# Load .env
load_dotenv()

# —————— Configuration ——————
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "llama3-70b-8192")
TEMPERATURE     = float(os.getenv("TEMPERATURE", 0.3))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", 1024))

# —————— Clients ——————
client = Groq(api_key=GROQ_API_KEY)

# —————— System Prompt ——————
SYSTEM_PROMPT = """
You are AgriBot, a friendly and knowledgeable agricultural assistant specializing in Pakistani agriculture. While your primary expertise is in agriculture, you're also capable of engaging in general conversation and providing helpful information on various topics.

Your core expertise includes:
- Crop management (cotton, wheat, maize, sugar)
- Market prices and trends
- Agricultural best practices
- Economic factors affecting agriculture

You have access to real data about:
- Crop prices across Pakistani provinces and districts
- Crude oil prices affecting farm operations
- Petrol prices impacting transportation costs
- USD-PKR exchange rates affecting imported inputs
- Economic indicators like GDP and inflation

When users ask for specific data points, prices, trends, or statistics, you'll retrieve and analyze this information automatically. For general agricultural questions, you'll respond with expert guidance based on Pakistani farming practices.

For non-agricultural topics:
- Engage naturally in general conversation
- Share knowledge about Pakistani culture, traditions, and daily life
- Discuss weather, local events, and community matters
- Provide helpful information about various topics while maintaining a friendly tone

Always respond in a helpful, accurate manner. Be conversational and warm, but concise. Never reveal your internal data retrieval process or technical details. Engage in conversations with the user and do what you can to help them, whether it's about agriculture or general topics.
"""

# —————— Logger Setup ——————
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('agribot')

# —————— Database Schema Knowledge ——————
DB_SCHEMA = """
STRICT DATABASE SCHEMA (case-sensitive, never modify):
- cottonmaster(date, by_product, province, district, station, price)
- wheatmaster(date, by_product, province, district, station, price) 
- maizemaster(date, by_product, province, district, station, price)
- sugarmaster(date, by_product, province, district, station, price)
- crude_oil(Date, Price)
- petrol_prices(Date, Petrol_Price_PKR)
- usd_pkr_historical_data(Date, Price)
- inflation_rate(Date, GDP_Billions_USD, Per_Capita_USD)
"""

# ---------- Intent Classification ----------
def classify_intent(question: str) -> str:
    """Enhanced intent classification with improved agricultural context awareness and caching"""
    # Cache for frequently asked questions
    if not hasattr(classify_intent, 'cache'):
        classify_intent.cache = {}
    
    # Check cache first
    if question in classify_intent.cache:
        return classify_intent.cache[question]
    
    question_lower = question.lower()
    
    # Enhanced keyword sets with agricultural context
    data_keywords = {
        'price', 'cost', 'rate', 'market', 'trend', 'forecast', 'statistics',
        'yield', 'production', 'harvest', 'acreage', 'hectares', 'tonnes',
        'export', 'import', 'trade', 'demand', 'supply', 'stock', 'average',
        'maximum', 'minimum', 'highest', 'lowest', 'compare', 'difference',
        'trend', 'over time', 'growth', 'decline', 'increase', 'decrease'
    }
    
    conversation_keywords = {
        'how to', 'what is', 'when to', 'where to', 'why', 'explain',
        'guide', 'help', 'advice', 'suggestion', 'recommendation',
        'best practice', 'method', 'technique', 'process'
    }
    
    weather_keywords = {
        'weather', 'climate', 'rain', 'temperature', 'humidity',
        'forecast', 'season', 'drought', 'flood', 'irrigation'
    }
    
    # Enhanced classification logic with priority and context
    if any(keyword in question_lower for keyword in weather_keywords):
        classify_intent.cache[question] = "weather"
        return "weather"
        
    if any(keyword in question_lower for keyword in data_keywords):
        # Check for stronger conversational signals
        if any(c_keyword in question_lower for c_keyword in conversation_keywords):
            # Look for specific numeric indicators
            numeric_indicators = ["how much", "how many", "price of", "cost of", "rate of"]
            if any(indicator in question_lower for indicator in numeric_indicators):
                classify_intent.cache[question] = "database"
                return "database"
            else:
                classify_intent.cache[question] = "conversation"
                return "conversation"
        classify_intent.cache[question] = "database"
        return "database"
    
    # Default to conversation for general agricultural queries
    classify_intent.cache[question] = "conversation"
    return "conversation"

def format_response(answer: str, question: str) -> str:
    """Enhanced response formatting with contextual relevance"""
    question_lower = question.lower()
    
    # Extract commodity and location context for more relevant formatting
    commodity_context = None
    for commodity in ["cotton", "wheat", "maize", "sugar", "crude oil", "petrol", "dollar"]:
        if commodity in question_lower:
            commodity_context = commodity
            break
            
    location_context = None
    for location in ["punjab", "sindh", "kpk", "khyber pakhtunkhwa", "balochistan", 
                     "lahore", "karachi", "islamabad", "peshawar", "quetta", "multan", 
                     "faisalabad", "rawalpindi", "hyderabad", "sukkur", "sialkot",
                     "badin", "dadu", "ghotki", "jamshoro", "khairpur", "mirpur khas",
                     "naushro feroze", "nawabshah", "sanghar", "umarkot", "vehari",
                     "rahim yar khan", "ahmadpur east", "arifwala", "burewala", "chichawatni",
                     "dera ghazi khan", "faqirwali", "fortabbas", "hasilpur", "jhang",
                     "kahror pacca", "mailsi", "mian channu", "mianwali", "pakpattan",
                     "sadiqabad", "samundari", "shorkot", "toba tek singh", "yazman mandi",
                     "khuzdar", "lasbela", "winder", "khadro", "khairpur nathan shah",
                     "madi sadiq ganj", "minchinabad", "okara", "digri", "kunri",
                     "mirpur sakro", "tando allahyar", "tando muhammad khan", "thatta",
                     "naukot", "uthal", "khanpur mahar", "jhudo", "golarchi taluka",
                     "hala", "matiari", "pir mahal", "alipur", "kasur", "khipro",
                     "sinjhoro", "mureed wala", "rajanpur", "nauabad", "fazil pur",
                     "layyah", "piplan", "qazi ahmed", "bhakkar", "gojra", "bhawana",
                     "jehanian", "qaboola", "shah jeewna", "tando jan muhammad",
                     "shaher sultan", "gaggo mandi", "kabirwala", "garh maharaja",
                     "sillanwali", "tando ghulam muhammad", "malakwal", "garh more",
                     "johi", "jhol", "daur", "taunsa sharif", "basti maluk", "dunyapur",
                     "bhit shah", "makhdoom pur pahoran", "ahmedpur sial", "kumb",
                     "tando ghulam ali", "rasoolabad", "kot addu", "kot ghulam muhammad",
                     "matli", "moro", "renala khurd", "sarhari", "ada flour", "chondko",
                     "dhoronaro", "kot lalu", "mongi bangla", "saeedabad", "tharu shah",
                     "dahranwala", "maqsoodo rind", "sakrand", "chiniot", "head bakkani",
                     "marot", "setharja", "pithoro", "dipalpur", "hingorja", "madrassa",
                     "akri", "kassowal", "pull bagar", "bandhi", "shadan lund", "hub",
                     "gambat", "khichi wala", "mitro", "bucheri", "karoondi", "ranipur",
                     "sui", "chak jhumra", "qutabpur", "odero lal", "new saeedabad",
                     "kandiaro", "thengmore", "uch sharif", "chowdagi", "mamu kanjan",
                     "ghazi ghat", "jampur", "tando bagho", "jalalpur sharif",
                     "kamalia", "karor lal esan", "kot banglo", "mirpur batharo",
                     "karam pur", "gupchani", "pano aqil", "noorpur thal",
                     "muhammadpur dewan", "mirpur mathelo", "jatoi", "shah jamal",
                     "chini goth", "mohenjo-daro", "kot sabzal", "tibba sultanpur",
                     "firoza", "rohillanwali", "turbat", "daulatpur", "chowk munda",
                     "muzaffargarh", "kot samaba", "pattoki", "isa khel",
                     "tandlianwala", "chunian", "ghaziabad", "khider wala",
                     "kacha khuh", "panjmoro", "sargodha", "chowk maitla", "sarhad",
                     "choti zareen", "kotla musa khan", "bhiria city", "halani",
                     "faisalabad", "kharan", "kot diji", "chashma", "takht mahal",
                     "adilpur", "qaimpur", "fatehpur", "sultanabad", "bela", "talhar",
                     "jam sahib", "khairpur tamewali", "mando dero", "sibi", "gharo",
                     "alipur chatta", "abdul hakim", "janpur", "pir wasan",
                     "darya khan", "bagh", "chuhar jamali", "perumal", "barkhan",
                     "rojhan", "wadh", "nal", "dajal", "kalur kot", "samaro", "sujawal",
                     "jati", "rajo khanani", "kech", "chowk azam", "kot mithan",
                     "d.i.khan", "kadhan", "kandiari", "dureji", "kundian", "lakhra",
                     "faqirabad", "zehri", "shahpur jahania", "chak", "bhalwal",
                     "nurpur", "dalbandin", "sheikhupura"]:
        if location in question_lower:
            location_context = location.replace("kpk", "Khyber Pakhtunkhwa").title()
            break
    
    # Extract query type for better response formatting
    query_type = "information"
    if "price" in question_lower or "cost" in question_lower or "rate" in question_lower:
        query_type = "price"
    elif "trend" in question_lower or "over time" in question_lower:
        query_type = "trend"
    elif "compare" in question_lower or "difference" in question_lower:
        query_type = "comparison"
    elif "lowest" in question_lower or "minimum" in question_lower:
        query_type = "minimum"
    elif "highest" in question_lower or "maximum" in question_lower:
        query_type = "maximum"
    
    # Build a context-aware formatting prompt
    prompt = f"""
    As AgriBot, refine this answer about {commodity_context or "agriculture"} in {location_context or "Pakistan"} to be:
    
    Original question: "{question}"
    Query type: {query_type}
    
    Formatting guidelines:
    1. Start with the most relevant data point answering the specific question
    2. Always include units with prices (PKR/kg, PKR/40kg, etc.)
    3. For prices, include which specific market/by-product this applies to
    4. Add ONE relevant market insight specific to {location_context or "this region"}
    5. End with ONE practical farming/trading recommendation based directly on this data
    6. End with local greeting or encouragement and be friendly
    
    DO NOT:
    - Mention data sources or technical terms
    - Use generic farming tips unrelated to the price data
    - Say "no variations observed" unless specifically analyzing variations
    - Use bullet points unless comparing multiple items
    
    Current response: {answer}
    
    Refined response (3-4 sentences max):
    """
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

def format_response_conversation(answer: str, question: str) -> str:
    """Enhanced response formatting with a conversational tone"""
    question_lower = question.lower()
    
    # Extract commodity and location context for more conversational relevance
    commodity_context = None
    for commodity in ["cotton", "wheat", "maize", "sugar", "crude oil", "petrol", "dollar"]:
        if commodity in question_lower:
            commodity_context = commodity
            break
            
    location_context = None
    for location in ["punjab", "sindh", "kpk", "khyber pakhtunkhwa", "balochistan", 
                     "lahore", "karachi", "islamabad", "sukkur", "multan", "faisalabad"]:
        if location in question_lower:
            location_context = location.replace("kpk", "Khyber Pakhtunkhwa").title()
            break
    
    # Build a conversational formatting prompt
    prompt = f"""
    As AgriBot, refine this answer to make it more conversational and engaging for the user.
    
    Original question: "{question}"
    Context: {commodity_context or "general agriculture"} in {location_context or "Pakistan"}
    
    Guidelines:
    1. Start with a friendly and engaging opening that acknowledges the user's question.
    2. Provide the answer in a conversational tone, keeping it simple and easy to understand.
    3. If relevant, include a practical tip or insight that adds value to the user's query.
    4. End with a warm and encouraging closing, inviting further questions if needed.
    
    DO NOT:
    - Include starting sentences like "Here's the refined response..."
    - Use overly technical language or jargon.
    - Include unnecessary details or repeat the question verbatim.
    - Be overly formal; keep the tone friendly and approachable.
    
    Current response: {answer}
    
    Refined conversational response:
    """
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content


# ---------- NL to SQL Conversion ----------
def sql_from_nl(question: str) -> str:
    """Enhanced SQL generator with advanced location and by-product awareness"""
    question_lower = question.lower()

    # Temporal analysis
    years_pattern = re.search(r'last (\d+) years', question_lower)
    date_filter = ""
    if years_pattern:
        years = int(years_pattern.group(1))
        date_filter = f" WHERE Date >= CURRENT_DATE - INTERVAL '{years} years'"
    
    # Define commodity mappings with by-products
    commodity_mappings = {
        "cotton": {
            "table": "cottonmaster",
            "by_products": ["phutti", "lint_cotton"]
        },
        "wheat": {
            "table": "wheatmaster",
            "by_products": []  # Add wheat by-products here
        },
        "maize": {
            "table": "maizemaster",
            "by_products": []  # Add maize by-products here
        },
        "sugar": {
            "table": "sugarmaster",
            "by_products": ["jodia_bazaar", "akbari_mandi", "sugar_mills", "mill_gate"]
        }
    }
    
    # Other tables without by-products
    other_tables = {
        "crude oil": "crude_oil",
        "petrol": "petrol_prices",
        "usd": "usd_pkr_historical_data",
        "dollar": "usd_pkr_historical_data",
        "inflation": "inflation_rate"
    }
    
    # Operation detection with context
    operation_map = {
        "average": "AVG", "avg": "AVG", "mean": "AVG",
        "max": "MAX", "highest": "MAX", "maximum": "MAX", "most expensive": "MAX",
        "min": "MIN", "lowest": "MIN", "minimum": "MIN", "cheapest": "MIN", "least expensive": "MIN",
        "sum": "SUM", "total": "SUM"
    }
    
    # Location hierarchies
    location_hierarchy = {
        "province": ["punjab", "sindh", "khyber pakhtunkhwa", "kpk", "balochistan"],

        "district": ['badin', 'dadu', 'ghotki', 'hyderabad', 'jamshoro', 'khairpur',
       'mirpur khas', 'naushro feroze', 'nawabshah', 'sanghar', 'sukkur',
       'bahawalnagar', 'bahawalpur', 'khanewal', 'rahim yar khan',
       'lodhran', 'multan', 'umarkot', 'vehari', 'pakpattan', 'sahiwal',
       'dera ghazi khan', 'jhang', 'mianwali', 'faisalabad',
       'toba tek singh', 'kalat', 'lasbella', 'okara', 'thatta',
       'tando muhammad khan', 'tharparkar', 'matiari', 'muzaffargarh',
       'kasur', 'rajan pur', 'layyah', 'bhakkar', 'chiniot', 'sargodha',
       'mandi bahauddin', 'new mirpur', 'gujranwala', 'dera bughti',
       'jhelum', 'khushab', 'larkana', 'kech(turbat)', 'karachi',
       'kharan', 'sibi', 'barkhan', 'khuzdar', 'dera ismail khan',
       'shikarpur', 'chagai', 'sheikhupura', 'lahore city', 'peshawar',
       'mardan', 'rawalpindi', 'chakwal', 'gujrat', 'sialkot',
       'nankana sahib', 'hafizabad', 'narowal', 'attock', 'haripur',
       'jacobabad'],

        "station":['Badin', 'Dadu', 'Ghotki', 'Hyderabad', 'Nooriabad', 'Khairpur',
       'Mirpur Khas', 'Naushahro Feroze', 'Nawab Shah', 'Sanghar',
       'Sukkur', 'Bahawalnagar', 'Bahawalpur', 'Chishtian', 'Dunga Bunga',
       'Haroonabad', 'Khanewal', 'Khanpur', 'Liaqatpur', 'Lodhran',
       'Shujabad', 'Daharki', 'Kotri', 'Mehrabpur', 'Rohri', 'Salehput',
       'Shahdadpur', 'Shahpur Chakar', 'Tando Adam Khan', 'Umarkot',
       'Multan', 'Vehari', 'Rahim Yar Khan', 'Ahmadpur East', 'Arifwala',
       'Burewala', 'Chichawatni', 'Dera Ghazi Khan', 'Faqirwali',
       'Fortabbas', 'Hasilpur', 'Jhang', 'Kahror Pacca', 'Mailsi',
       'Mian Channu', 'Mianwali', 'Pakpattan', 'Sadiqabad', 'Sahiwal',
       'Samundari', 'Shorkot', 'Toba Tek Singh', 'Yazman Mandi',
       'Khuzdar', 'Lasbela', 'Winder', 'Khadro', 'Khairpur Nathan Shah',
       'Mandi Sadiq Ganj', 'Minchinabad', 'Okara', 'Digri', 'Kunri',
       'Mirpur Sakro', 'Tando Allahyar', 'Tando Muhammad Khan', 'Thatta',
       'Naukot', 'Uthal', 'Khanpur Mahar', 'Jhudo', 'Golarchi Taluka',
       'Hala', 'Matiari', 'Pir Mahal', 'Alipur', 'Kasur', 'Khipro',
       'Sinjhoro', 'Mureed Wala', 'Rajanpur', 'Nauabad', 'Fazil Pur',
       'Layyah', 'Piplan', 'Qazi Ahmed', 'Bhakkar', 'Gojra', 'Bhawana',
       'Jehanian', 'Qaboola', 'Shah Jeewna', 'Tando Jan Muhammad',
       'Shaher Sultan', 'Gaggo Mandi', 'Kabirwala', 'Garh Maharaja',
       'Sillanwali', 'Tando Ghulam Muhammad', 'Malakwal', 'Garh More',
       'Johi', 'Jhol', 'Daur', 'Taunsa sharif', 'Basti Maluk', 'Dunyapur',
       'Bhit Shah', 'Makhdoom Pur Pahoran', 'Ahmedpur Sial', 'Kumb',
       'Tando Ghulam Ali', 'Rasoolabad', 'Kot Addu',
       'Kot Ghulam Muhammad', 'Matli', 'Moro', 'Renala Khurd', 'Sarhari',
       'Ada Flour', 'Chondko', 'Dhoronaro', 'Kot Lalu', 'Mongi Bangla',
       'Saeedabad', 'Tharu Shah', 'Dahranwala', 'Maqsoodo Rind',
       'Sakrand', 'Chiniot', 'Head Bakkani', 'Marot', 'Setharja',
       'Pithoro', 'Dipalpur', 'Hingorja', 'Madrassa', 'Akri', 'Kassowal',
       'Pull Bagar', 'Bandhi', 'Shadan Lund', 'Hub', 'Gambat',
       'Khichi Wala', 'Mitro', 'Bucheri', 'Karoondi', 'Ranipur', 'Sui',
       'Chak Jhumra', 'Qutabpur', 'Odero Lal', 'New Saeedabad',
       'Kandiaro', 'Thengmore', 'Uch Sharif', 'Chowdagi', 'Mamu kanjan',
       'Ghazi Ghat', 'Jampur', 'Tando Bagho', 'Jalalpur Sharif',
       'Kamalia', 'Karor Lal Esan', 'Kot Banglo', 'Mirpur Batharo',
       'Karam Pur', 'Gupchani', 'Pano Aqil', 'Noorpur Thal',
       'Muhammadpur Dewan', 'Mirpur Mathelo', 'Jatoi', 'Shah Jamal',
       'Chani Goth', 'Mohenjo-daro', 'Kot Sabzal', 'Tibba Sultanpur',
       'Firoza', 'Rohillanwali', 'Turbat', 'Daulatpur', 'Chowk Munda',
       'Karachi', 'Tatepur', 'Ubauro', 'Noor Pur Nauranga',
       'Muzaffargarh', 'Kot Samaba', 'Pattoki', 'Isa khel',
       'Tandlianwala', 'Chunian', 'Ghaziabad', 'Khider Wala',
       'Kacha Khuh', 'Panjmoro', 'Sargodha', 'Chowk Maitla', 'Sarhad',
       'Choti Zareen', 'Kotla Musa Khan', 'Bhiria City', 'Halani',
       'Faisalabad', 'Kharan', 'Kot Diji', 'Chashma', 'Takht Mahal',
       'Adilpur', 'Qaimpur', 'Fatehpur', 'Sultanabad', 'Bela', 'Talhar',
       'Jam Sahib', 'Khairpur Tamewali', 'Mando Dero', 'Sibi', 'Gharo',
       'Alipur Chatta', 'Abdul Hakim', 'Janpur', 'Pir Wasan',
       'Darya Khan', 'Bagh', 'Chuhar Jamali', 'Perumal', 'Barkhan',
       'Rojhan', 'Wadh', 'Nal', 'Dajal', 'Kalur Kot', 'Samaro', 'Sujawal',
       'Jati', 'Rajo Khanani', 'Kech', 'Chowk Azam', 'Kot Mithan',
       'D.I.Khan', 'Kadhan', 'Kandiari', 'Dureji', 'Kundian', 'Lakhra',
       'Faqirabad', 'Zehri', 'Shahpur Jahania', 'Chak', 'Bhalwal',
       'Nurpur', 'Dalbandin', 'Sheikhupura']
    }
    
    # Components to extract
    components = {
        "table": None,
        "operation": "AVG",  # Default to average if not specified
        "province": None,
        "district": None,
        "station": None,
        "by_product": None,
        "is_price": "price" in question_lower
    }
    
    # Detect commodity and table
    for commodity, info in commodity_mappings.items():
        if commodity in question_lower:
            components["table"] = info["table"]
            # Look for by-products
            for by_product in info["by_products"]:
                if by_product.replace("_", " ") in question_lower:
                    components["by_product"] = by_product
                    break
            break
    
    # Check other tables if no commodity found
    if not components["table"]:
        for keyword, table in other_tables.items():
            if keyword in question_lower:
                components["table"] = table
                break
    
    # Detect operation (priority: average > max/min > others)
    for op_keyword, op_func in operation_map.items():
        if op_keyword in question_lower:
            components["operation"] = op_func
            break
    
    # Detect location (hierarchical approach)
    # Check for provinces
    for province in location_hierarchy["province"]:
        if province in question_lower:
            if province == "kpk":
                components["province"] = "Khyber Pakhtunkhwa"
            else:
                components["province"] = province.capitalize()
            break
    
    # Check for districts
    for district in location_hierarchy["district"]:
        if district in question_lower:
            components["district"] = district.title()
            break
    
    # Check for stations
    for station in location_hierarchy["station"]:
        station_search = station.replace("_", " ")
        if station_search in question_lower:
            components["station"] = station
            break
    
    # Build appropriate query based on components
    if components["table"] in ["cottonmaster", "wheatmaster", "maizemaster", "sugarmaster"]:
        # For min/max operations, include more details
        if components["operation"] in ["MIN", "MAX"]:
            base_query = f"""
            SELECT price AS value, by_product, province, district, station, date
            FROM {components['table']}
            """
            # This will be ordered later to get MIN/MAX with details
        else:
            # For aggregate operations like AVG
            base_query = f"SELECT {components['operation']}(price) AS value FROM {components['table']}"
        where_clauses = []
        
        # Add filters based on location hierarchy and by-product
        if components["province"]:
            where_clauses.append(f"province = '{components['province']}'")
        
        if components["district"]:
            where_clauses.append(f"district = '{components['district']}'")
        
        if components["station"]:
            where_clauses.append(f"station = '{components['station']}'")
            
        if components["by_product"]:
            where_clauses.append(f"by_product = '{components['by_product']}'")
        
        # Combine WHERE clauses if they exist
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        
        # Special handling for MIN/MAX operations to include details
        if components["operation"] in ["MIN", "MAX"] and "SELECT price AS value" in base_query:
            order_direction = "ASC" if components["operation"] == "MIN" else "DESC"
            base_query += f" ORDER BY price {order_direction} LIMIT 1"
        
        # Group by if no specific location provided, or if trends requested
        elif not (components["province"] or components["district"] or components["station"]):
            if "trend" in question_lower or "compare" in question_lower:
                base_query = f"""
                SELECT province, {components['operation']}(price) AS value 
                FROM {components['table']}
                GROUP BY province
                ORDER BY value DESC
                """
            elif "by_product" in question_lower or "by product" in question_lower:
                base_query = f"""
                SELECT by_product, {components['operation']}(price) AS value 
                FROM {components['table']}
                GROUP BY by_product
                ORDER BY value DESC
                """
        
        # Time series trends
        if "over time" in question_lower or "trend" in question_lower:
            if "monthly" in question_lower:
                base_query = f"""
                SELECT DATE_TRUNC('month', date) AS period, {components['operation']}(price) AS value 
                FROM {components['table']}
                """
                if where_clauses:
                    base_query += " WHERE " + " AND ".join(where_clauses)
                base_query += """
                GROUP BY period
                ORDER BY period DESC
                LIMIT 12
                """
            elif "yearly" in question_lower:
                base_query = f"""
                SELECT DATE_TRUNC('year', date) AS period, {components['operation']}(price) AS value 
                FROM {components['table']}
                """
                if where_clauses:
                    base_query += " WHERE " + " AND ".join(where_clauses)
                base_query += """
                GROUP BY period
                ORDER BY period DESC
                LIMIT 5
                """
        
        return base_query
        
    elif components["table"] in ["crude_oil", "petrol_prices", "usd_pkr_historical_data", "inflation_rate"]:
        # Handle simple time series tables
        operation_col = "Price"
        if components["table"] == "petrol_prices":
            operation_col = "Petrol_Price_PKR"
        elif components["table"] == "inflation_rate":
            if "gdp" in question_lower:
                operation_col = "GDP_Billions_USD"
            else:
                operation_col = "Per_Capita_USD"
                
        base_query = f"SELECT {components['operation']}({operation_col}) AS value FROM {components['table']}"
        
        # Time trends for economic indicators
        if "trend" in question_lower or "over time" in question_lower:
            base_query = f"""
            SELECT Date, {operation_col} AS value 
            FROM {components['table']}
            ORDER BY Date DESC
            LIMIT 10
            """
            
        return base_query
    
    # Fallback query - should be rare
    return "SELECT AVG(price) AS average FROM cottonmaster LIMIT 10"

# ---------- SQL Validation & Execution ----------
def execute_sql(query: str) -> Tuple[List[Dict], bool]:
    """Safe SQL execution with enhanced error handling"""
    try:
        logger.info(f"Executing: {query}")
        result = supabase.rpc("execute_sql", {"sql": query}).execute()
        return (result.data, True) if result.data else ([], False)
    except Exception as e:
        logger.error(f"SQL Error: {str(e)}")
        return ([], False)

# ---------- Data Interpretation ----------
def interpret_data(rows: List[Dict], question: str) -> str:
    """Agricultural-focused data interpretation with location and by-product awareness"""
    if not rows:
        return get_domain_knowledge(question)
    
    question_lower = question.lower()
    
    # Detect context elements in the question
    context = {
        "commodity": None,
        "by_product": None,
        "location_type": None,
        "location": None,
        "time_reference": None
    }
    
    # Detect commodity
    for commodity in ["cotton", "wheat", "maize", "sugar"]:
        if commodity in question_lower:
            context["commodity"] = commodity
            break
    
    # Detect by-products
    by_products = {
        "cotton": ["phutti", "lint_cotton"],
        "sugar": ["jodia_bazaar", "akbari_mandi", "sugar_mills", "mill_gate"]
    }
    
    if context["commodity"] and context["commodity"] in by_products:
        for by_product in by_products[context["commodity"]]:
            if by_product.replace("_", " ") in question_lower:
                context["by_product"] = by_product
                break
    
    # Location detection
    provinces = ["punjab", "sindh", "khyber pakhtunkhwa", "kpk", "balochistan"]
    for province in provinces:
        if province in question_lower:
            context["location_type"] = "province"
            context["location"] = province
            break
    
    # Districts/stations would need a comprehensive list, simplified here
    common_districts = ["lahore", "karachi", "sukkur", "peshawar", "quetta", "multan"]
    for district in common_districts:
        if district in question_lower:
            context["location_type"] = "district/station"
            context["location"] = district
            break
    
    # Time reference
    time_refs = ["current", "recent", "today", "monthly", "yearly", "last year"]
    for ref in time_refs:
        if ref in question_lower:
            context["time_reference"] = ref
            break
    
    # Build a context-aware interpretation prompt
    interpretation_prompt = f"""
    As a Pakistani agricultural expert analyzing this data for: "{question}"
    
    Data: {json.dumps(rows, indent=2)}
    
    Context detected:
    - Commodity: {context["commodity"] or "general"}
    - By-product: {context["by_product"] or "all types"}
    - Location: {context["location"] or "nationwide"} ({context["location_type"] or "general"})
    - Time reference: {context["time_reference"] or "current"}
    
    Guidelines:
    1. Start with the specific data point or trend most relevant to the question
    2. Compare to typical range for this specific commodity and location
    3. Mention any seasonal factors relevant to {context["commodity"] or "the commodity"} in {context["location"] or "Pakistan"}
    4. For by-products, explain how this affects the main commodity market
    5. Add a practical recommendation specific to {context["location"] or "Pakistani"} farmers
    
    Use local agricultural terminology where helpful.
    """
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":interpretation_prompt}],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def summarize_data(rows: List[Dict], question: str) -> str:
    """Enhanced data summarization with location and by-product context"""
    formatted_data = json.dumps(rows, indent=2)
    question_lower = question.lower()
    
    # Extract key context indicators
    commodity = None
    for c in ["cotton", "wheat", "maize", "sugar"]:
        if c in question_lower:
            commodity = c
            break
    
    location = "Pakistan"
    for loc in ["punjab", "sindh", "kpk", "khyber pakhtunkhwa", "balochistan"]:
        if loc in question_lower:
            location = loc.capitalize()
            if loc == "kpk":
                location = "Khyber Pakhtunkhwa"
            break
    
    # Also check for districts and stations
    for district in ["lahore", "karachi", "sukkur", "peshawar", "quetta", "multan", "faisalabad"]:
        if district in question_lower:
            location = district.title()
            break
    
    by_product_context = ""
    by_product = None
    cotton_by_products = ["phutti", "lint_cotton"]
    sugar_by_products = ["jodia_bazaar", "akbari_mandi", "sugar_mills", "mill_gate"]
    
    for bp in cotton_by_products + sugar_by_products:
        if bp.replace("_", " ") in question_lower:
            by_product = bp
            by_product_context = f" for {bp.replace('_', ' ')}"
            break
    
    # Determine units based on commodity
    units = {
        "cotton": "PKR/40kg",
        "wheat": "PKR/40kg",
        "maize": "PKR/40kg",
        "sugar": "PKR/kg"
    }
    
    # Extract operation type (min, max, avg)
    operation_type = "current"
    if "lowest" in question_lower or "minimum" in question_lower:
        operation_type = "lowest"
    elif "highest" in question_lower or "maximum" in question_lower:
        operation_type = "highest"
    elif "average" in question_lower or "mean" in question_lower:
        operation_type = "average"
        
    # Create a contextual prompt
    prompt = f"""
    Analyze this data for: "{question}"
    Data: {formatted_data}
    
    Provide a focused agricultural summary about {commodity or "crops"} in {location}{by_product_context}.
    
    Response must have:
    1. Direct answer as your first sentence including: 
       - The exact {operation_type} price from the data 
       - Proper units ({units.get(commodity, "PKR")})
       - Date of the price (if available)
       - The specific by-product name if applicable
    2. ONE market insight relevant to this specific location and commodity
    3. ONE practical recommendation for farmers or traders based specifically on this price data
    
    Example format: "The lowest sugar price in Sukkur is 130 PKR/kg for mill_gate sugar as of May 2, 2025. This price is 15% below the national average, indicating good buying opportunity for local traders. Farmers should consider timing their harvest to match this market low point for better profitability."
    
    Use farming terminology appropriate for Pakistani context.
    Maximum 3-4 sentences total with specific, actionable information.
    """
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=250
    )
    return response.choices[0].message.content

# ---------- Streamlined Domain Knowledge ----------
def get_domain_knowledge(question: str) -> str:
    """Precise, actionable advice without fluff"""
    prompt = f"""
    As Pakistani agriculture expert, answer concisely:
    "{question}"
    
    Structure:
    - 1-2 sentence core answer
    - 1-2 regional examples
    - 1-2 practical tip
    
    Example:
    Input: "Best crops for Sindh?"
    Output: "Sindh's climate suits rice and cotton. Example: Thatta district's 
    successful IRRI-6 rice yields. Tip: Use laser leveling for water efficiency."
    """
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content

# ---------- Main Handler ----------
def ask_agri_bot(question: str) -> str:
    """Enhanced orchestration for the complete query handling process"""
    # Handle greeting patterns
    greetings = ["hello", "hi", "good morning", "hey", "salam", "assalam-o-alaikum", "aoa", "Assalamu Alaikum"]
    greeting_pattern = re.compile(r'^(?:' + '|'.join(re.escape(g) for g in greetings) + r')[\s!,.?]*$', re.IGNORECASE)
    if greeting_pattern.match(question.strip()):
        return "Assalam-o-Alaikum! I'm AgriBot. How can I assist with Pakistani agriculture today?"
    
    # Handle thank you patterns
    thanks = ["thank", "thanks", "shukriya", "meherbani", "shukran"]
    if all(word in question.lower() for word in thanks) and len(question.split()) <= 5:
        return "You're welcome! Feel free to ask any other questions about Pakistani agriculture."
    
    try:
        # Intent classification
        intent = classify_intent(question)
        logger.info(f"Classified intent: {intent}")
        
        if intent == "conversation":
            # Handle pure conversational queries without database
            knowledge = get_domain_knowledge(question)
            return format_response_conversation(knowledge, question)
            
        # Data handling flow
        logger.info("Generating SQL from natural language")
        sql = sql_from_nl(question)
        logger.info(f"Generated SQL: {sql}")
        
        # Execute query
        data, success = execute_sql(sql)
        
        if not success or not data:
            logger.warning("No data returned or query failed")
            # Enhanced fallback with domain knowledge
            fallback_response = f"I don't have the specific data you're looking for. However, here's some general information: {get_domain_knowledge(question)}"
            return format_response(fallback_response, question)
            
        # Log data response
        logger.info(f"Data returned: {len(data)} rows")
        
        # Enhanced data analysis based on query complexity
        if "trend" in question.lower() or "compare" in question.lower() or len(data) > 5:
            analysis = interpret_data(data, question)
        else:
            analysis = summarize_data(data, question)
            
        # Final formatting for user-friendly response
        return format_response(analysis, question)
        
    except Exception as e:
        logger.error(f"Handler Error: {str(e)}")
        error_trace = traceback.format_exc()
        logger.error(f"Traceback: {error_trace}")
        return "Apologies, I'm having temporary difficulties processing your request. Please try rephrasing your question or ask something different."
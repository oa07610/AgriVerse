from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from datetime import date, datetime, timedelta
from flask_cors import CORS
import google.generativeai as genai
from googletrans import Translator
from deep_translator import GoogleTranslator
import requests
import random
import base64
import os
from functools import wraps
import folium
from folium import Map, CircleMarker, Marker, Tooltip, Icon
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
import branca
import json
import math
import secrets
from supabase_client import supabase
from dotenv import load_dotenv
from rag_utils import ask_sql_rag, upsert_weather

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.secret_key = app.config['SECRET_KEY']

# Configure the Gemini API key
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_KEY)

# Whisper API configuration
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
WHISPER_HEADERS = {"Authorization": "Bearer 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"}

# WeatherAPI configuration
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
WEATHER_BASE_URL = os.getenv('WEATHER_BASE_URL', 'https://api.weatherapi.com/v1/current.json')

# Initialize the translator
translator = Translator()

# File to store user data
SUGAR_ACTUAL_FILE = 'data/final_sugar_province_actual.csv'
SUGAR_PRED_FILE = 'data/final_sugar_province_pred_formatted.csv'
MAIZE_ACTUAL_FILE = 'data/maize_actual.csv'
MAIZE_PRED_FILE = 'data/maize_pred.csv'
COTTON_ACTUAL_FILE = 'data/cotton_actual.csv'
COTTON_PRED_FILE = 'data/cotton_pred.csv'
WHEAT_ACTUAL_FILE = 'data/wheat_actual.csv'
WHEAT_PRED_FILE = 'data/wheat_pred.csv'

#email verification
EMAIL_VERIFICATION_API_KEY = os.getenv('EMAIL_VERIFICATION_API_KEY')

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'agriverseofficial@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')  # Replace with your email password
mail = Mail(app)

# Update the UPLOAD_FOLDER to be an absolute path

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'danger')
            return redirect(url_for('login'))
        # fetch user from supabase
        resp = supabase.from_('user').select('*').eq('id', session['user_id']).single().execute()
        user = resp.data
        if not user or not user.get('is_admin'):
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

@app.route('/admin')
@admin_required
def admin_dashboard():
    # Fetch newsletter posts ordered by creation timestamp descending
    resp = (supabase
            .from_('newsletter_post')
            .select("*, author:user(username)")
            .order('created_at', desc=True)
            .execute())
    posts = resp.data or []
    return render_template('admin/admin_dashboard.html', posts=posts)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/admin/create_post', methods=['GET', 'POST'])
@admin_required
def create_post():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        if not title or not content:
            flash('Title and content are required.', 'danger')
            return redirect(url_for('create_post'))
        image_url = video_url = None
        for field in ('image', 'video'):
            file = request.files.get(field)
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                path = os.path.join(UPLOAD_FOLDER, unique)
                file.save(path)
                if field == 'image':
                    image_url = f'/static/uploads/{unique}'
                else:
                    video_url = f'/static/uploads/{unique}'
        payload = {
            'title': title,
            'content': content,
            'image_url': image_url,
            'video_url': video_url,
            'author_id': session['user_id']
        }
        resp = supabase.from_('newsletter_post').insert(payload).execute()
        # Check insertion success via resp.data
        if not getattr(resp, 'data', None):
            flash('Error creating post. Please try again.', 'danger')
            return redirect(url_for('create_post'))
        flash('Post created successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/create_post.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/newsletter')
def newsletter():
    resp = (supabase
            .from_('newsletter_post')
            .select("*, author:user(username)")
            .order('created_at', desc=True)
            .execute())
    posts = resp.data if resp.data else []
    return render_template('newsletter.html', posts=posts)

TABLE_MAP = {
    'wheat':  'wheatmaster',
    'maize':  'maizemaster',
    'cotton': 'cottonmaster',
    'sugar':  'sugarmaster'
}

@app.route('/admin/upload_csv', methods=['GET', 'POST'])
@admin_required
def upload_csv():
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or not file.filename.lower().endswith('.csv'):
            flash("Please upload a valid CSV file.", "danger")
            return redirect(url_for('admin_dashboard'))

        crop_type = request.form.get('crop_type')
        if crop_type not in TABLE_MAP:
            flash("Unknown crop type selected.", "danger")
            return redirect(url_for('admin_dashboard'))
        table_name = TABLE_MAP[crop_type]

        # Save temp
        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)

        # Read CSV
        try:
            df_new = pd.read_csv(temp_path)
        except Exception as e:
            os.remove(temp_path)
            flash(f"Error reading CSV: {e}", "danger")
            return redirect(url_for('admin_dashboard'))

        # Sanitize every cell: convert numpy types, drop non-finite floats
        records = []
        for row in df_new.to_dict(orient='records'):
            clean_row = {}
            for k, v in row.items():
                # Unwrap numpy scalar
                if isinstance(v, np.generic):
                    v = v.item()
                # Convert NaN or infinite floats to None
                if isinstance(v, float):
                    if not math.isfinite(v):
                        v = None
                clean_row[k] = v
            records.append(clean_row)

        try:
            resp = supabase.from_(table_name).insert(records).execute()
        finally:
            os.remove(temp_path)

        if getattr(resp, 'data', None):
            flash(f"Successfully inserted {len(resp.data)} rows into {table_name}.", "success")
        else:
            flash("Failed to insert records. Check data for invalid values.", "danger")

        return redirect(url_for('admin_dashboard'))

    return render_template('admin/upload_csv.html', table_map=TABLE_MAP)


@app.route('/external')
def external():
    try:
        # Read crop production data
        wheat_df = pd.read_csv('external_factor_data/wheat_production.csv')
        cotton_df = pd.read_csv('external_factor_data/cotton_production.csv')
        sugar_df = pd.read_csv('external_factor_data/sugar_production.csv')
        maize_df = pd.read_csv('external_factor_data/maize_production.csv')
        export_df = pd.read_csv('external_factor_data/export_trends_fake_data.csv')
        rainfall_df = pd.read_csv('external_factor_data/rainfall_crop_prices_2010_2025.csv')
        
        # Read USD exchange rate data
        usd_df = pd.read_csv('external_factor_data/USD_PKR Historical Data.csv')
        
        # Process USD data
        usd_dates = usd_df['Date'].tolist()
        usd_prices = usd_df['Price'].tolist()
        usd_open = usd_df['Open'].tolist()
        usd_high = usd_df['High'].tolist()
        usd_low = usd_df['Low'].tolist()
        usd_change_pct = usd_df['Change %'].str.rstrip('%').astype('float').tolist()

        # Get list of years
        years = wheat_df['YEAR'].unique().tolist()
        
        # Get initial data (latest year)
        latest_year = years[-1]  # Get the last year from the list
        
        # Convert data to lists for initial charts
        initial_wheat_data = [
            float(wheat_df[wheat_df['YEAR'] == latest_year]['Punjab'].iloc[0]),
            float(wheat_df[wheat_df['YEAR'] == latest_year]['Sindh'].iloc[0]),
            float(wheat_df[wheat_df['YEAR'] == latest_year]['KPK'].iloc[0]),
            float(wheat_df[wheat_df['YEAR'] == latest_year]['Balochistan'].iloc[0])
        ]
        
        initial_cotton_data = [
            float(cotton_df[cotton_df['YEAR'] == latest_year]['Punjab'].iloc[0]),
            float(cotton_df[cotton_df['YEAR'] == latest_year]['Sindh'].iloc[0]),
            float(cotton_df[cotton_df['YEAR'] == latest_year]['KPK'].iloc[0]),
            float(cotton_df[cotton_df['YEAR'] == latest_year]['Balochistan'].iloc[0])
        ]
        
        initial_sugar_data = [
            float(sugar_df[sugar_df['YEAR'] == latest_year]['Punjab'].iloc[0]),
            float(sugar_df[sugar_df['YEAR'] == latest_year]['Sindh'].iloc[0]),
            float(sugar_df[sugar_df['YEAR'] == latest_year]['KPK'].iloc[0]),
            float(sugar_df[sugar_df['YEAR'] == latest_year]['Balochistan'].iloc[0])
        ]
        
        initial_maize_data = [
            float(maize_df[maize_df['YEAR'] == latest_year]['Punjab'].iloc[0]),
            float(maize_df[maize_df['YEAR'] == latest_year]['Sindh'].iloc[0]),
            float(maize_df[maize_df['YEAR'] == latest_year]['KPK'].iloc[0]),
            float(maize_df[maize_df['YEAR'] == latest_year]['Balochistan'].iloc[0])
        ]

        # Process export trends data
        export_dates = (export_df['Year'].astype(str) + '-' + 
                       export_df['Month'].astype(str).str.zfill(2)).tolist()
        export_index = export_df['Export_Index'].tolist()
        export_prices = export_df['Price'].tolist()

        # Process rainfall and crop price data
        rainfall_years = rainfall_df['Year'].tolist()
        rainfall_data = rainfall_df['Rainfall (mm)'].tolist()
        crop_prices = rainfall_df['Crop Price (USD/ton)'].tolist()

    except Exception as e:
        flash(f'Error loading data: {str(e)}', 'danger')
        return redirect(url_for('index'))

    # Previous data for other charts
    petrol_df = pd.read_csv('external_factor_data/Petrol Prices.csv')
    petrol_dates = petrol_df['date'].tolist()
    petrol_prices = petrol_df['price'].tolist()

    inflation_df = pd.read_csv('external_factor_data/pakistan-inflation-rate-cpi.csv')
    inflation_dates = inflation_df['date'].tolist()
    inflation_rates = inflation_df['per_Capita'].tolist()

    return render_template('external.html',
        years=years,
        initial_wheat_data=initial_wheat_data,
        initial_cotton_data=initial_cotton_data,
        initial_sugar_data=initial_sugar_data,
        initial_maize_data=initial_maize_data,
        petrol_dates=json.dumps(petrol_dates),
        petrol_prices=json.dumps(petrol_prices),
        inflation_dates=json.dumps(inflation_dates),
        inflation_rates=json.dumps(inflation_rates),
        export_dates=json.dumps(export_dates),
        export_index=json.dumps(export_index),
        export_prices=json.dumps(export_prices),
        rainfall_years=json.dumps(rainfall_years),
        rainfall_data=json.dumps(rainfall_data),
        crop_prices=json.dumps(crop_prices),
        usd_dates=json.dumps(usd_dates),
        usd_prices=json.dumps(usd_prices),
        usd_open=json.dumps(usd_open),
        usd_high=json.dumps(usd_high),
        usd_low=json.dumps(usd_low),
        usd_change_pct=json.dumps(usd_change_pct),
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
@app.route('/get_crop_production_data/<int:year>')
def get_crop_production_data(year):
    # Read crop data
    wheat_df = pd.read_csv('external_factor_data/wheat_production.csv')
    cotton_df = pd.read_csv('external_factor_data/cotton_production.csv')
    sugar_df = pd.read_csv('external_factor_data/sugar_production.csv')
    maize_df = pd.read_csv('external_factor_data/maize_production.csv')

    # Get data for the selected year
    return jsonify({
        'wheat': get_province_data(wheat_df, year),
        'cotton': get_province_data(cotton_df, year),
        'sugar': get_province_data(sugar_df, year),
        'maize': get_province_data(maize_df, year)
    })

def get_province_data(df, year):
    # Get row for the specified year
    year_data = df[df['YEAR'] == year].iloc[0]
    
    # Return province-wise data
    return {
        'Punjab': year_data['Punjab'],
        'Sindh': year_data['Sindh'],
        'KPK': year_data['KPK'],
        'Balochistan': year_data['Balochistan']
    }


# Temporary storage for verification codes
verification_data = {}

def signup_user(data):
    # Hash and insert via Supabase
    hashed = generate_password_hash(data['password'], method='pbkdf2:sha256')
    payload = {
        "username": data['username'],
        "email":    data['email'],
        "password": hashed,
        "fullname": data['fullname'],
        "phone":    data['phone']
    }
    return supabase.from_("user").insert(payload).execute()

@app.route('/signup', methods=['POST'])
def signup():
    fullname = request.form['fullname']
    username = request.form['username']
    email    = request.form['email']
    phone    = request.form['phone']
    password = request.form['password']
    confirm  = request.form['confirm-password']

    # 1) Email format check (defensive against missing keys)
    api_url = f"https://emailvalidation.abstractapi.com/v1/?api_key={EMAIL_VERIFICATION_API_KEY}&email={email}"
    resp_json = requests.get(api_url).json()
    valid_format    = resp_json.get('is_valid_format', {}).get('value', False)
    deliverability  = resp_json.get('deliverability', '').upper()

    if not valid_format or deliverability != "DELIVERABLE":
        flash('Invalid or undeliverable email address. Please use a valid email.', 'danger')
        return render_template('index.html', email=email)

    # 2) Username rules
    import re
    if not re.match(r'^[a-zA-Z0-9._]{3,15}$', username) or any(x in username for x in ["..","__","._","_."]):
        flash('Username rules violation.', 'danger')
        return render_template('index.html', username=username)

    # 3) Password strength & match
    pwd_pattern = (r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)'
                   r'(?=.*[!@#$%^&*()_+={}\[\]:;"<>,.?/\\|~-])'
                   r'.{8,}$')
    if not re.match(pwd_pattern, password):
        flash('Password must be min 8 chars, mixed case, number & special.', 'danger')
        return render_template('index.html', username=username, email=email)

    if password != confirm:
        flash('Passwords do not match!', 'danger')
        return render_template('index.html', username=username, email=email)

    # 4) Check existing user in Supabase
    #    Using or() to search username OR email
    check = supabase.from_("user") \
        .select("id") \
        .or_(f"username.eq.{username},email.eq.{email}") \
        .limit(1) \
        .execute()
    if check.data:
        flash('Username or email already exists!', 'danger')
        return render_template('index.html', username=username, email=email)

    # 5) Generate & store OTP
    code = ''.join(secrets.choice("0123456789") for _ in range(6))
    verification_data[email] = {
        "code":   code,
        "expiry": datetime.now() + timedelta(minutes=5),
        "data":   {"fullname":fullname,"username":username,"email":email,"phone":phone,"password":password}
    }
    session['signup_data'] = verification_data[email]['data']

    # 6) Send OTP email
    msg = Message('Your Verification Code',
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[email])
    msg.body = f"Your code is {code}. Expires in 5 minutes."
    mail.send(msg)

    flash('Verification code sent. Check your email!', 'info')
    return redirect(url_for('verify', email=email))

@app.route('/verify/<email>', methods=['GET', 'POST'])
def verify(email):
    if request.method == 'POST':
        entered = request.form['verification_code']
        record  = verification_data.get(email)

        if not record:
            flash('No verification found. Please sign up again.', 'danger')
            return redirect(url_for('index'))

        if datetime.now() > record['expiry']:
            verification_data.pop(email, None)
            flash('Code expired. Please sign up again.', 'danger')
            return redirect(url_for('index'))

        if entered != record['code']:
            flash('Invalid code. Try again.', 'danger')
            return redirect(url_for('verify', email=email))

        # Attempt the insert
        signup_resp = signup_user(record['data'])

        # Check the returned data list
        if signup_resp.data and isinstance(signup_resp.data, list) and len(signup_resp.data) > 0:
            flash('Signup complete! You can now log in.', 'success')
        else:
            # Insertion failed for some reason
            flash('Signup failed: no user was created.', 'danger')
            return redirect(url_for('index'))

        # Clean up and redirect
        verification_data.pop(email, None)
        session.pop('signup_data', None)
        return redirect(url_for('index'))

    return render_template('verify.html', email=email)


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Fetch exactly one user with that username
    resp = supabase \
        .from_("user") \
        .select("*") \
        .eq("username", username) \
        .single() \
        .execute()

    # If no data, user not found
    user = resp.data if resp.data else None

    if user and check_password_hash(user['password'], password):
        session['user_id']   = user['id']
        session['username']  = user['username']
        flash('Logged in successfully!', 'success')
        return redirect(
            url_for('admin_dashboard') if user.get('is_admin')
            else url_for('dashboard')
        )

    flash('Invalid username or password!', 'danger')
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')  # Redirect to the English home page


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    
    response = make_response(redirect(url_for('index')))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Add this to handle direct access to pages that require login
@app.before_request
def before_request():
    # List of routes that require authentication
    protected_routes = ['dashboard', 'admin_dashboard']
    
    if request.endpoint in protected_routes and 'user_id' not in session:
        flash('Please log in to access this page.', 'info')
        return redirect(url_for('login'))
# @app.route('/en/home')
# def en_home():
#     return render_template('home.html')  # English home page


# @app.route('/dashboard')
# def home():
#     return render_template('dashboard.html')

# Add dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Add FAQs route
@app.route('/faqs')
def faqs():
    return render_template('faqs.html')
# Add CHATBOT route
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# @app.route('/ur/verify')
# def ur_verify():
#     return render_template('ur_verify.html')  # Urdu verification page

# @app.route('/ur/dashboard')
# def ur_dashboard():
#     return render_template('ur_dashboard.html')  # Urdu dashboard

# @app.route('/ur/help-centre')
# def ur_help_center():
#     return render_template('ur_help_centre.html')  # Urdu Help Centre template

# @app.route('/ur/faqs')
# def ur_faqs():
#     return render_template('ur_faqs.html')  # Urdu FAQs page

#chatbot code#

# def transcribe_audio(audio_data):
#     try:
#         response = requests.post(WHISPER_API_URL, headers=WHISPER_HEADERS, data=audio_data)
#         output = response.json()
#         return output.get("text", "Error in transcription")
#     except Exception as e:
#         return f"Error in transcription: {str(e)}"

# def translate_roman_urdu(text):
#     translated = translator.translate(text, src='auto', dest='en')
#     return translated.text

# def get_weather_data(location):
#     try:
#         params = {"key": WEATHER_API_KEY, "q": location, "days": 1}
#         response = requests.get(WEATHER_BASE_URL, params=params)
#         data = response.json()

#         if "error" in data:
#             return f"Could not fetch weather data for '{location}'. Please try again."

#         location_name = data["location"]["name"]
#         region = data["location"]["region"]
#         country = data["location"]["country"]
#         current = data["current"]
#         condition = current["condition"]["text"]
#         temp_c = current["temp_c"]
#         chance_of_rain = data["forecast"]["forecastday"][0]["day"]["daily_chance_of_rain"]

#         return (f"Weather in {location_name}, {region}, {country}:\n"
#                 f"- Condition: {condition}\n"
#                 f"- Temperature: {temp_c}°C\n"
#                 f"- Chance of Rain: {chance_of_rain}%")
#     except Exception as e:
#         return f"Error fetching weather data: {e}"

# def extract_location_with_gemini(query):
#     gemini_model = genai.GenerativeModel('gemini-1.5-flash-002')
#     prompt = f"""
#     Extract the location from the following user query. If no location is mentioned, return 'None'.
#     Query: {query}
#     Location:
#     """
#     response = gemini_model.generate_content(prompt)
#     location = response.text.strip()
#     return location if location.lower() != "none" else None

# def generate_gemini_answer(conversation_history):
#     gemini_model = genai.GenerativeModel('gemini-1.5-flash-002')
#     response = gemini_model.generate_content([f"""
#     You are a helpful and knowledgeable agricultural chatbot. Assist with crop-related and weather-related queries.
#     Use the provided weather information if applicable. If not, reply based on your general knowledge.

#     ## Conversation History:
#     {conversation_history}

#     ## Current User Query:
#     {conversation_history.splitlines()[-1]}

#     ## Answer:
#     """])
#     return response.text

# @app.route('/api/chat', methods=['POST'])
# def chat():
#     try:
#         data = request.json
#         input_type = data.get('type', 'text')
        
#         if input_type == 'audio':
#             # Handle audio input
#             audio_data = base64.b64decode(data['audio'].split(',')[1])
#             user_message = transcribe_audio(audio_data)
#             # Translate transcribed text to Urdu for display
#             translated_output = GoogleTranslator(source="auto", target="ur").translate(user_message)
#         else:
#             # Handle text input
#             user_message = data.get('message', '')
            
#         # Translate to English for processing
#         translated_question = translate_roman_urdu(user_message)
#         conversation_history = f"User: {translated_question}\n"
        
#         # Check for weather-related queries
#         if "weather" in translated_question.lower() or "rain" in translated_question.lower():
#             location = extract_location_with_gemini(translated_question)
#             if not location:
#                 location = "Pakistan"
#             weather_response = get_weather_data(location)
#             conversation_history += f"Weather Info: {weather_response}\n"
        
#         # Generate Gemini AI response
#         response = generate_gemini_answer(conversation_history)
        
#         return jsonify({
#             'response': response,
#             'translated_question': translated_question,
#             'urdu_translation': translated_output if input_type == 'audio' else None
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    input_type = data.get('type','text')

    # # Transcribe if audio
    # if input_type=='audio':
    #     audio_data = base64.b64decode(data['audio'].split(',')[1])
    #     user_text = transcribe_audio(audio_data)
    # else:
    user_text = data.get('message','')

    # Optional: detect if weather enrich needed
    if 'weather' in user_text.lower():
        # Example: static coords for Punjab
        upsert_weather(31.0,74.0, date.today().isoformat(), 'Punjab')

    # Ask via SQL‑first RAG
    answer = ask_sql_rag(user_text)
    return jsonify({'response': answer})


# Update the get_sugar_data function to handle Wheat
@app.route('/get_sugar_data', methods=['GET'])
def get_sugar_data():
    crop = request.args.get('crop')
    by_product = request.args.get('by_product')
    regions = request.args.get('regions').split(',')

    if crop == 'Sugar':
        actual_file = SUGAR_ACTUAL_FILE
        pred_file = SUGAR_PRED_FILE
    elif crop == 'Maize':
        actual_file = MAIZE_ACTUAL_FILE
        pred_file = MAIZE_PRED_FILE
    elif crop == 'Cotton':
        actual_file = COTTON_ACTUAL_FILE
        pred_file = COTTON_PRED_FILE
    elif crop == 'Wheat':
        actual_file = WHEAT_ACTUAL_FILE
        pred_file = WHEAT_PRED_FILE
    else:
        return jsonify({'error': 'Invalid crop selected.'}), 400

    # Read the actual and predicted prices data
    actual_data = pd.read_csv(actual_file)
    pred_data = pd.read_csv(pred_file)

    # Filter by by_product if specified and if it's Cotton
    if by_product and crop == 'Cotton':
        actual_data = actual_data[actual_data['by_product'] == int(by_product)]
        pred_data = pred_data[pred_data['by_product'] == int(by_product)]

    # Filter and prepare data for each region
    result = []
    colors_actual = ['rgba(76, 175, 80, 1)', 'rgba(255, 87, 34, 1)', 'rgba(33, 150, 243, 1)']
    colors_predicted = ['rgba(76, 175, 80, 0.5)', 'rgba(255, 87, 34, 0.5)', 'rgba(33, 150, 243, 0.5)']

    for i, region in enumerate(regions):
        region_actual = actual_data[actual_data['province'] == region].copy()
        region_pred = pred_data[pred_data['province'] == region].copy()

        if region_actual.empty or region_pred.empty:
            continue

        # Standardize dates to ISO 8601 format
        region_actual['date'] = pd.to_datetime(region_actual['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        region_pred['date'] = pd.to_datetime(region_pred['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        dates_actual = region_actual['date'].tolist()
        dates_predicted = region_pred['date'].tolist()

        actual_prices = region_actual['price'].tolist()
        predicted_prices = region_pred['price'].tolist()

        result.append({
            'region': region,
            'dates_actual': dates_actual,
            'actual_prices': actual_prices,
            'dates_predicted': dates_predicted,
            'predicted_prices': predicted_prices,
            'color_actual': colors_actual[i % len(colors_actual)],
            'color_predicted': colors_predicted[i % len(colors_predicted)],
        })

    return jsonify(result)

# Update the get_crop_data function to handle Wheat
@app.route('/get_crop_data', methods=['GET'])
def get_crop_data():
    province = request.args.get('province')
    crops = request.args.get('crops').split(',')

    if not province or not crops:
        return jsonify({'error': 'Invalid input. Please select a province and at least one crop.'}), 400

    crop_files = {
        'Sugar': (SUGAR_ACTUAL_FILE, SUGAR_PRED_FILE),
        'Maize': (MAIZE_ACTUAL_FILE, MAIZE_PRED_FILE),
        'Cotton-1': (COTTON_ACTUAL_FILE, COTTON_PRED_FILE),
        'Cotton-2': (COTTON_ACTUAL_FILE, COTTON_PRED_FILE),
        'Wheat': (WHEAT_ACTUAL_FILE, WHEAT_PRED_FILE)
    }

    result = []
    colors_actual = ['rgba(76, 175, 80, 1)', 'rgba(255, 87, 34, 1)', 'rgba(33, 150, 243, 1)', 'rgba(156, 39, 176, 1)']
    colors_predicted = ['rgba(76, 175, 80, 0.5)', 'rgba(255, 87, 34, 0.5)', 'rgba(33, 150, 243, 0.5)', 'rgba(156, 39, 176, 0.5)']

    for i, crop in enumerate(crops):
        if crop not in crop_files:
            continue

        actual_file, pred_file = crop_files[crop]
        actual_data = pd.read_csv(actual_file)
        pred_data = pd.read_csv(pred_file)

        # Handle Cotton by-products
        if crop.startswith('Cotton-'):
            by_product = int(crop.split('-')[1])
            actual_data = actual_data[actual_data['by_product'] == by_product]
            pred_data = pred_data[pred_data['by_product'] == by_product]

        region_actual = actual_data[actual_data['province'] == province].copy()
        region_pred = pred_data[pred_data['province'] == province].copy()

        if region_actual.empty or region_pred.empty:
            continue

        # Standardize dates
        region_actual['date'] = pd.to_datetime(region_actual['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        region_pred['date'] = pd.to_datetime(region_pred['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        dates_actual = region_actual['date'].tolist()
        dates_predicted = region_pred['date'].tolist()

        actual_prices = region_actual['price'].tolist()
        predicted_prices = region_pred['price'].tolist()

        result.append({
            'crop': crop,
            'dates_actual': dates_actual,
            'actual_prices': actual_prices,
            'dates_predicted': dates_predicted,
            'predicted_prices': predicted_prices,
            'color_actual': colors_actual[i % len(colors_actual)],
            'color_predicted': colors_predicted[i % len(colors_predicted)],
        })

    return jsonify(result)
# @app.route('/get_heatmap_data')
# def get_heatmap_data():
#     try:
#         # Read and process the data
#         df = pd.read_csv("new_data/CottonMaster.csv")
#         df['province'] = df['province'].str.strip()
#         df['price'] = (df['minimum'] + df['maximum']) / 2
        
#         df = df[['date', 'station_id', 'by_product_id', 'province', 'Lat', 'Long', 'price']]
#         df['date'] = pd.to_datetime(df['date'])
#         df = df.drop_duplicates()
#         df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
#         df['Long'] = pd.to_numeric(df['Long'], errors='coerce')
        
#         # Filter DataFrame
#         df = df[(df['date'] == '2020-12-31') & (df['by_product_id'] == 7)]
        
#         # Prepare heatmap data
#         heat_data = df[['Lat', 'Long', 'price']].values.tolist()
        
#         # Return data needed for rendering
#         return jsonify({
#             'heat_data': heat_data,
#             'min_price': float(df['price'].min()),
#             'max_price': float(df['price'].max())
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


from folium import Map
from folium.plugins import HeatMap
import pandas as pd
import branca
import json
from flask import jsonify

# Mapping crop names to CSV file paths
CROP_CSV_MAPPING = {
    "cotton": "new_data/CottonMaster.csv",
    "sugar": "new_data/SugarMaster.csv",
    "wheat": "new_data/WheatMaster.csv",
    "maize": "new_data/MaizeMaster.csv"
}
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        # Fetch user from Supabase
        resp = supabase.from_('user') \
                       .select('id') \
                       .eq('email', email) \
                       .single() \
                       .execute()
        user = resp.data if resp.data else None
        if not user:
            flash('No account found with that email address.', 'danger')
            return redirect(url_for('forgot_password'))

        # Generate reset code
        reset_code = ''.join(secrets.choice('0123456789') for _ in range(6))
        verification_data[email] = {
            'code':    reset_code,
            'expiry':  datetime.now() + timedelta(minutes=15),
            'purpose': 'password_reset'
        }

        # Send reset email
        msg = Message('Password Reset Code',
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[email])
        msg.body = (
            f'Your password reset code is {reset_code}. '
            'It will expire in 15 minutes.'
        )
        mail.send(msg)

        flash('A password reset code has been sent to your email.', 'info')
        return redirect(url_for('reset_password', email=email))

    return render_template('forgot_password.html')


@app.route('/reset-password/<email>', methods=['GET', 'POST'])
def reset_password(email):
    if request.method == 'POST':
        reset_code      = request.form['reset_code']
        new_password    = request.form['new_password']
        confirm_password= request.form['confirm_password']

        # Validate code entry
        record = verification_data.get(email)
        if (not record
            or datetime.now() > record['expiry']
            or record.get('purpose') != 'password_reset'):
            verification_data.pop(email, None)
            flash('Invalid or expired reset link. Please try again.', 'danger')
            return redirect(url_for('forgot_password'))

        if reset_code != record['code']:
            flash('Invalid reset code. Please try again.', 'danger')
            return redirect(url_for('reset_password', email=email))

        # Password rules
        import re
        pwd_regex = (
            r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)'
            r'(?=.*[!@#$%^&*()_+={}[\]:;"<>,.?/\\|~-]).{8,}$'
        )
        if not re.match(pwd_regex, new_password):
            flash('Password must be at least 8 characters long, '
                  'include uppercase, lowercase, number, and special char.',
                  'danger')
            return redirect(url_for('reset_password', email=email))

        if new_password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('reset_password', email=email))

        # Update the user's password
        hashed = generate_password_hash(
            new_password, method='pbkdf2:sha256'
        )
        resp = supabase.from_('user') \
                       .update({'password': hashed}) \
                       .eq('email', email) \
                       .execute()
        if not getattr(resp, 'data', None):
            flash('Failed to reset password. Please try again.', 'danger')
            return redirect(url_for('reset_password', email=email))

        # Cleanup
        verification_data.pop(email, None)
        flash('Your password has been reset! You can now log in.', 'success')
        return redirect(url_for('index'))

    return render_template('reset_password.html', email=email)

@app.route('/get_heatmap_data')
def get_heatmap_data():
    try:
        # Get parameters from the request
        crop = request.args.get('crop', '').strip().lower()
        print(f"Received crop: {crop}")
        by_product = request.args.get('by_product', '')

        # Ensure a valid crop is selected
        if crop not in CROP_CSV_MAPPING:
            return jsonify({'error': 'Invalid or missing crop selection'}), 400

        # Load CSV based on selected crop
        csv_path = CROP_CSV_MAPPING[crop]
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Debug: Check if data loaded correctly
        print(f"DataFrame Shape before filtering: {df.shape}")
        print("Column Names:", df.columns)
        print(df.head())

        # Data Cleaning & Processing
        df.columns = df.columns.str.lower()  # Standardize column names
        df['province'] = df['province'].str.strip()
        df['price'] = (df['minimum'] + df['maximum']) / 2
        if crop == "wheat":
            df = df[['date', 'station_id', 'province', 'lat', 'long', 'maximum','price']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates()
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['long'] = pd.to_numeric(df['long'], errors='coerce')
        else:
            df = df[['date', 'station_id', 'by_product_id', 'province', 'lat', 'long', 'price']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates()
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['long'] = pd.to_numeric(df['long'], errors='coerce')

        # Debug: Check unique dates before filtering
        print("Available Dates before filtering:", df['date'].unique())
        # Apply by_product filter if provided
        if by_product:
            try:
                by_product_id = int(by_product)
                print("by product value",by_product_id)
                print(f"Filtering for by_product_id: {by_product_id}")
                print("Available by_product_id values:", df['by_product_id'].unique())
                df = df[df['by_product_id'] == by_product_id]
                print(f"Rows after filtering by by_product_id: {len(df)}")
            except ValueError:
                return jsonify({'error': 'Invalid by-product ID'}), 400
        if df.empty:
            return jsonify({'error': 'No data available for the selected filters'}), 404
        # Apply date filter
        latest_date = df['date'].max()
    # Convert to datetime object
        latest_date = pd.to_datetime(latest_date)

    # Format it to 'YYYY-MM-DD' (remove the time part)
        target_date = latest_date.strftime('%Y-%m-%d')
        df = df[df['date'] == target_date]
        print(f"Rows after filtering by date {target_date}: {len(df)}")
        # Ensure data is available
        

                # Create Folium Map
                        
                
        # Ensure lat/lon are numeric
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['long'] = pd.to_numeric(df['long'], errors='coerce')

        # Create the map with an initial view
        m = Map(
            location=[30.0, 70.0],  # Default center (Pakistan)
            zoom_start=4,  # Initial zoom
            max_bounds=True,
            tiles="CartoDB positron",
            width="100%",
            height="350px"
        )

        # Add the heatmap
        heat_data = list(zip(df['lat'], df['long'], df['price']))  # Assuming price influences intensity
        HeatMap(heat_data).add_to(m)

        # **Fit map to the data points**
        if not df.empty:
            sw = [df['lat'].min(), df['long'].min()]  # Southwest corner
            ne = [df['lat'].max(), df['long'].max()]  # Northeast corner
            m.fit_bounds([sw, ne])  # Auto-zoom based on bounds

        # Prepare heatmap data
        heat_data = df[['lat', 'long', 'price']].dropna().values.tolist()

        # Add HeatMap layer
        HeatMap(
            heat_data,
            name="Heatmap",
            min_opacity=0.2,
            radius=20,
            blur=15,
            max_zoom=4,
        ).add_to(m)

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=5,  # Smallest possible size
                color="transparent",  # No border color
                fill=True,
                fill_color="transparent",  # Fully transparent fill
                fill_opacity=0,  # 0 opacity to ensure invisibility
                tooltip=f"Price: {row['price']:.2f} PKR"
            ).add_to(m)
        # Add color scale legend
        colormap = branca.colormap.LinearColormap(
            colors=['blue', 'green', 'yellow', 'red'],
            vmin=df['price'].min(),
            vmax=df['price'].max(),
            caption="Price Intensity"
        ).to_step(n=5)

        colormap.add_to(m)
        m.fit_bounds([[23, 60], [38, 77]])

        # Convert map to HTML
        map_html = m._repr_html_()

        return jsonify({
            'map_html': map_html,
            'min_price': float(df['price'].min()),
            'max_price': float(df['price'].max())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
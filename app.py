import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from datetime import datetime, timedelta
from flask_cors import CORS
from deep_translator import GoogleTranslator
import detectlanguage
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests
import os
from functools import wraps
import folium
from folium.plugins import HeatMap
from folium import Map
import pandas as pd
import numpy as np
import branca
import json
import math
import secrets
import tempfile
import uuid
from supabase_client import supabase
from dotenv import load_dotenv
from agribot import ask_agri_bot
from postgrest.exceptions import APIError
from flask_caching import Cache

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

detectlanguage.configuration.api_key = os.getenv('DETECT_LANGUAGE_API')

# Whisper API configuration
WHISPER_API_URL = os.getenv('WHISPER_API_URL')
WHISPER_HEADERS = {"Authorization": os.getenv('WHISPER_HEADERS')}

#email verification
EMAIL_VERIFICATION_API_KEY = os.getenv('EMAIL_VERIFICATION_API_KEY')

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'agriverseofficial@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')  # Replace with your email password
mail = Mail(app)

# Configure Flask-Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache',
    'CACHE_DEFAULT_TIMEOUT': 300  # Cache for 5 minutes
})

TABLE_MAP = {
    'wheat':  'wheatmaster',
    'maize':  'maizemaster',
    'cotton': 'cottonmaster',
    'sugar':  'sugarmaster'
}

# Mapping crop names to CSV file paths
CROP_CSV_MAPPING = {
    "cotton": "new_data/CottonMaster.csv",
    "sugar": "new_data/SugarMaster.csv",
    "wheat": "new_data/WheatMaster.csv",
    "maize": "new_data/MaizeMaster.csv"
}

# File to store user data
SUGAR_ACTUAL_FILE = 'data/final_sugar_province_actual.csv'
SUGAR_PRED_FILE = 'data/final_sugar_province_pred_formatted.csv'
MAIZE_ACTUAL_FILE = 'data/maize_actual.csv'
MAIZE_PRED_FILE = 'data/maize_pred.csv'
MAIZE_ACTUAL_FILE_final = 'data/maize_actual_final.csv'
MAIZE_PRED_FILE_final = 'data/maize_pred_final.csv'
COTTON_ACTUAL_FILE = 'data/cotton_actual.csv'
COTTON_PRED_FILE = 'data/cotton_pred.csv'
WHEAT_ACTUAL_FILE = 'data/wheat_actual.csv'
WHEAT_PRED_FILE = 'data/wheat_pred.csv'


# Update the UPLOAD_FOLDER to be an absolute path

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Google Drive API setup
def get_drive_service():
    # Load the service account key JSON file
    # You need to create this file from Google Cloud Console
    # and save it in a secure location
    # Provide the service account key JSON via an environment variable.
    # Never commit the key file to the repo.
    service_account_info = json.loads(os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON', '{}'))
    
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, 
        scopes=['https://www.googleapis.com/auth/drive']
    )
    
    service = build('drive', 'v3', credentials=credentials)
    return service


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
    posts = resp.data if resp.data else []
    return render_template('admin/admin_dashboard.html', posts=posts)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/newsletter')
def newsletter():
    resp = (supabase
            .from_('newsletter_post')
            .select("*, author:user(username)")
            .order('created_at', desc=True)
            .execute())
    posts = resp.data if resp.data else []
    return render_template('newsletter.html', posts=posts)

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

def upload_to_drive(file_path, original_filename):
    """Upload a file to Google Drive and return the public URL"""
    try:
        print(f"Starting upload process for file: {original_filename}")
        service = get_drive_service()
        
        # Create unique filename
        file_ext = os.path.splitext(original_filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        print(f"Generated unique filename: {unique_filename}")
        
        # Define file metadata
        folder_id = "1rON0REMLjjHb6hU7mmWa0otd-2_WwZPQ"  # Fixed folder ID format
        
        file_metadata = {
            'name': unique_filename,
            'parents': [folder_id]
        }
        
        print(f"Uploading file to folder ID: {folder_id}")
        
        # Upload file
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webContentLink'
        ).execute()
        
        print(f"File uploaded successfully with ID: {file['id']}")
        
        # Make the file publicly accessible
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        
        service.permissions().create(
            fileId=file['id'],
            body=permission
        ).execute()
        
        print("Added public permission to file")
        
        # Get the direct download link that works for viewing
        direct_url = f"https://drive.google.com/uc?export=view&id={file['id']}"
        print(f"Generated direct URL: {direct_url}")
        
        return direct_url
    
    except Exception as e:
        print(f"Drive upload error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        raise e


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


#EXTERNAL CHART ROUTES
@app.route('/external')
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes, vary by query parameters
def external():
    try:
        # Fetch crop production data from Supabase
        wheat_data = supabase.from_('wheat_production').select('*').execute()
        cotton_data = supabase.from_('cotton_production').select('*').execute()
        sugar_data = supabase.from_('sugar_production').select('*').execute()
        maize_data = supabase.from_('maize_production').select('*').execute()
        
        # Convert Supabase responses to dataframes
        wheat_df = pd.DataFrame(wheat_data.data)
        cotton_df = pd.DataFrame(cotton_data.data)
        sugar_df = pd.DataFrame(sugar_data.data)
        maize_df = pd.DataFrame(maize_data.data)

        # Fetch USD exchange rate data
        usd_data = supabase.from_('usd_pkr').select('*').execute()
        usd_df = pd.DataFrame(usd_data.data)

        # Process USD data
        usd_dates = usd_df['Date'].tolist()
        usd_prices = usd_df['Price'].tolist()
        usd_open = usd_df['Open'].tolist()
        usd_high = usd_df['High'].tolist()
        usd_low = usd_df['Low'].tolist()
        usd_change_pct = [float(x.rstrip('%')) if isinstance(x, str) else x for x in usd_df['Change %'].tolist()]

        # Fetch crude oil data
        crude_data = supabase.from_('crude_oil').select('*').execute()
        crude_df = pd.DataFrame(crude_data.data)

        # Process crude oil data
        crude_dates = crude_df['Date'].tolist()
        crude_prices = crude_df['Price'].tolist()

        # Get list of years
        years = sorted(wheat_df['YEAR'].unique().tolist())
        
        # Get initial data (latest year)
        latest_year = years[-1]  # Get the last year from the list
        
        initial_wheat_data = get_province_data(wheat_df, latest_year)
        initial_cotton_data = get_province_data(cotton_df, latest_year)
        initial_sugar_data = get_province_data(sugar_df, latest_year)
        initial_maize_data = get_province_data(maize_df, latest_year)

        # Convert the data to lists for the template
        initial_wheat_data = [initial_wheat_data['Punjab'], initial_wheat_data['Sindh'], 
                            initial_wheat_data['KPK'], initial_wheat_data['Balochistan']]
        initial_cotton_data = [initial_cotton_data['Punjab'], initial_cotton_data['Sindh'], 
                             initial_cotton_data['KPK'], initial_cotton_data['Balochistan']]
        initial_sugar_data = [initial_sugar_data['Punjab'], initial_sugar_data['Sindh'], 
                            initial_sugar_data['KPK'], initial_sugar_data['Balochistan']]
        initial_maize_data = [initial_maize_data['Punjab'], initial_maize_data['Sindh'], 
                            initial_maize_data['KPK'], initial_maize_data['Balochistan']]

    except Exception as e:
        flash(f'Error loading data: {str(e)}', 'danger')
        return redirect(url_for('index'))

    # Fetch other chart data from Supabase
    petrol_data = supabase.from_('petrol_prices').select('*').execute()
    petrol_df = pd.DataFrame(petrol_data.data)
    petrol_dates = petrol_df['Date'].tolist()
    petrol_prices = petrol_df['Petrol_Price_PKR'].tolist()

    inflation_data = supabase.from_('inflation_rate').select('*').execute()
    infla_df = pd.DataFrame(inflation_data.data)
    infla_dates = infla_df['year'].tolist()
    infla_prices = infla_df['inflation rate'].tolist()

    return render_template('external.html',
        years=years,
        initial_wheat_data=initial_wheat_data,
        initial_cotton_data=initial_cotton_data,
        initial_sugar_data=initial_sugar_data,
        initial_maize_data=initial_maize_data,
        petrol_dates=json.dumps(petrol_dates),
        petrol_prices=json.dumps(petrol_prices),
        infla_dates=json.dumps(infla_dates),
        infla_prices=json.dumps(infla_prices),
        usd_dates=json.dumps(usd_dates),
        usd_prices=json.dumps(usd_prices),
        usd_open=json.dumps(usd_open),
        usd_high=json.dumps(usd_high),
        usd_low=json.dumps(usd_low),
        usd_change_pct=json.dumps(usd_change_pct),
        crude_dates=json.dumps(crude_dates),
        crude_prices=json.dumps(crude_prices),
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route('/get_crop_production_data/<year>')
def get_crop_production_data(year):
    try:
        # Convert year string to the correct format (e.g., "1947-48")
        year = str(year)
        
        # Fetch crop production data from Supabase
        wheat_data = supabase.from_('wheat_production').select('*').execute()
        cotton_data = supabase.from_('cotton_production').select('*').execute()
        sugar_data = supabase.from_('sugar_production').select('*').execute()
        maize_data = supabase.from_('maize_production').select('*').execute()
        
        # Convert Supabase responses to dataframes
        wheat_df = pd.DataFrame(wheat_data.data)
        cotton_df = pd.DataFrame(cotton_data.data)
        sugar_df = pd.DataFrame(sugar_data.data)
        maize_df = pd.DataFrame(maize_data.data)

        # Get data for the selected year
        wheat_data = get_province_data(wheat_df, year)
        cotton_data = get_province_data(cotton_df, year)
        sugar_data = get_province_data(sugar_df, year)
        maize_data = get_province_data(maize_df, year)

        return jsonify({
            'wheat': wheat_data,
            'cotton': cotton_data,
            'sugar': sugar_data,
            'maize': maize_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def get_province_data(df, year):
    try:
        # Get row for the specified year
        year_data = df[df['YEAR'] == year].iloc[0]
        
        # Return province-wise data
        return {
            'Punjab': float(year_data['Punjab']),
            'Sindh': float(year_data['Sindh']),
            'KPK': float(year_data['KPK']),
            'Balochistan': float(year_data['Balochistan'])
        }
    except IndexError:
        # Return zeros if year not found
        return {
            'Punjab': 0,
            'Sindh': 0,
            'KPK': 0,
            'Balochistan': 0
        }
    except Exception as e:
        raise Exception(f"Error processing data for year {year}: {str(e)}")


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
        flash('Password must be at least 8 characters long, include uppercase, lowercase, number, and special char.', 'danger')
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

    try:
        resp = supabase \
            .from_("user") \
            .select("*") \
            .eq("username", username) \
            .single() \
            .execute()
        user = resp.data
    except APIError:
        flash('Invalid username or password!', 'danger')
        return render_template('index.html'), 200

    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        flash('Logged in successfully!', 'success')
        return redirect(url_for('admin_dashboard') if user.get('is_admin') else url_for('dashboard'))

    flash('Invalid username or password!', 'danger')
    return render_template('index.html'), 200

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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("chat_api")

LANGUAGE_MAPPING = {
    'en': 'English',
    'ur': 'Urdu',
    'pa': 'Punjabi',
    'sd': 'Sindhi',
    'ps': 'Pashto',
    'bal': 'Balochi',
    'hi': 'Urdu'
}

@app.route("/api/chat", methods=["POST"])
def chat():
    # Use request.files for audio, request.json for text
    if request.content_type.startswith('multipart/'):
        # ---- AUDIO MODE ----
        file = request.files.get('file')
        pref_lang = request.form.get('preferred_language', 'en')
        if not file:
            return jsonify(error="No audio file"), 400

        try:
            # 1) Save temp with quality settings
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            file.save(tmp.name)
            tmp.close()

            # 2) Process audio with quality settings
            with open(tmp.name, 'rb') as f:
                # Configure Whisper API with quality settings
                headers = {
                    "Content-Type": "audio/wav",
                    **WHISPER_HEADERS,
                    "X-Whisper-Model": "whisper-1",  # Use latest model
                    "X-Whisper-Language": pref_lang,  # Specify language for better accuracy
                    "X-Whisper-Temperature": "0.2",   # Lower temperature for more focused results
                    "X-Whisper-Prompt": "This is a conversation about agriculture in Pakistan."  # Context prompt
                }
                
                resp = requests.post(
                    WHISPER_API_URL,
                    headers=headers,
                    data=f.read(),
                    timeout=30  # Add timeout
                )
            
            os.unlink(tmp.name)
            resp.raise_for_status()
            transcript = resp.json().get('text','').strip()

            # 3) Enhanced language detection
            try:
                src = detectlanguage.simple_detect(transcript)
            except Exception as e:
                logger.error(f"Language detection error: {e}")
                src = 'en'  # Fallback to English

            # 4) Improved translation handling
            try:
                transcript_en = (transcript if src=='en'
                             else GoogleTranslator(source=src, target='en').translate(transcript))
            except Exception as e:
                logger.error(f"Translation error: {e}")
                transcript_en = transcript  # Fallback to original

            # 5) Get bot response
            try:
                answer_en = ask_agri_bot(transcript_en)
            except Exception as e:
                logger.error(f"Bot response error: {e}")
                answer_en = "I apologize, but I'm having trouble processing your request right now."

            # 6) Enhanced response translation
            try:
                response_text = (answer_en if pref_lang=='en'
                             else GoogleTranslator(source='en', target=pref_lang).translate(answer_en))
            except Exception as e:
                logger.error(f"Response translation error: {e}")
                response_text = answer_en  # Fallback to English

            # 7) Optional Urdu back-translation with error handling
            urdu_trans = None
            if pref_lang != 'ur':
                try:
                    urdu_trans = GoogleTranslator(source=pref_lang, target='ur').translate(response_text)
                except Exception as e:
                    logger.error(f"Urdu translation error: {e}")

            return jsonify({
                'detected_language': LANGUAGE_MAPPING.get(src, src),
                'transcript': transcript,
                'response': response_text,
                'urdu_translation': urdu_trans
            })

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return jsonify(error="Error processing audio"), 500

    else:
        # ---- TEXT MODE ----
        data = request.json or {}
        q = (data.get('message','') or '').strip()
        pref_lang = data.get('preferred_language','en')
        if not q:
            return jsonify(error="No message provided"), 400

        try:
            # 1) Enhanced language detection
            src = detectlanguage.simple_detect(q)
            
            # 2) Improved translation
            q_en = (q if src=='en'
                    else GoogleTranslator(source=src, target='en').translate(q))
            
            # 3) Get bot response
            answer_en = ask_agri_bot(q_en)
            
            # 4) Enhanced response translation
            response_text = (answer_en if pref_lang=='en'
                         else GoogleTranslator(source='en', target=pref_lang).translate(answer_en))

            return jsonify({
                'detected_language': LANGUAGE_MAPPING.get(src, src),
                'response': response_text
            })
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return jsonify(error="Error processing message"), 500
    

# Update the regional_price_analysis function to handle Regional Price Analysis
@app.route('/regional_price_analysis', methods=['GET'])
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes, vary by query parameters
def regional_price_analysis():
    crop = request.args.get('crop')
    by_product = request.args.get('by_product')
    regions = request.args.get('regions').split(',')

    if not crop or not regions:
        return jsonify({'error': 'Invalid input. Please select a crop and at least one region.'}), 400

    # Map crop names to table names
    crop_tables = {
        'Sugar': 'sugarmaster',
        'Maize': 'maizemaster',
        'Cotton': 'cottonmaster',
        'Wheat': 'wheatmaster'
    }

    if crop not in crop_tables:
        return jsonify({'error': 'Invalid crop selected.'}), 400

    table_name = crop_tables[crop]
    result = []
    colors_actual = ['rgba(76, 175, 80, 1)', 'rgba(255, 87, 34, 1)', 'rgba(33, 150, 243, 1)']
    colors_predicted = ['rgba(76, 175, 80, 0.5)', 'rgba(255, 87, 34, 0.5)', 'rgba(33, 150, 243, 0.5)']

    try:
        # Base query for the selected regions
        query = supabase.from_(table_name).select('*').in_('province', regions)
        
        # Handle Cotton by-products
        if crop == 'Cotton' and by_product:
            try:
                by_product_id = int(by_product)
                query = query.eq('by_product_id', by_product_id)
            except ValueError:
                return jsonify({'error': 'Invalid by-product ID'}), 400

        # Execute query
        response = query.execute()
        
        if not response.data:
            return jsonify({'error': 'No data available for the selected parameters'}), 404
            
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(response.data)
        
        # Convert dates to datetime objects
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        current_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        # Filter data from 2015 onwards
        df = df[df['date'] >= pd.to_datetime('2015-01-01')]
        
        if df.empty:
            return jsonify({'error': 'No data available after 2015'}), 404
        
        # Process data for each region
        for i, region in enumerate(regions):
            # Filter data for this region
            region_data = df[df['province'] == region].copy()
            
            if region_data.empty:
                continue

            # Split into actual and predicted based on current date
            actual_data = region_data[region_data['date'] <= current_date].copy()
            pred_data = region_data[region_data['date'] > current_date].copy()
            
            if actual_data.empty or pred_data.empty:
                continue

            # Convert back to string format for JSON serialization
            actual_data['date'] = actual_data['date'].dt.strftime('%Y-%m-%d')
            pred_data['date'] = pred_data['date'].dt.strftime('%Y-%m-%d')

            # Sort dates to ensure chronological order
            actual_data = actual_data.sort_values('date')
            pred_data = pred_data.sort_values('date')

            # Calculate price as average of minimum and maximum if price is not present
            if 'price' not in actual_data.columns:
                actual_data['price'] = (actual_data['minimum'] + actual_data['maximum']) / 2
            if 'price' not in pred_data.columns:
                pred_data['price'] = (pred_data['minimum'] + pred_data['maximum']) / 2

            dates_actual = actual_data['date'].tolist()
            dates_predicted = pred_data['date'].tolist()
            actual_prices = actual_data['price'].tolist()
            predicted_prices = pred_data['price'].tolist()

            result.append({
                'region': region,
                'dates_actual': dates_actual,
                'actual_prices': actual_prices,
                'dates_predicted': dates_predicted,
                'predicted_prices': predicted_prices,
                'color_actual': colors_actual[i % len(colors_actual)],
                'color_predicted': colors_predicted[i % len(colors_predicted)],
            })

        if not result:
            return jsonify({'error': 'No data available for the selected regions'}), 404

        return jsonify(result)

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return jsonify({'error': 'Error processing data'}), 500


@app.route('/price_comparsion', methods=['GET'])
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes, vary by query parameters
def price_comparsion():
    province = request.args.get('province')
    crops = request.args.get('crops').split(',')

    if not province or not crops:
        return jsonify({'error': 'Invalid input. Please select a province and at least one crop.'}), 400

    # Map crop names to table names
    crop_tables = {
        'Sugar': 'sugarmaster',
        'Maize': 'maizemaster',
        'Cotton-1': 'cottonmaster',
        'Cotton-2': 'cottonmaster',
        'Wheat': 'wheatmaster'
    }

    result = []
    colors_actual = ['rgba(76, 175, 80, 1)', 'rgba(255, 87, 34, 1)', 'rgba(33, 150, 243, 1)', 'rgba(156, 39, 176, 1)']
    colors_predicted = ['rgba(76, 175, 80, 0.5)', 'rgba(255, 87, 34, 0.5)', 'rgba(33, 150, 243, 0.5)', 'rgba(156, 39, 176, 0.5)']

    # Get current date for splitting actual vs predicted data
    current_date = datetime.now().strftime('%Y-%m-%d')

    for i, crop in enumerate(crops):
        if crop not in crop_tables:
            continue

        table_name = crop_tables[crop]
        
        try:
            # Base query for the selected province
            query = supabase.from_(table_name).select('*').eq('province', province)
            
            # Handle Cotton by-products
            if crop.startswith('Cotton-'):
                by_product_id = '7' if crop == 'Cotton-1' else '14'
                print(f"Filtering for Cotton by-product {by_product_id}")
                query = query.eq('by_product_id', by_product_id)

            # Execute query
            response = query.execute()
            
            if not response.data:
                continue
                
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(response.data)
            
            # Convert dates to datetime objects first
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            current_date = pd.to_datetime(current_date)

            # Filter data from 2015 onwards
            df = df[df['date'] >= pd.to_datetime('2015-01-01')]
            
            if df.empty:
                return jsonify({'error': 'No data available after 2015'}), 404
                                                    
            
            # Split into actual and predicted based on current date
            # For actual data, use historical data up to current date
            actual_data = df[df['date'] <= current_date].copy()
            # For predicted data, use future dates only
            pred_data = df[df['date'] > current_date].copy()
            
            if actual_data.empty or pred_data.empty:
                print(f"No data available for {crop} in {province}")
                continue

            # Convert back to string format for JSON serialization
            actual_data['date'] = actual_data['date'].dt.strftime('%Y-%m-%d')
            pred_data['date'] = pred_data['date'].dt.strftime('%Y-%m-%d')

            # Sort dates to ensure chronological order
            actual_data = actual_data.sort_values('date')
            pred_data = pred_data.sort_values('date')

            # Calculate price as average of minimum and maximum if price is not present
            if 'price' not in actual_data.columns:
                actual_data['price'] = (actual_data['minimum'] + actual_data['maximum']) / 2
            if 'price' not in pred_data.columns:
                pred_data['price'] = (pred_data['minimum'] + pred_data['maximum']) / 2

            dates_actual = actual_data['date'].tolist()
            dates_predicted = pred_data['date'].tolist()
            actual_prices = actual_data['price'].tolist()
            predicted_prices = pred_data['price'].tolist()

            result.append({
                'crop': crop,
                'dates_actual': dates_actual,
                'actual_prices': actual_prices,
                'dates_predicted': dates_predicted,
                'predicted_prices': predicted_prices,
                'color_actual': colors_actual[i % len(colors_actual)],
                'color_predicted': colors_predicted[i % len(colors_predicted)],
            })

        except Exception as e:
            print(f"Error processing {crop}: {str(e)}")
            continue

    return jsonify(result)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        try:
            # Fetch user from Supabase
            resp = supabase.from_('user') \
                           .select('id, username') \
                           .eq('email', email) \
                           .single() \
                           .execute()
            user = resp.data
        except APIError:
            flash('No account found with that email address.', 'danger')
            return render_template('forgot_password.html'), 200

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
            f'Hello {user["username"]},\n\n'
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
            flash('Password must be at least 8 characters long, include uppercase, lowercase, number, and special char.', 'danger')
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
    
        print(f"Rows after filtering out missing minimum/maximum values: {len(df)}")
        
        if crop == "sugar":
            df['province'] = df['province'].str.strip()
            df['price'] = (df['minimum'] + df['maximum']) / 2
            # Filter by by_product = sugar_mills
            df = df[df['by_product'] == 'sugar_mills']
            df = df[['date', 'station_id', 'province', 'lat', 'long', 'maximum', 'price']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates()
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['long'] = pd.to_numeric(df['long'], errors='coerce')
        # Updated processing logic to handle sugar just like wheat
        elif crop == "wheat":
            df['province'] = df['province'].str.strip()
            df['price'] = (df['minimum'] + df['maximum']) / 2
            df = df[['date', 'station_id', 'province', 'lat', 'long', 'maximum', 'price']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates()
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['long'] = pd.to_numeric(df['long'], errors='coerce')
        else:
            # For cotton and maize which have by_product_id
            df['province'] = df['province'].str.strip()
            df['price'] = (df['minimum'] + df['maximum']) / 2
            df = df[['date', 'station_id', 'by_product_id', 'province', 'lat', 'long', 'price']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates()
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['long'] = pd.to_numeric(df['long'], errors='coerce')

        # Debug: Check unique dates before filtering
        print("Available Dates before filtering:", df['date'].unique())
        
        # Apply by_product filter if provided and not sugar or wheat
        if by_product and crop not in ["sugar", "wheat"]:
            try:
                by_product_id = int(by_product)
                print("by product value", by_product_id)
                print(f"Filtering for by_product_id: {by_product_id}")
                print("Available by_product_id values:", df['by_product_id'].unique())
                df = df[df['by_product_id'] == by_product_id]
                print(f"Rows after filtering by by_product_id: {len(df)}")
            except ValueError:
                return jsonify({'error': 'Invalid by-product ID'}), 400
                
        # Check if dataframe is empty after filtering
        if df.empty:
            return jsonify({'error': 'No data available for the selected filters'}), 404
            
        # Apply date filter - get the latest date
        latest_date = df['date'].max()
        # Format it to 'YYYY-MM-DD' (remove the time part)
        target_date = latest_date.strftime('%Y-%m-%d')
        df = df[df['date'] == target_date]
        print(f"Rows after filtering by latest date {target_date}: {len(df)}")
        
        # Ensure data is available after date filtering
        if df.empty:
            return jsonify({'error': 'No data available for the latest date'}), 404
                
        # Make sure we have data after all the filtering steps
        if df.empty:
            return jsonify({'error': 'No valid data available after filtering rows with missing minimum/maximum values'}), 404

        # Create Folium Map
        m = Map(
            location=[30.0, 70.0],  # Default center (Pakistan)
            zoom_start=4,  # Initial zoom
            max_bounds=True,
            tiles="CartoDB positron",
            width="100%",
            height="350px"
        )

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

        # Add circle markers with tooltips
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=5,  # Smallest possible size
                color="transparent",  # No border color
                fill=True,
                fill_color="transparent",  # Fully transparent fill
                fill_opacity=0,  # 0 opacity to ensure invisibility
                tooltip=f"Province: {row['province']}, Price: {row['price']:.2f} PKR"
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
        import traceback
        print(traceback.format_exc())  # Print detailed error traceback for debugging
        return jsonify({'error': str(e)}), 500
    

def _read_final(crop: str) -> pd.DataFrame:
    """
    Load and normalize the cleaned {crop}-final.csv.
    Parses the 'date' column (dd/mm/YYYY) with dayfirst=True.
    """
    path = os.path.join("predicted_station_data", f"{crop}-final.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    # ensure first col is 'date'
    first = df.columns[0]
    if first.lower() != "date":
        df = df.rename(columns={first: "date"})
    # parse dates
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # strip any text fields
    for txt in ("province_en", "by_product_en", "station_en"):
        if txt in df.columns:
            df[txt] = df[txt].astype(str).str.strip()
    return df

@app.route("/list_provinces")
def list_provinces():
    crop = request.args.get("crop", "").strip().capitalize()
    try:
        df = _read_final(crop)
    except FileNotFoundError:
        return jsonify([]), 404

    provinces = sorted(df["province_en"].dropna().unique().tolist())
    return jsonify(provinces)

@app.route("/list_by_products")
def list_by_products():
    crop = request.args.get("crop", "").strip().capitalize()
    try:
        df = _read_final(crop)
    except FileNotFoundError:
        return jsonify([]), 404

    if "by_product_en" not in df.columns:
        return jsonify([])

    prods = sorted(df["by_product_en"].dropna().unique().tolist())
    return jsonify(prods)

@app.route("/list_stations")
def list_stations():
    crop       = request.args.get("crop",       "").strip().capitalize()
    province   = request.args.get("province",   "").strip().lower()
    by_product = request.args.get("by_product", "").strip().lower()

    try:
        df = _read_final(crop)
    except FileNotFoundError:
        return jsonify([]), 404

    if province:
        df = df[df["province_en"].str.lower() == province]
    if by_product and "by_product_en" in df.columns:
        df = df[df["by_product_en"].str.lower() == by_product]

    stations = sorted(df["station_en"].dropna().unique().tolist())
    return jsonify(stations)

@app.route("/get_station_trend")
def get_station_trend():
    crop       = request.args.get("crop",       "").strip().capitalize()
    station    = request.args.get("station",    "").strip().lower()
    province   = request.args.get("province",   "").strip().lower()
    by_product = request.args.get("by_product", "").strip().lower()

    if not crop or not station:
        return jsonify(error="crop & station required"), 400

    try:
        df = _read_final(crop)
    except FileNotFoundError:
        return jsonify(error="no data for that crop"), 404

    # apply same filters as list_stations
    if province:
        df = df[df["province_en"].str.lower() == province]
    if by_product and "by_product_en" in df.columns:
        df = df[df["by_product_en"].str.lower() == by_product]

    # station filter
    sub = df[df["station_en"].str.lower() == station]
    if sub.empty:
        return jsonify(error="station not found"), 404

    # midpoint price
    sub = sub.copy()
    sub["price"] = (sub["minimum"] + sub["maximum"]) / 2

    # split actual vs forecast at 2025-05-31
    cutoff = pd.Timestamp("2025-05-31")
    hist = sub[sub["date"] <= cutoff][["date","price"]].assign(type="actual")
    pred = sub[sub["date"] >  cutoff][["date","price"]].assign(type="predicted")

    combined = pd.concat([hist, pred], ignore_index=True).sort_values("date")
    out = [
        {
          "date": row.date.strftime("%Y-%m-%d"),
          "price": float(row.price),
          "type":  row.type
        }
        for _, row in combined.iterrows()
    ]
    return jsonify(out)



@app.route("/list_districts")
def list_districts():
    crop     = request.args.get("crop",     "").strip().capitalize()
    province = request.args.get("province", "").strip().lower()
    if crop != "Maize":
        return jsonify([]), 404

    df = _read_final(crop)
    df = df[df["province_en"].str.lower() == province] if province else df
    districts = sorted(df["district_en"].str.strip().dropna().unique().tolist())
    return jsonify(districts)

@app.route("/get_district_trend")
def get_district_trend():
    crop     = request.args.get("crop",     "").strip().capitalize()
    province = request.args.get("province", "").strip().lower()
    district = request.args.get("district", "").strip().lower()
    if crop != "Maize" or not province or not district:
        return jsonify(error="maize + province + district required"), 400

    df = _read_final(crop)
    df = df[df["province_en"].str.lower() == province]
    df = df[df["district_en"].str.lower() == district]
    if df.empty:
        return jsonify(error="district not found"), 404

    # midpoint price
    df["price"] = (df["minimum"] + df["maximum"]) / 2
    cutoff = pd.Timestamp("2025-05-31")
    hist = df[df["date"] <= cutoff][["date","price"]].assign(type="actual")
    pred = df[df["date"] >  cutoff][["date","price"]].assign(type="predicted")
    combined = pd.concat([hist, pred], ignore_index=True).sort_values("date")

    out = [
      {"date": r.date.strftime("%Y-%m-%d"), "price": float(r.price), "type": r.type}
      for _, r in combined.iterrows()
    ]
    return jsonify(out)




if __name__ == '__main__':
    app.run(debug=True)

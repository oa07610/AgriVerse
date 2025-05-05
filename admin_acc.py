from werkzeug.security import generate_password_hash
from supabase_client import supabase

def signup_user():
    # Hash and insert via Supabase
    hashed = generate_password_hash('admin', method='pbkdf2:sha256')
    payload = {
        "username": 'admin',
        "email":    "admin@agriverse.com",
        "password": hashed,
        "fullname": 'admin'
    }
    return supabase.from_("user").insert(payload).execute()

signup_user()
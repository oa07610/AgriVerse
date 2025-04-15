from app import app, db
from models import User
from werkzeug.security import generate_password_hash

with app.app_context():
    admin = User(
        username='admin',
        email='admin1@agriverse.com',
        password=generate_password_hash('What_isthe_password1'),
        is_admin=True
    )
    db.session.add(admin)
    db.session.commit()

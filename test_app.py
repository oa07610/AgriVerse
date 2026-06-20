import pytest
from app import app
import json
import os
import pandas as pd
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from supabase_client import supabase

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ['TESTING'] = 'True'
    os.environ['WTF_CSRF_ENABLED'] = 'False'
    yield
    # Cleanup after tests
    try:
        # Clean up test user
        supabase.from_('user').delete().eq('username', 'testuser').execute()
    except Exception as e:
        print(f"Error cleaning up test data: {e}")

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            # Create test user
            test_user = {
                'username': 'testuser',
                'password': generate_password_hash('Test@123'),
                'email': 'test@example.com',
                'fullname': 'Test User',
                'phone': '1234567890',
                'is_admin': False
            }
            try:
                # Check if user exists
                resp = supabase.from_('user').select('id').eq('username', 'testuser').execute()
                if not resp.data:
                    supabase.from_('user').insert(test_user).execute()
            except Exception as e:
                print(f"Error handling test user: {e}")
        yield client

@pytest.fixture
def auth_client(client):
    """Client with authenticated session"""
    with client.session_transaction() as session:
        session['user_id'] = 'test-user-id'
        session['username'] = 'testuser'
    return client

@pytest.fixture
def admin_client(client):
    """Client with admin session"""
    with client.session_transaction() as session:
        session['user_id'] = 'admin-user-id'
        session['username'] = 'admin'
    return client

def test_index_route(client):
    """Test the index route returns 200 status code"""
    response = client.get('/')
    assert response.status_code == 200

def test_login_route_post_invalid(client):
    """Test login with invalid credentials"""
    response = client.post('/login', data={
        'username': 'invalid_user',
        'password': 'invalid_pass'
    })
    assert response.status_code == 200
    assert b'Invalid username or password' in response.data

def test_login_route_post_valid(client):
    """Test login with valid credentials"""
    response = client.post('/login', data={
        'username': 'testuser',
        'password': 'Test@123'
    })
    assert response.status_code == 302  # Redirects to dashboard
    with client.session_transaction() as session:
        assert 'user_id' in session
        assert 'username' in session

def test_signup_route_post_invalid_email(client):
    """Test signup with invalid email format"""
    response = client.post('/signup', data={
        'fullname': 'Test User',
        'username': 'testuser2',
        'email': 'invalid-email',
        'phone': '1234567890',
        'password': 'Test@123',
        'confirm-password': 'Test@123'
    })
    assert response.status_code == 200
    assert b'Invalid or undeliverable email address' in response.data

def test_signup_route_post_invalid_password(client):
    """Test signup with invalid password format"""
    response = client.post('/signup', data={
        'fullname': 'Test User',
        'username': 'testuser2',
        'email': 'test2@example.com',
        'phone': '1234567890',
        'password': 'weak',
        'confirm-password': 'weak'
    })
    assert response.status_code == 200
    # Check for flash message in session instead of response data
    with client.session_transaction() as session:
        flash_messages = session.get('_flashes', [])
        assert any('Password must be at least 8 characters' in msg for _, msg in flash_messages)

def test_get_sugar_data_route(client):
    """Test get_sugar_data route with valid parameters"""
    response = client.get('/get_sugar_data?crop=Sugar&by_product=&regions=Punjab,Sindh')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_get_crop_data_route(client):
    """Test get_crop_data route with valid parameters"""
    response = client.get('/get_crop_data?province=Punjab&crops=Sugar,Maize')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_get_heatmap_data_route(auth_client):
    """Test get_heatmap_data route with valid parameters"""
    response = auth_client.get('/get_heatmap_data?crop=sugar')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)
    assert 'map_html' in data
    assert 'min_price' in data
    assert 'max_price' in data

def test_list_provinces_route(client):
    """Test list_provinces route"""
    response = client.get('/list_provinces?crop=Sugar')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_list_stations_route(client):
    """Test list_stations route"""
    response = client.get('/list_stations?crop=Sugar&province=punjab')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_get_station_trend_route(auth_client):
    """Test get_station_trend route"""
    # Skip this test as the endpoint is not implemented
    pytest.skip("Station trend endpoint not implemented")

def test_forgot_password_route_get(client):
    """Test forgot_password route GET method"""
    response = client.get('/forgot-password')
    assert response.status_code == 200

def test_forgot_password_route_post_invalid(client):
    """Test forgot_password route POST method with invalid email"""
    response = client.post('/forgot-password', data={
        'email': 'nonexistent@example.com'
    })
    assert response.status_code == 200
    # Check for flash message in session instead of response data
    with client.session_transaction() as session:
        flash_messages = session.get('_flashes', [])
        assert any('Please enter a valid email address' in msg for _, msg in flash_messages)

def test_reset_password_route_get(client):
    """Test reset_password route GET method"""
    response = client.get('/reset-password/test@example.com')
    assert response.status_code == 200

def test_reset_password_route_post_invalid(client):
    """Test reset password with invalid password format"""
    response = client.post('/reset-password/test@example.com', data={
        'reset_code': '123456',
        'new_password': 'weak',
        'confirm_password': 'weak'
    })
    # The endpoint redirects on invalid password
    assert response.status_code == 302
    # Check for flash message in session
    with client.session_transaction() as session:
        flash_messages = session.get('_flashes', [])
        assert any('Password must be at least 8 characters' in msg for _, msg in flash_messages)

def test_chat_route_invalid(client):
    """Test chat route with invalid data"""
    response = client.post('/api/chat', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_chat_route_valid(client):
    """Test chat route with valid data"""
    response = client.post('/api/chat', json={
        'message': 'What is the weather like?',
        'preferred_language': 'en'
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'response' in data

def test_external_route(auth_client):
    """Test external route"""
    response = auth_client.get('/external')
    assert response.status_code == 200

def test_newsletter_route(auth_client):
    """Test newsletter route"""
    response = auth_client.get('/newsletter')
    assert response.status_code == 200

def test_admin_dashboard_route_unauthorized(client):
    """Test admin_dashboard route without authentication"""
    response = client.get('/admin')
    assert response.status_code == 302  # Redirects to login

def test_create_post_route_unauthorized(client):
    """Test create_post route without authentication"""
    response = client.get('/admin/create_post')
    assert response.status_code == 302  # Redirects to login

def test_upload_csv_route_unauthorized(client):
    """Test upload_csv route without authentication"""
    response = client.get('/admin/upload_csv')
    assert response.status_code == 302  # Redirects to login 
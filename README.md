Running the Project via Docker
Clone the repo (if you haven’t already):

git clone https://github.com/aizaimran/Agriverse.git

Open a terminal in the project root:
Windows (PowerShell / CMD):
cd C:\path\to\YourRepo

Verify you have Docker & Docker Compose installed:
docker --version
docker-compose --version

Never mount db_data with -v on your first up or it will skip the init scripts.

Start everything:
docker-compose up --build -d

This will spin up:
MySQL 8.0 on host port 3307, data persisted in the db_data volume.
Flask/Gunicorn web app on port 5000, auto-waiting for MySQL to come online.

Check containers:
docker ps --filter name=wholeupdatedcode

You should see one db container (MySQL) and one web container (your Flask app).

Open the app in your browser
→ http://localhost:5000/

To stop & tear down:
docker-compose down

To also wipe the database volume (so the init-script runs fresh next time), do:
docker-compose down -v

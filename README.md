# Makergrid

A Django-based platform connecting makers, builders, and the wider maker community — accounts, makers, community, and core modules wired together as a multi-app Django project.

## Stack

- **Backend:** Django (Python)
- **Database:** SQLite (development)
- **Apps:** `accounts`, `makers`, `community`, `core`

## Layout

```
makergrid/
├── accounts/      # User accounts & authentication
├── makers/        # Maker profiles & catalogue
├── community/     # Community features
├── core/          # Shared models, settings, routing
├── media/         # User-uploaded content
├── static/        # Static assets
├── manage.py
└── requirements.txt
```

## Local setup

```bash
python -m venv venv
source venv/bin/activate          # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>.

## Status

Active prototype. Code is shared as-is for portfolio review; production deployment lives elsewhere.

## License

This project is shared for review purposes. Contact the author before commercial use.

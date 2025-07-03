import requests
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

def save_uploaded_file_to_media(uploaded_file, filename):
    path = default_storage.save(f"uploads/{filename}", ContentFile(uploaded_file.read()))
    return settings.MEDIA_URL + path

def download_and_save_to_media(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        path = default_storage.save(f"models/{filename}", ContentFile(response.content))
        return settings.MEDIA_URL + path
    raise Exception("Failed to download file from external source.")

def normalize_text(text):
    return text.lower().replace("-", " ").replace("_", " ").strip()

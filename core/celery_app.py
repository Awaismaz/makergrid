import os
from celery import Celery
from celery.schedules import crontab  # ✅ REQUIRED for crontab()

# Set default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Create Celery app instance
app = Celery('core')

# Load config from Django settings using namespace CELERY_
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed apps
app.autodiscover_tasks()

# ✅ Use Django DB-based scheduler (django-celery-beat)
app.conf.beat_scheduler = 'django_celery_beat.schedulers:DatabaseScheduler'

# ✅ Crontab-based auto-scheduled task
app.conf.beat_schedule = {
    'daily_token_refill': {
        'task': 'accounts.tasks.refill_tokens',
        'schedule': crontab(hour=0, minute=0),  # daily at midnight
    },
}

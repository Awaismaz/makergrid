from celery import shared_task
from accounts.models import CustomUser

@shared_task
def refill_tokens():
    for user in CustomUser.objects.all():
        user.tokens = min(user.tokens + 20, 200)
        user.save()

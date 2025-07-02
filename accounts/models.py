from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.conf import settings  # ✅ Import this

import uuid

class PendingSignup(models.Model):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=150)
    full_name = models.CharField(max_length=150, blank=True)
    organization = models.CharField(max_length=150, blank=True)
    password = models.CharField(max_length=128)
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

SUBSCRIPTION_CHOICES = [
    ('free', 'Free'),
    ('maker', 'Maker'),
    ('artisan', 'Artisan'),
]

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    is_email_verified = models.BooleanField(default=False)

    # Usage tracking
    models_generated = models.IntegerField(default=0)
    last_active = models.DateTimeField(default=timezone.now)
    tokens = models.IntegerField(default=0)

    # Optional profile info
    full_name = models.CharField(max_length=100, blank=True)
    organization = models.CharField(max_length=100, blank=True)
    profile_picture = models.URLField(blank=True, null=True)

    REQUIRED_FIELDS = ['email']

    def __str__(self):
        return self.username

    @property
    def is_subscription_active(self):
        return (
            hasattr(self, 'subscription') and
            self.subscription.subscription_end and
            self.subscription.subscription_end > timezone.now()
        )


class Subscription(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='subscription')
    plan = models.CharField(max_length=20, choices=SUBSCRIPTION_CHOICES)
    stripe_customer_id = models.CharField(max_length=255)
    stripe_subscription_id = models.CharField(max_length=255)
    active = models.BooleanField(default=False)
    subscription_start = models.DateTimeField(null=True, blank=True)
    subscription_end = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.plan}"

class Purchase(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=255)
    stripe_session_id = models.CharField(max_length=255)
    paid = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)


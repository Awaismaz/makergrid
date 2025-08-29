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

class TokensPrice(models.Model):
    quantity_of_tokens = models.IntegerField(default=1)
    price_in_cents = models.IntegerField()  # Price in cents to avoid floating point issues
    stripe_price_id = models.CharField(max_length=255, unique=True)  # Corresponding Stripe Price ID
    minimum_quantity_to_buy = models.IntegerField(default=1)  # Minimum quantity required for purchase
    maximum_quantity_to_buy = models.IntegerField(default=1000)  # Maximum quantity allowed for purchase

    def __str__(self):
        return f"{self.quantity_of_tokens} tokens for ${self.price_in_cents / 100:.2f}"

class Subscription(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='subscription')
    plan = models.CharField(max_length=20, choices=SUBSCRIPTION_CHOICES)
    stripe_customer_id = models.CharField(max_length=255)
    stripe_subscription_id = models.CharField(max_length=255)
    status = models.CharField(max_length=50, default='active')
    duration_type=models.CharField(max_length=20,default="month")
    active = models.BooleanField(default=False)
    subscription_start = models.DateTimeField(null=True, blank=True)
    subscription_end = models.DateTimeField(null=True, blank=True)
    payement_method = models.CharField(max_length=255, null=True, blank=True)
    cancel_at_period_end=models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.plan}"
    

class OTP(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='otp')
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)  # Automatically set the creation time

    def is_expired(self):
        # Check if the OTP is older than 1 hour
        return timezone.now() > self.created_at + timezone.timedelta(hours=1)

class Purchase(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=255)
    stripe_session_id = models.CharField(max_length=255)
    paid = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)


class Coupon(models.Model):
    code = models.CharField(max_length=50, unique=True)  # Unique coupon code
    discount_type = models.CharField(
        max_length=20, 
        choices=[('percentage', 'Percentage'), ('flat', 'Flat')],
        default='percentage'
    )  # Type of discount: percentage or flat
    discount_value = models.DecimalField(max_digits=10, decimal_places=2)  # Discount value (e.g., 10% or $5)
    valid_from = models.DateTimeField(null=True, blank=True)  # Optional start date
    valid_until = models.DateTimeField(null=True, blank=True)  # Optional expiry date
    active = models.BooleanField(default=True)  # Whether the coupon is currently active
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp for when the coupon is created
    updated_at = models.DateTimeField(auto_now=True)  # Timestamp for when the coupon was last updated

    def __str__(self):
        return self.code

    def is_valid(self):
        """Check if the coupon is valid based on the current time."""
        now = timezone.now()
        return self.active and (not self.valid_from or self.valid_from <= now) and (not self.valid_until or self.valid_until >= now)


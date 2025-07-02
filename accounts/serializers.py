from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from .models import Subscription
from django.utils import timezone

from rest_framework import serializers
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken

class SignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = get_user_model()
        fields = ('username', 'email', 'password')

    def create(self, validated_data):
        User = get_user_model()
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            tokens=200
        )

        # Create a free subscription by default
        Subscription.objects.create(
            user=user,
            plan='free',
            active=True,
            subscription_start=timezone.now(),
            subscription_end=timezone.now() + timezone.timedelta(days=365)  # or 30
        )

        return user



class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        user = authenticate(username=data['username'], password=data['password'])

        if user is None:
            raise serializers.ValidationError("Invalid credentials")

        refresh = RefreshToken.for_user(user)
        refresh['username'] = user.username
        refresh['email'] = user.email

        # Access related subscription safely
        subscription_type = None
        if hasattr(user, 'subscription'):
            subscription_type = user.subscription.plan

        return {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'tokens': getattr(user, 'tokens', None),
                'subscription_type': subscription_type,
            }
        }


# accounts/serializers.py (or wherever your serializers live)
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['username'] = user.username
        token['email'] = user.email
        token['full_name'] = user.full_name  # if exists
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data["refresh"] = str(self.get_token(self.user))
        data["access"] = str(self.get_token(self.user).access_token)
        return data

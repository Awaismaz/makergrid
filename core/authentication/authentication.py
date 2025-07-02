# authentication.py

from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from django.contrib.auth import get_user_model

User = get_user_model()

class JWTAuthentication(BaseAuthentication):
    def authenticate(self, request):
        token = request.headers.get('Authorization')

        if not token:
            raise AuthenticationFailed('Authorization header missing')

        if token.startswith("Bearer "):
            token = token.split("Bearer ")[1]

        try:
            # Try validating the access token first
            access = AccessToken(token)
            user_id = access['user_id']

        except TokenError:
            # If access token fails, try to use refresh token
            refresh_token = request.COOKIES.get('refresh_token')
            if not refresh_token:
                raise AuthenticationFailed('Access token expired. Refresh token missing.')

            try:
                refresh = RefreshToken(refresh_token)
                new_access = refresh.access_token

                # Set new token to request (to be added in response later)
                request.META['HTTP_NEW_ACCESS'] = str(new_access)

                user_id = new_access['user_id']
            except TokenError:
                raise AuthenticationFailed('Invalid refresh token. Please login again.')

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise AuthenticationFailed('User not found')

        return (user, token)

import random
import string
import time
import json

from django.core.mail import send_mail
from django.conf import settings
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.settings import api_settings as jwt_settings

from core.authentication.authentication import JWTAuthentication
from .serializers import SignupSerializer, LoginSerializer

from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import CustomTokenObtainPairSerializer

import stripe
from django.views import View
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from .models import Purchase, Subscription,CustomUser
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from core.authentication.authentication import JWTAuthentication
import logging

from datetime import timezone as dt_timezone

logger = logging.getLogger(__name__)
stripe.api_key = settings.STRIPE_SECRET_KEY
# Temporary in-memory OTP store
temp_user_store = {}

stripe.api_key = settings.STRIPE_SECRET_KEY
logger = logging.getLogger(__name__)

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))


class SignupView(generics.GenericAPIView):
    serializer_class = SignupSerializer
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            if email in temp_user_store:
                return Response({"detail": "OTP already sent. Please verify."}, status=400)

            otp = generate_otp()
            now = time.time()
            temp_user_store[email] = {
                "data": serializer.validated_data,
                "otp": otp,
                "expires_at": now + 3600,
                "last_sent_at": now
            }

            send_mail(
                subject="Your OTP Code",
                message=f"Your OTP is {otp}. It will expire in 1 hour.",
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email],
                fail_silently=False,
            )

            return Response({"success":True,"detail": "OTP sent to your email."}, status=200)

        return Response(serializer.errors, status=400)


class VerifyOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        otp_input = request.data.get("otp")

        entry = temp_user_store.get(email)
        if not entry:
            return Response({"detail": "No signup attempt found."}, status=400)
        if time.time() > entry["expires_at"]:
            del temp_user_store[email]
            return Response({"detail": "OTP expired."}, status=400)
        if entry["otp"] != otp_input:
            return Response({"detail": "Wrong OTP."}, status=400)

        user_data = entry["data"]
        user = CustomUser.objects.create_user(
            username=user_data.get("username"),
            email=user_data.get("email"),
            password=user_data.get("password"),
            full_name=user_data.get("full_name", ""),
            organization=user_data.get("organization", ""),
            # subscription_type="free",
            tokens=200
        )
        user.is_email_verified = True
        user.save()

        del temp_user_store[email]
        return Response({"success":True,"detail": "Account created successfully."}, status=201)


class ResendOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        entry = temp_user_store.get(email)
        if not entry:
            return Response({"detail": "No signup attempt found."}, status=400)

        now = time.time()
        if now - entry["last_sent_at"] < 60:
            remaining = int(60 - (now - entry["last_sent_at"]))
            return Response({"detail": f"Please wait {remaining}s to resend OTP."}, status=429)

        new_otp = generate_otp()
        entry.update({
            "otp": new_otp,
            "expires_at": now + 300,
            "last_sent_at": now
        })

        send_mail(
            subject="Your OTP Code (Resent)",
            message=f"Your new OTP is {new_otp}. It will expire in 5 minutes.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )

        return Response({"detail": "OTP resent to your email."})


class ForgotPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        try:
            user = CustomUser.objects.get(email=email)
        except CustomUser.DoesNotExist:
            return Response({"detail": "User not found."}, status=404)

        otp = generate_otp()
        now = time.time()
        temp_user_store[email] = {
            "otp": otp,
            "expires_at": now + 300,
            "last_sent_at": now,
            "type": "reset"
        }

        send_mail(
            subject="Password Reset OTP",
            message=f"Your OTP is {otp}. It will expire in 5 minutes.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )

        return Response({"detail": "Password reset OTP sent."})


class VerifyResetOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        otp_input = request.data.get("otp")
        entry = temp_user_store.get(email)

        if not entry or entry.get("type") != "reset":
            return Response({"detail": "No password reset request found."}, status=400)
        if time.time() > entry["expires_at"]:
            del temp_user_store[email]
            return Response({"detail": "OTP expired."}, status=400)
        if entry["otp"] != otp_input:
            return Response({"detail": "Wrong OTP."}, status=400)

        return Response({"detail": "OTP verified."})


class ResetPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        new_password = request.data.get("password")
        entry = temp_user_store.get(email)

        if not entry or entry.get("type") != "reset":
            return Response({"detail": "No verified reset request."}, status=400)

        try:
            user = CustomUser.objects.get(email=email)
            user.set_password(new_password)
            user.save()
            del temp_user_store[email]
            return Response({"detail": "Password updated successfully."})
        except CustomUser.DoesNotExist:
            return Response({"detail": "User not found."}, status=404)


class ResendResetOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        entry = temp_user_store.get(email)
        if not entry or entry.get("type") != "reset":
            return Response({"detail": "No reset request found."}, status=400)

        now = time.time()
        if now - entry["last_sent_at"] < 60:
            remaining = int(60 - (now - entry["last_sent_at"]))
            return Response({"detail": f"Wait {remaining}s to resend OTP."}, status=429)

        new_otp = generate_otp()
        entry.update({
            "otp": new_otp,
            "expires_at": now + 300,
            "last_sent_at": now
        })

        send_mail(
            subject="Password Reset OTP (Resent)",
            message=f"Your new OTP is {new_otp}. It will expire in 5 minutes.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )

        return Response({"detail": "OTP resent."})




class LoginView(generics.GenericAPIView):
    serializer_class = LoginSerializer
    permission_classes = [AllowAny]
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            validated_data = serializer.validated_data
            refresh_token = validated_data.pop("refresh")
            response = Response(validated_data, status=status.HTTP_200_OK)

            refresh_lifetime = jwt_settings.REFRESH_TOKEN_LIFETIME.total_seconds()

            response.set_cookie(
                key='refresh_token',
                value=refresh_token,
                httponly=True,
                secure=not settings.DEBUG,
                samesite='Lax',
                max_age=int(refresh_lifetime),
                path='/'
            )
            return response

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




class RefreshView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [AllowAny]

    def post(self, request):
        refresh_token = request.COOKIES.get('refresh_token')
        if not refresh_token:
            return Response({"detail": "Refresh token missing."}, status=400)

        try:
            token = RefreshToken(refresh_token)
            access_token = str(token.access_token)
            return Response({"access": access_token})
        except TokenError:
            return Response({"detail": "Invalid refresh token."}, status=403)


class CreditTokensView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response({"credits": request.user.tokens})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    user = request.user
    subscription = getattr(user, 'subscription', None)

    return Response({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'tokens': user.tokens,
        'subscription_type': subscription.plan if subscription else 'free',
        'subscription_end': subscription.subscription_end if subscription else None,
        'is_email_verified': user.is_email_verified,
    })


class LogoutView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        response = Response({"detail": "Logged out successfully."})
        response.delete_cookie("refresh_token")
        return response
    


# views.py (Django)
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def create_checkout_session(request):
    domain = settings.FRONTEND_DOMAIN
    plan = request.data.get('plan')

    price_ids = {
        'maker': 'price_1RNJZiP2zGsc8dEjrAohl7Ab',
        'artisan': 'price_1RNJafP2zGsc8dEj5vY1pAHy',
    }

    price_id = price_ids.get(plan)
    if not price_id:
        return Response({'error': 'Invalid plan'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            mode='subscription',
            line_items=[{'price': price_id, 'quantity': 1}],
            customer_email=request.user.email,
            success_url=f'{domain}/account/billing-success/?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{domain}/cancel/',
        )
        return Response({'id': session.id})
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@csrf_exempt
def stripe_webhook(request):
    payload = request.body
    sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
    endpoint_secret = settings.STRIPE_WEBHOOK_SECRET

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        logger.info(f"✅ Received Stripe webhook: {event['type']}")
    except ValueError as e:
        logger.error(f"❌ Invalid payload: {e}")
        return JsonResponse({'error': 'Invalid payload'}, status=400)
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"❌ Signature verification failed: {e}")
        return JsonResponse({'error': 'Invalid signature'}, status=400)

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        logger.info(f"📦 Handling checkout.session.completed for session: {session.get('id')}")

        customer_email = session.get('customer_email')
        stripe_subscription_id = session.get('subscription')
        stripe_customer_id = session.get('customer')

        if not customer_email or not stripe_subscription_id:
            logger.error("❌ Missing customer_email or subscription ID")
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        try:
            user = CustomUser.objects.get(email=customer_email)
        except CustomUser.DoesNotExist:
            logger.error(f"❌ No user found with email: {customer_email}")
            return JsonResponse({'error': 'User not found'}, status=404)

        try:
            stripe_subscription = stripe.Subscription.retrieve(stripe_subscription_id)
            print("⚡️ Stripe Subscription Dump:", json.dumps(stripe_subscription, indent=2))

            items = stripe_subscription.get('items', {}).get('data', [])
            if not items:
                logger.warning(f"⚠️ Subscription {stripe_subscription_id} has no items")
                return JsonResponse({'error': 'No items in subscription'}, status=400)

            plan_id = items[0].get('price', {}).get('id')
            plan_mapping = {
                'price_1RNJZiP2zGsc8dEjrAohl7Ab': 'maker',
                'price_1RNJafP2zGsc8dEj5vY1pAHy': 'artisan',
            }
            plan = plan_mapping.get(plan_id, 'free')

            period_start = items[0].get('current_period_start')
            period_end = items[0].get('current_period_end')


            if period_start is None or period_end is None:
                logger.warning("⚠️ Stripe subscription missing period timestamps")
                return JsonResponse({'error': 'Missing period timestamps'}, status=400)

            start = timezone.datetime.fromtimestamp(period_start, tz=dt_timezone.utc)
            end = timezone.datetime.fromtimestamp(period_end, tz=dt_timezone.utc)

            Subscription.objects.update_or_create(
                user=user,
                defaults={
                    'plan': plan,
                    'stripe_customer_id': stripe_customer_id,
                    'stripe_subscription_id': stripe_subscription_id,
                    'active': True,
                    'subscription_start': start,
                    'subscription_end': end,
                }
            )

            logger.info(f"✅ Subscription updated for {user.email} → {plan}")
            return JsonResponse({'status': 'subscription saved'})

        except Exception as e:
            logger.exception(f"❌ Unexpected error while processing session: {e}")
            return JsonResponse({'error': 'Failed to sync subscription'}, status=500)

    return JsonResponse({'status': 'ignored'})


@api_view(['GET'])
def validate_session(request, session_id):
    try:
        # Fetch the session from Stripe
        session = stripe.checkout.Session.retrieve(session_id)
        customer_email = session.get('customer_email')

        if not customer_email:
            return Response({'error': 'Customer email not found in session'}, status=status.HTTP_400_BAD_REQUEST)

        # Find user and return minimal info to trigger frontend refresh
        try:
            user = CustomUser.objects.get(email=customer_email)
        except CustomUser.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

        logger.info(f"✅ Stripe session validated for user: {user.email}")
        return Response({'status': 'success'}, status=status.HTTP_200_OK)

    except stripe.error.InvalidRequestError as e:
        logger.error(f"❌ Stripe error: {e}")
        return Response({'error': 'Invalid session ID'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        return Response({'error': 'Server error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class BuyModelCheckoutSession(View):
#     def post(self, request, model_id, *args, **kwargs):
#         # Replace this with your actual 3D model lookup logic
#         model = get_object_or_404(YourModel, id=model_id)

#         domain = "https://yourdomain.com"
#         session = stripe.checkout.Session.create(
#             customer_email=request.user.email,
#             payment_method_types=['card'],
#             line_items=[{
#                 'price_data': {
#                     'currency': 'usd',
#                     'product_data': {'name': model.name},
#                     'unit_amount': int(model.price * 100),
#                 },
#                 'quantity': 1,
#             }],
#             mode='payment',
#             success_url=domain + '/purchase-success/',
#             cancel_url=domain + '/cancel/',
#         )

#         Purchase.objects.create(
#             user=request.user,
#             model_name=model.name,
#             stripe_session_id=session.id
#         )

#         return JsonResponse({'id': session.id})

# @method_decorator(csrf_exempt, name='dispatch')
# class StripeWebhookView(View):
#     def post(self, request, *args, **kwargs):
#         payload = request.body
#         sig_header = request.META['HTTP_STRIPE_SIGNATURE']
#         endpoint_secret = settings.STRIPE_WEBHOOK_SECRET

#         try:
#             event = stripe.Webhook.construct_event(
#                 payload, sig_header, endpoint_secret
#             )
#         except ValueError:
#             return HttpResponse(status=400)
#         except stripe.error.SignatureVerificationError:
#             return HttpResponse(status=400)

#         # Handle subscription created
#         if event['type'] == 'checkout.session.completed':
#             session = event['data']['object']
#             customer_email = session.get('customer_email')
#             customer = stripe.Customer.retrieve(session['customer'])
#             user = User.objects.filter(email=customer_email).first()

#             if session['mode'] == 'subscription':
#                 Subscription.objects.update_or_create(
#                     user=user,
#                     defaults={
#                         'stripe_customer_id': customer.id,
#                         'stripe_subscription_id': session['subscription'],
#                         'active': True,
#                     }
#                 )
#             elif session['mode'] == 'payment':
#                 Purchase.objects.filter(stripe_session_id=session.id).update(paid=True)

#         return HttpResponse(status=200)


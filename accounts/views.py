import random
import string
import time
import json
import threading
from datetime import datetime
from datetime import timedelta
otp_lock = threading.Lock()
from django.db.models import F
import smtplib
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
from decimal import Decimal
from core.authentication.authentication import JWTAuthentication
from .serializers import SignupSerializer, LoginSerializer

from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import CustomTokenObtainPairSerializer
from .models import CustomUser, Coupon
import stripe
from django.views import View
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from .models import Purchase, Subscription,CustomUser,PendingSignup,TokensPrice,OTP
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
    return str(random.randint(100000, 999999))



class SignupView(generics.GenericAPIView):
    serializer_class = SignupSerializer
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        # Start by logging the request for debugging
        logger.info(f"Signup request received for email: {request.data.get('email')}")

        # Deserialize the request data using the serializer
        serializer = self.get_serializer(data=request.data)
        
        # Validate the serializer
        if serializer.is_valid():
            email = serializer.validated_data['email']

            # Check if the email already exists in the PendingSignup
            # if PendingSignup.objects.filter(email=email).exists():
            #     logger.warning(f"OTP already sent to {email}")
            #     return Response({"detail": "OTP already sent. Please verify."}, status=400)

            try:
                # Generate OTP and set expiration time
                otp = generate_otp()
                expires_at = timezone.now() + timedelta(hours=1)

                # Create the PendingSignup object
                PendingSignup.objects.create(
                    email=email,
                    username=serializer.validated_data["username"],
                    full_name=serializer.validated_data.get("full_name", ""),
                    organization=serializer.validated_data.get("organization", ""),
                    password=serializer.validated_data["password"],
                    otp=otp,
                    expires_at=expires_at
                )

                print(f"Generated OTP: {otp} for email: {email}")

                # Send the OTP to the user's email
                send_mail(
                    subject="Your OTP Code for Makergrid Account Registration",
                    message=f"Your OTP is {otp}. It will expire in 1 hour.",
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[email],
                    fail_silently=False,
                )

                # sender_email = settings.DEFAULT_FROM_EMAIL # Hostinger webmail email
                # host_email = settings.EMAIL_HOST_USER  # Hostinger email
                # password = settings.EMAIL_HOST_PASSWORD  # App password for your Gmail account (but it should be used for Gmail SMTP)
                # receiver_email = email
                # message = f"Your OTP is {otp}. It will expire in 1 hour."

                # try:
                #     # Connect to Hostinger's SMTP server with SSL (port 465)
                #     with smtplib.SMTP_SSL("smtp.hostinger.com", 465) as server:
                #         server.login(host_email, password)  # Log in with Hostinger email credentials
                #         server.sendmail(sender_email, receiver_email, message)
                #     print("Test email sent successfully!")
                # except Exception as e:
                #     print(f"Error: {e}")

                logger.info(f"OTP sent to {email}")
                return Response({"success": True, "detail": "OTP sent to your email."}, status=200)

            except Exception as e:
                logger.error(f"Error occurred while processing signup for {email}: {str(e)}")
                return Response({"detail": "An error occurred while processing your request. Please try again later."}, status=500)

        # If serializer is invalid, return errors
        logger.error(f"Invalid data: {serializer.errors}")
        return Response(serializer.errors, status=400)
# class SignupView(generics.GenericAPIView):
#     serializer_class = SignupSerializer
#     parser_classes = [JSONParser, FormParser, MultiPartParser]
#     permission_classes = [AllowAny]

#     def post(self, request):
#         serializer = self.get_serializer(data=request.data)
#         if serializer.is_valid():
#             email = serializer.validated_data['email']
#             if PendingSignup.objects.filter(email=email).exists():
#                 return Response({"detail": "OTP already sent. Please verify."}, status=400)

#             otp = generate_otp()
#             expires_at = timezone.now() + timedelta(hours=1)

#             PendingSignup.objects.create(
#                 email=email,
#                 username=serializer.validated_data["username"],
#                 full_name=serializer.validated_data.get("full_name", ""),
#                 organization=serializer.validated_data.get("organization", ""),
#                 password=serializer.validated_data["password"],
#                 otp=otp,
#                 expires_at=expires_at
#             )

#             send_mail(
#                 subject="Your OTP Code",
#                 message=f"Your OTP is {otp}. It will expire in 1 hour.",
#                 from_email=settings.DEFAULT_FROM_EMAIL,
#                 recipient_list=[email],
#                 fail_silently=False,
#             )

#             return Response({"success": True, "detail": "OTP sent to your email."}, status=200)

#         return Response(serializer.errors, status=400)


class VerifyPasswordResetOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        otp_input = request.data.get("otp")

        if not email or not otp_input:
            return Response({"detail": "Email and OTP are required."}, status=400)

        try:
            # Retrieve the OTP object associated with the user
            user = CustomUser.objects.get(email=email)
            otp_record = OTP.objects.get(user=user)
        except CustomUser.DoesNotExist:
            return Response({"detail": "User not found."}, status=404)
        except OTP.DoesNotExist:
            return Response({"detail": "OTP not found for this user."}, status=404)

        # Check if the OTP is expired
        if otp_record.is_expired():
            otp_record.delete()  # Optionally delete expired OTPs
            return Response({"detail": "OTP has expired. Please request a new one."}, status=400)

        # Verify if the OTP matches
        if otp_record.otp != otp_input:
            return Response({"detail": "Invalid OTP."}, status=400)

        # OTP is valid, allow password reset (you can now proceed to the next step)
        return Response({"success": "OTP is valid. You can now reset your password."}, status=200)

class VerifyOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        otp_input = request.data.get("otp")

        try:
            signup = PendingSignup.objects.get(email=email)
        except PendingSignup.DoesNotExist:
            return Response({"detail": "No signup attempt found."}, status=400)

        if timezone.now() > signup.expires_at:
            signup.delete()
            return Response({"detail": "OTP expired."}, status=400)

        if signup.otp != otp_input:
            return Response({"detail": "Wrong OTP."}, status=400)

        user = CustomUser.objects.create_user(
            username=signup.username,
            email=signup.email,
            password=signup.password,
            full_name=signup.full_name,
            organization=signup.organization,
            tokens=100
        )
        user.is_email_verified = True
        user.save()
        signup.delete()

        return Response({"success": True, "detail": "Account created successfully."}, status=201)

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


def validate_otp(user, otp_input):
    try:
        otp = OTP.objects.get(user=user)
        if otp.is_expired():
            otp.delete()  # Delete expired OTP
            return "OTP has expired, please request a new one"
        if otp.otp != otp_input:
            return "Invalid OTP"
        return "OTP valid"
    except OTP.DoesNotExist:
        return "OTP does not exist"
# class ForgotPassword(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         email = request.data.get("email")

#         if email :
#             try:
#                 user = CustomUser.objects.get(email=email)
#             except CustomUser.DoesNotExist:
#                 return Response({"detail": f"User not found for email : {email}","status":"404"}, status=200)

#         # Generate OTP
#         otp = generate_otp()

#         # Save OTP in the OTP model
#         otp_record, created = OTP.objects.update_or_create(
#             user=user,
#             defaults={'otp': otp}
#         )

#         # Expiration time for the OTP
#         otp_record.created_at = time.time()
#         otp_record.save()

#         # Send OTP to the user via email
#         send_mail(
#             subject="Password Reset OTP",
#             message=f"Your OTP is {otp}. It will expire in 1 hour.",
#             from_email=settings.DEFAULT_FROM_EMAIL,
#             recipient_list=[email],
#             fail_silently=False,
#         )

#         return Response({"success": "Password reset OTP sent."}, status=200)


class ForgotPassword(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")

        if email:
            try:
                user = CustomUser.objects.get(email=email)
            except CustomUser.DoesNotExist:
                return Response({"detail": f"User not found for email : {email}", "status": "404"}, status=200)

        # Generate OTP
        otp = generate_otp()

        # Save OTP in the OTP model
        otp_record, created = OTP.objects.update_or_create(
            user=user,
            defaults={'otp': otp}
        )

        # Expiration time for the OTP (use timezone.now() instead of time.time())
        otp_record.created_at = timezone.now()  # Use timezone.now() to get a datetime object
        otp_record.save()

        # Send OTP to the user via email
        send_mail(
            subject="Password Reset OTP",
            message=f"Your OTP is {otp}. It will expire in 1 hour.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )

        return Response({"success": "Password reset OTP sent."}, status=200)


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
        try:
            user = CustomUser.objects.get(email=email)
            user.set_password(new_password)
            user.save()
            del temp_user_store[email]
            return Response({"detail": "Password updated successfully.","success":True})
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
    
class BlenderLoginView(generics.GenericAPIView):
    serializer_class = LoginSerializer
    permission_classes = [AllowAny]
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            validated_data = serializer.validated_data
            # Just return the validated data without setting cookies
            return Response(validated_data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class RefreshView(generics.GenericAPIView):
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
        return Response({"credits": request.user.tokens,"subscription_type": request.user.subscription.plan if hasattr(request.user, 'subscription') else 'free'})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_required_data_view(request):
    user = request.user
    
    # Get the first token price (example)
    token_price = TokensPrice.objects.all().order_by('quantity_of_tokens').first()
    
    # Get subscription end date
    try:
        sub_end_date = user.subscription.subscription_end
    except Subscription.DoesNotExist:
        sub_end_date = None  # User has no subscription

    sub_end_date_naive = sub_end_date.replace(tzinfo=None)

    payload = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'tokens': user.tokens,
        'token_quantity': token_price.quantity_of_tokens if token_price else None,
        "token_price": token_price.price_in_cents if token_price else None,
        "sub_end_date": sub_end_date_naive
    }

    print(f"Payload : {payload}")

    return Response(payload)





@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    user = request.user
    subscription = getattr(user, 'subscription', None)
    
    # Ensure subscription_end is not None before comparing
    if subscription and subscription.cancel_at_period_end and subscription.subscription_end:
        if subscription.subscription_end < timezone.now():
            # Subscription has ended
            # subscription.active = False
            # subscription.plan = 'free'
            # subscription.save()
            canceled_sub = True
    else:
        canceled_sub = False


    sub_end_date=subscription.subscription_end
    sub_end_date_naive = sub_end_date.replace(tzinfo=None)


    return Response({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'tokens': user.tokens,
        'subscription_type': subscription.plan if subscription else 'free',
        'subscription_end':  sub_end_date_naive if subscription else None,
        'is_email_verified': user.is_email_verified,
        "payement_method": subscription.payement_method if subscription else None,
        'sub_canceled': subscription.cancel_at_period_end if subscription else None
    })



class LogoutView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        response = Response({"detail": "Logged out successfully."})
        response.delete_cookie("refresh_token")
        return response
    



# @api_view(['POST'])
# @authentication_classes([JWTAuthentication])
# @permission_classes([IsAuthenticated])
# def create_checkout_session(request):
#     domain = settings.FRONTEND_DOMAIN
#     plan = request.data.get('plan')
#     coupon_code = request.data.get('coupon_code')  # Get the coupon code from request
#     # coupon_code = "mg-216"

#     # Test prices
#     price_ids = {
#         'maker-annually': 'price_1RnSnvCZA4DdscMXvP9WByuM',
#         'artisan-annually': 'price_1RnSpFCZA4DdscMXerr4Xpqw',
#         'maker-monthly': 'price_1RnSmGCZA4DdscMX8wGzeNys',
#         'artisan-monthly': 'price_1RnSmyCZA4DdscMXbDkoz88E',
#     }

#     price_id = price_ids.get(plan)
    
#     if plan == 'maker-annually' or plan == 'maker-monthly':
#         metaPlan = "maker"     
#     elif plan == 'artisan-annually' or plan == 'artisan-monthly':
#         metaPlan = "artisan"
#     if not price_id:
#         return Response({'error': 'Invalid plan'}, status=status.HTTP_400_BAD_REQUEST)

#     # Check if coupon code is provided
#     # discount = 0
#     # if coupon_code:
#     #     try:
#     #         coupon = Coupon.objects.get(code=coupon_code, active=True)
#     #         # Validate coupon date range
#     #         if not coupon.is_valid():
#     #             print("Coupon is expired or inactive")
#     #             return Response({'error': 'Coupon code is expired or inactive'}, status="200")

#     #         # Apply the discount based on the coupon type
#     #         if coupon.discount_type == 'percentage':
#     #             discount = coupon.discount_value / 100  # Convert to decimal
#     #         elif coupon.discount_type == 'flat':
#     #             discount = coupon.discount_value

#     #     except Coupon.DoesNotExist:
#     #         return Response({'error': 'Invalid coupon code'}, status=status.HTTP_400_BAD_REQUEST)
   
#     try:
#         session = stripe.checkout.Session.create(
#             payment_method_types=['card'],
#             mode='subscription',
#             line_items=[{'price': price_id, 'quantity': 1}],
#             customer_email=request.user.email,
#             success_url=f'{domain}/account/billing-success/?session_id={{CHECKOUT_SESSION_ID}}',
#             cancel_url=f'{domain}/account/billing-cancel',
#             metadata={'plan': metaPlan, 'coupon_code': coupon_code if coupon_code else None},
#         )
#         return Response({'id': session.id})
#     except stripe.error.StripeError as e:
#         return Response({'error': f"Stripe error: {e.user_message}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#     except Exception as e:
#         return Response({'error': f"General error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _get_or_create_stripe_customer(user: CustomUser) -> str:
    sub = getattr(user, 'subscription', None)
    if sub and sub.stripe_customer_id:
        try:
            stripe.Customer.retrieve(sub.stripe_customer_id)  # Check if it still exists
            return sub.stripe_customer_id
        except stripe.error.InvalidRequestError:
            # Customer doesn't exist on Stripe anymore, reset
            sub.stripe_customer_id = None
            sub.save()

    # If no valid customer exists, search by email in Stripe
    customers = stripe.Customer.list(email=user.email, limit=1).data
    if customers:
        customer_id = customers[0].id
    else:
        customer = stripe.Customer.create(
            email=user.email,
            name=user.get_full_name() or user.username
        )
        customer_id = customer.id

    Subscription.objects.update_or_create(
        user=user,
        defaults={'stripe_customer_id': customer_id}
    )
    return customer_id


def _maybe_resolve_promotion_code_id(code: str) -> str | None:
    """
    Users type human-readable codes (e.g., SAVE20). Stripe needs a 'promotion_code' id.
    Returns the promotion_code id if found/enabled.
    """
    try:
        promo_list = stripe.PromotionCode.list(code=code, active=True, limit=1)
        if promo_list.data:
            return promo_list.data[0].id
    except stripe.error.StripeError:
        print("Stripe error when looking up promotion code",stripe.error.StripeError)
        pass
    return None


# @api_view(['POST'])
# @authentication_classes([JWTAuthentication])
# @permission_classes([IsAuthenticated])
# def create_checkout_session(request):
#     domain = settings.FRONTEND_DOMAIN
#     plan = request.data.get('plan')
#     coupon_code = request.data.get('coupon_code')  # Get the coupon code from request

#     # Test prices
#     price_ids = {
#         'maker-annually': 'price_1RnSnvCZA4DdscMXvP9WByuM',
#         'artisan-annually': 'price_1RnSpFCZA4DdscMXerr4Xpqw',
#         'maker-monthly': 'price_1RnSmGCZA4DdscMX8wGzeNys',
#         'artisan-monthly': 'price_1RnSmyCZA4DdscMXbDkoz88E',
#     }

#     price_id = price_ids.get(plan)
    
#     if plan == 'maker-annually' or plan == 'maker-monthly':
#         metaPlan = "maker"     
#     elif plan == 'artisan-annually' or plan == 'artisan-monthly':
#         metaPlan = "artisan"
    
#     if not price_id:
#         return Response({'error': 'Invalid plan'}, status=status.HTTP_400_BAD_REQUEST)

#     try:
#         # Create Stripe Checkout session with or without coupon code
#         session_params = {
#             'payment_method_types': ['card'],
#             'mode': 'subscription',
#             'line_items': [{'price': price_id, 'quantity': 1}],
#             'customer_email': request.user.email,
#             'success_url': f'{domain}/account/billing-success/?session_id={{CHECKOUT_SESSION_ID}}',
#             'cancel_url': f'{domain}/account/billing-cancel',
#             'metadata': {'plan': metaPlan, 'coupon_code': coupon_code if coupon_code else None},
#         }
        
#         # Add coupon code to session if it's provided
#         if coupon_code:
#             session_params['discounts'] = [{
#                 'coupon': coupon_code  # Apply the coupon from Stripe Dashboard
#             }]

#         session = stripe.checkout.Session.create(**session_params)

#         return Response({'id': session.id})
    
#     except stripe.error.StripeError as e:
#         return Response({'error': f"Stripe error: {e.user_message}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#     except Exception as e:
#         return Response({'error': f"General error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def create_checkout_session_for_token(request):
    domain = settings.FRONTEND_DOMAIN
    # plan = request.data.get('plan')
    # print("Plan selected:", plan)
    # coupon_code = request.data.get('coupon_code')  # can be coupon id or human code (promotion code)

    price_id= TokensPrice.objects.all().order_by('quantity_of_tokens').first()
    if not price_id:    
        print("No token price available")
        return Response({'error': 'Invalid plan'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        customer_id = _get_or_create_stripe_customer(request.user)

        # session = stripe.checkout.Session.create(
        #     mode='payment',
        #     customer=customer_id,
        #     payment_method_types=['card'],
        #     line_items=[{'price': price_id.stripe_price_id , 'quantity': 1}],
        #     success_url=f'{domain}/account/tokens/billing-success/?session_id={{CHECKOUT_SESSION_ID}}',
        #     cancel_url=f'{domain}/account/tokens/billing-cancel',
        #     metadata={'user_id': request.user.id, 'tokens_price_id': price_id.id},
        # )

        minimum_quantity = TokensPrice.objects.all().order_by('quantity_of_tokens').first().minimum_quantity_to_buy if TokensPrice.objects.exists() else 1
        maximum_quantity = TokensPrice.objects.all().order_by('-quantity_of_tokens').first().maximum_quantity_to_buy if TokensPrice.objects.exists() else 1000

        session = stripe.checkout.Session.create(
            mode='payment',
            customer=customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': price_id.stripe_price_id,
                'quantity': minimum_quantity,  # default starting quantity
                'adjustable_quantity': {
                    'enabled': True,
                    'minimum': minimum_quantity,
                    'maximum': maximum_quantity  # or any upper limit you decide
                }
            }],
            success_url=f'{domain}/account/tokens/billing-success/?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{domain}/account/tokens/billing-cancel',
            metadata={
                'user_id': request.user.id,
                'tokens_price_id': price_id.id
            }
        )

        return Response({'id': session.id,'url': session.url})
    except stripe.error.StripeError as e:
        print("Stripe error:", e.user_message)
        return Response({'error': f"stripe_error: {e.user_message or str(e)}"}, status=500)
    except Exception as e:
        print("General error:", e)
        return Response({'error': f"General error: {str(e)}"}, status=500)


@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def create_checkout_session(request):
    domain = settings.FRONTEND_DOMAIN
    plan = request.data.get('plan')
    print("Plan selected:", plan)
    coupon_code = request.data.get('coupon_code')  # can be coupon id or human code (promotion code)

    # price_ids = {
    #     'maker-annually': 'price_1RnSnvCZA4DdscMXvP9WByuM',
    #     'artisan-annually': 'price_1RnSpFCZA4DdscMXerr4Xpqw',
    #     'maker-monthly': 'price_1RnSmGCZA4DdscMX8wGzeNys',
    #     'artisan-monthly': 'price_1RnSmyCZA4DdscMXbDkoz88E',
    # }
    price_ids = {
        'maker-annually': 'price_1RnSnvCZA4DdscMXvP9WByuM',
        'artisan-annually': 'price_1RnSpFCZA4DdscMXerr4Xpqw',
        'maker-monthly': 'price_1S1XZQE8fUU6TnbUV4hfTUG1',
        'artisan-monthly': 'price_1RnSmyCZA4DdscMXbDkoz88E',
    }
    # price_ids = {
    #     'maker-annually': 'price_1RnSnvCZA4DdscMXvP9WByuM',
    #     'artisan-annually': 'price_1RnSpFCZA4DdscMXerr4Xpqw',
    #     'maker-monthly': 'price_1RnSumE8fUU6TnbUjcboZ6nM',
    #     'artisan-monthly': 'price_1RnSmyCZA4DdscMXbDkoz88E',
    # }

    duration_types = {
        'maker-annually': "year",
        'artisan-annually': "year",
        'maker-monthly': "month",
        'artisan-monthly':"month",
    }
    price_id = price_ids.get(plan)
    duration_type = duration_types.get(plan)
    if not price_id:
        return Response({'error': 'Invalid plan'}, status=status.HTTP_400_BAD_REQUEST)

    meta_plan = 'maker' if plan.startswith('maker') else 'artisan'

    try:
        customer_id = _get_or_create_stripe_customer(request.user)

        session_params = {
            "mode":'subscription',
            "customer":customer_id,  # ensures default_payment_method gets attached to the customer
            "payment_method_types":['card'],
            "line_items":[{'price': price_id, 'quantity': 1}],
            "success_url":f'{domain}/account/billing-success/?session_id={{CHECKOUT_SESSION_ID}}',
            "cancel_url":f'{domain}/account/billing-cancel',
            # This is important: charge automatically on renewals.
            "subscription_data":{
                'metadata': {'plan': meta_plan,"duration_type":duration_type},
                # 'payment_behavior': 'default_incomplete',  # Checkout will finalize after payment_method is set
            },
            "metadata":{'plan': meta_plan,"duration_type":duration_type}
            }
        
        if coupon_code:
            session_params['discounts'] = [{
                'coupon': coupon_code  # Apply the coupon from Stripe Dashboard
            }]
        
        session = stripe.checkout.Session.create(
            **session_params
        )

        return Response({'id': session.id,'url': session.url})
    except stripe.error.StripeError as e:
        print("Stripe error:", e.user_message)
        if e.user_message.startswith("No such coupon"):
            return Response({'error': 'Invalid coupon code',"coupon_error":True}, status=200)
        return Response({'error': f"stripe_error: {e.user_message or str(e)}"}, status=500)
    except Exception as e:
        print("General error:", e)
        return Response({'error': f"General error: {str(e)}"}, status=500)

# @csrf_exempt
# def stripe_webhook(request):
#     payload = request.body
#     sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
#     endpoint_secret = settings.STRIPE_WEBHOOK_SECRET

#     try:
#         event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
#         logger.info(f"✅ Received Stripe webhook: {event['type']}")
#     except ValueError as e:
#         logger.error(f"❌ Invalid payload: {e}")
#         return JsonResponse({'error': 'Invalid payload'}, status=400)
#     except stripe.error.SignatureVerificationError as e:
#         logger.error(f"❌ Signature verification failed: {e}")
#         return JsonResponse({'error': 'Invalid signature'}, status=400)

#     if event['type'] == 'checkout.session.completed':
#         session = event['data']['object']
#         logger.info(f"📦 Handling checkout.session.completed for session: {session.get('id')}")

#         customer_email = session.get('customer_email')
#         stripe_subscription_id = session.get('subscription')
#         stripe_customer_id = session.get('customer')

#         if not customer_email or not stripe_subscription_id:
#             logger.error("❌ Missing customer_email or subscription ID")
#             return JsonResponse({'error': 'Missing required fields'}, status=400)

#         try:
#             user = CustomUser.objects.get(email=customer_email)
#         except CustomUser.DoesNotExist:
#             logger.error(f"❌ No user found with email: {customer_email}")
#             return JsonResponse({'error': 'User not found'}, status=404)

#         try:
#             stripe_subscription = stripe.Subscription.retrieve(stripe_subscription_id)
#             print("⚡️ Stripe Subscription Dump:", json.dumps(stripe_subscription, indent=2))

#             items = stripe_subscription.get('items', {}).get('data', [])
#             if not items:
#                 logger.warning(f"⚠️ Subscription {stripe_subscription_id} has no items")
#                 return JsonResponse({'error': 'No items in subscription'}, status=400)

#             plan_id = items[0].get('price', {}).get('id')
#             plan_mapping = {
#                 'price_1RNJZiP2zGsc8dEjrAohl7Ab': 'maker',
#                 'price_1RNJafP2zGsc8dEj5vY1pAHy': 'artisan',
#             }
#             plan = plan_mapping.get(plan_id, 'free')

#             period_start = items[0].get('current_period_start')
#             period_end = items[0].get('current_period_end')


#             if period_start is None or period_end is None:
#                 logger.warning("⚠️ Stripe subscription missing period timestamps")
#                 return JsonResponse({'error': 'Missing period timestamps'}, status=400)

#             start = timezone.datetime.fromtimestamp(period_start, tz=dt_timezone.utc)
#             end = timezone.datetime.fromtimestamp(period_end, tz=dt_timezone.utc)

#             Subscription.objects.update_or_create(
#                 user=user,
#                 defaults={
#                     'plan': plan,
#                     'stripe_customer_id': stripe_customer_id,
#                     'stripe_subscription_id': stripe_subscription_id,
#                     'active': True,
#                     'subscription_start': start,
#                     'subscription_end': end,
#                 }
#             )

#             # update_tokens_based_on_plan(user, plan)

#             logger.info(f"✅ Subscription updated for {user.email} → {plan}")
#             return JsonResponse({'status': 'subscription saved'})

#         except Exception as e:
#             logger.exception(f"❌ Unexpected error while processing session: {e}")
#             return JsonResponse({'error': 'Failed to sync subscription'}, status=500)

#     return JsonResponse({'status': 'ignored'})


def get_sub_data (sub_id):
    subscription = stripe.Subscription.retrieve(sub_id)
    print (f"DATA : {subscription}")


def _sync_subscription(sub_obj, session_obj=None):
        # print(f"DATA : {sub_obj}")
        customer_id = sub_obj.get('customer')
        stripe_subscription_id = sub_obj.get('id')
        items = (sub_obj.get('items') or {}).get('data', [])
        status_s = sub_obj.get('status')  # 'active', 'trialing', 'past_due', 'canceled', etc.

        # Prefer subscription metadata; optionally fall back to session metadata (when called from checkout.session.completed)
        meta = (sub_obj.get('metadata') or {})
        if not meta and session_obj:
            meta = (session_obj.get('metadata') or {})

        duration_type = meta.get('duration_type')  # 'monthly' / 'annually' (or whatever you set)
        meta_plan = meta.get('plan')  # optional fallback for plan
        # Map plan by price id, with fallback to metadata.plan
        plan_id = items[0].get('price', {}).get('id') if items else None
        plan_mapping = {
            'price_1RnSnvCZA4DdscMXvP9WByuM': 'maker',   # maker-annually
            'price_1RnSmGCZA4DdscMX8wGzeNys': 'maker',   # maker-monthly
            'price_1RnSpFCZA4DdscMXerr4Xpqw': 'artisan', # artisan-annually
            'price_1RnSmyCZA4DdscMXbDkoz88E': 'artisan', # artisan-monthly
        }
        plan = plan_mapping.get(plan_id) or meta_plan or 'free'

        for item in sub_obj['items']['data']:
            current_period_start = item['current_period_start']
            current_period_end = item['current_period_end']

            print(f"current_period_start : {current_period_start}")
            print(f"current_period_end : {current_period_end}")
            
            # Use timezone-aware datetime objects
            current_period_start_date = datetime.fromtimestamp(current_period_start, tz=dt_timezone.utc)
            current_period_end_date = datetime.fromtimestamp(current_period_end, tz=dt_timezone.utc)
            
            print(f"current_period_start_date (timezone-aware): {current_period_start_date}")
            print(f"current_period_end_date (timezone-aware): {current_period_end_date}")

        
        # current_period_start = sub_obj.get('current_period_start')
        # current_period_end = sub_obj.get('current_period_end')

        # current_period_start_date = datetime.utcfromtimestamp(current_period_start_timestamp)
        # current_period_end_date = datetime.utcfromtimestamp(current_period_end_timestamp)

        # Find user by our saved customer_id; fallback to email on the Stripe customer
        subs = Subscription.objects.filter(stripe_customer_id=customer_id)
        if subs.exists():
            user = subs.first().user
        else:
            cust = stripe.Customer.retrieve(customer_id) if customer_id else None
            email = (cust or {}).get('email')
            if not email:
                return
            try:
                user = CustomUser.objects.get(email=email)
            except CustomUser.DoesNotExist:
                return

        start_dt = timezone.datetime.fromtimestamp(current_period_start, tz=dt_timezone.utc) if current_period_start else None
        end_dt   = timezone.datetime.fromtimestamp(current_period_end, tz=dt_timezone.utc) if current_period_end else None

        active_flag = status_s in ('active', 'trialing', 'past_due')  # grace on past_due

        # Default PM may be on the subscription OR (fallback) on the customer invoice settings
        pm_id = sub_obj.get('default_payment_method')
        if not pm_id and customer_id:
            cust = stripe.Customer.retrieve(customer_id)
            pm_id = ((cust or {}).get('invoice_settings') or {}).get('default_payment_method')

        if duration_type == "year":
            # Add 12 months to the current date
            end_date = timezone.now() + timedelta(days=365)  # Roughly 1 year (considering leap years)

        elif duration_type == "month":
            # Add 30 days to the current date
            end_date = timezone.now() + timedelta(days=30)

        
        current_tokens = user.tokens
        print(f'Previous Tokens : {current_tokens}')
        if meta_plan == "maker" and duration_type == "month" :
            print("MAKER PLAN IS READY")
            user.tokens += 1000  # Add 1000 to the current token value
            user.save()
        # elif meta_plan is "maker" and duration_type is "year" :
        #     add_tokens = 4000
        elif meta_plan == "artisan" and duration_type == "month" :
            print("artisan PLAN IS READY")
            user.tokens += 4000  # Add 1000 to the current token value
            user.save()

        # elif meta_plan is "artisan" and duration_type is "year" :
        #     add_tokens = 4000     

        Subscription.objects.update_or_create(
            user=user,
            defaults={
                'plan': plan,
                'stripe_customer_id': customer_id,
                'stripe_subscription_id': stripe_subscription_id,
                'status': status_s,
                'active': active_flag,
                'subscription_start': current_period_start_date,
                'subscription_end': current_period_end_date,
                'cancel_at_period_end': sub_obj.get('cancel_at_period_end', False),
                'duration_type': duration_type,            # ✅ now sourced correctly
                'payment_method': pm_id,                   # ✅ fix typo from 'payement_method'
            }
        )


@csrf_exempt
def stripe_webhook(request):
    payload = request.body
    sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
    endpoint_secret = settings.STRIPE_WEBHOOK_SECRET

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except ValueError:
        return JsonResponse({'error': 'Invalid payload'}, status=400)
    except stripe.error.SignatureVerificationError:
        return JsonResponse({'error': 'Invalid signature'}, status=400)

    event_type = event.get('type')
    data = event['data']['object']

    # -----------------------------
    # 1) Checkout Completed (both subscription and one-time)
    # -----------------------------
    if event_type == 'checkout.session.completed':
        if data.get('mode') == 'subscription':
            # Subscription flow
            sub_id = data.get('subscription')
            if sub_id:
                sub = stripe.Subscription.retrieve(sub_id)
                # current_period_start = sub.get('current_period_start')
                # current_period_end = sub.get('current_period_end')
                # print(f"current_period_end : {current_period_end}")
                # print(f"current_period_start : {current_period_start}")
                _sync_subscription(sub)
            return JsonResponse({'ok': True})

        elif data.get('mode') == 'payment':
            # One-time token purchase flow
            session_id = data.get('id')
            user_id = data.get('metadata', {}).get('user_id')
            tokens_price_id = data.get('metadata', {}).get('tokens_price_id')

            if session_id and user_id and tokens_price_id:
                try:
                    # Get purchased line items (with quantity)
                    line_items = stripe.checkout.Session.list_line_items(session_id)
                    quantity = line_items.data[0].quantity
                    price_id = line_items.data[0].price.id

                    # Find user and token package
                    user = CustomUser.objects.get(id=user_id)
                    token_pack = TokensPrice.objects.get(stripe_price_id=price_id)

                    tokens_to_add = token_pack.quantity_of_tokens * quantity
                    user.tokens = (user.tokens or 0) + tokens_to_add
                    user.save()

                    logger.info(f"✅ Credited {tokens_to_add} tokens to {user.email}")
                except Exception as e:
                    logger.error(f"❌ Failed to credit tokens: {str(e)}")

            return JsonResponse({'ok': True})

    # -----------------------------
    # 2) Renewal Paid -> Extend Subscription and Allocate Tokens
    # -----------------------------
    if event_type == 'invoice.payment_succeeded':
        invoice = data
        sub_id = invoice.get('subscription')
        if sub_id:
            sub = stripe.Subscription.retrieve(sub_id)
            _sync_subscription(sub)
        return JsonResponse({'ok': True})

    # -----------------------------
    # 3) Subscription Updates
    # -----------------------------
    if event_type == 'customer.subscription.updated':
        sub = data
        _sync_subscription(sub)
        return JsonResponse({'ok': True})

    # -----------------------------
    # 4) Subscription Canceled
    # -----------------------------
    if event_type == 'customer.subscription.deleted':
        sub = data
        _sync_subscription(sub)
        customer_id = sub.get('customer')
        Subscription.objects.filter(stripe_customer_id=customer_id).update(active=False, status='canceled')
        return JsonResponse({'ok': True})

    # -----------------------------
    # 5) Payment Failed
    # -----------------------------
    if event_type == 'invoice.payment_failed':
        invoice = data
        sub_id = invoice.get('subscription')
        if sub_id:
            sub = stripe.Subscription.retrieve(sub_id)
            _sync_subscription(sub)
        return JsonResponse({'ok': True})

    return JsonResponse({'status': 'ignored'})


@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def cancel_subscription(request):
    """
    Body: {"at_period_end": true}  # default True
    """
    at_period_end = bool(request.data.get('at_period_end', True))

    sub = Subscription.objects.filter(user=request.user, active=True).first()
    if not sub or not sub.stripe_subscription_id:
        return Response({'error': 'No active subscription'}, status=400)

    try:
        if at_period_end:
            stripe.Subscription.modify(
                sub.stripe_subscription_id,
                cancel_at_period_end=True
            )
            Subscription.objects.update_or_create(
            user=request.user,
            defaults={
                'active': False,
                'status': 'canceled',
                'plan':"free",
                'subscription_end': sub.subscription_end,  # remains until period end
                'cancel_at_period_end': True,
            })
        else:
            stripe.Subscription.delete(sub.stripe_subscription_id)  # immediate cancel
        return Response({'status': 'cancellation_requested'})
    except stripe.error.StripeError as e:
        return Response({'error': e.user_message or str(e)}, status=500)


def update_tokens_based_on_plan(user, plan):
    """
    Updates the user's tokens based on the subscription plan.
    """
    if plan == 'maker':
        user.tokens += 1000  # Add 250 tokens for the 'maker' plan
    elif plan == 'artisan':
        user.tokens += 4000  # Unlimited tokens for the 'artisan' plan
    else:
        user.tokens = 0  # Default: no tokens if the plan is 'free' or undefined

    user.save()
    logger.info(f"✅ Tokens updated for {user.email} to {user.tokens} tokens based on plan: {plan}")



# @api_view(['GET'])
# def validate_session(request, session_id):
#     try:
#         # Fetch the session from Stripe
#         session = stripe.checkout.Session.retrieve(session_id)
#         logger.info(f"Stripe session retrieved: {session.id}")  # Log session ID for better traceability

#         customer_email = session.get('customer_email')
#         if not customer_email:
#             logger.error("Customer email not found in session.")
#             return Response({'error': 'Customer email not found in session'}, status=status.HTTP_400_BAD_REQUEST)
        
#         logger.info(f"Customer email: {customer_email}")

#         # Find user by email
#         try:
#             user = CustomUser.objects.get(email=customer_email)
#             logger.info(f"Found user: {user.email}")
#         except CustomUser.DoesNotExist:
#             logger.error(f"User with email {customer_email} not found.")
#             return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

#         # Retrieve subscription plan metadata from Stripe session
#         subscription_plan = session.get('metadata', {}).get('plan')
#         if not subscription_plan:
#             logger.error("Subscription plan not found in session metadata.")
#             return Response({'error': 'Subscription plan not found in session metadata'}, status=status.HTTP_400_BAD_REQUEST)
        
#         logger.info(f"Subscription plan: {subscription_plan}")

#         # Update user's tokens based on the subscription plan
#         if subscription_plan == 'maker':
#             user.tokens = 1000  # Add 500 tokens for 'maker' plan
#             logger.info(f"Updated tokens for 'maker' plan: {user.tokens}")
#         elif subscription_plan == 'artisan':
#             user.tokens = 4000  # Represent 'unlimited' tokens for 'artisan' plan
#             logger.info(f"Updated tokens for 'artisan' plan: {user.tokens}")
#         else:
#             logger.error(f"Unknown subscription plan: {subscription_plan}")
#             return Response({'error': 'Unknown subscription plan'}, status=status.HTTP_400_BAD_REQUEST)

#         # Update subscription details
#         subscription, created = Subscription.objects.update_or_create(
#             user=user,  # Ensure we update the existing subscription
#             defaults={
#                 'plan': subscription_plan,
#                 'active': True,
#                 'stripe_customer_id': session.get('customer'),
#                 'stripe_subscription_id': session.get('subscription'),
#                 'subscription_start': timezone.now(),
#                 'subscription_end': timezone.now() + timezone.timedelta(days=30),  # Adjust as needed
#             }
#         )
#         logger.info(f"Subscription for {user.email} updated (Plan: {subscription_plan}, Active: {subscription.active})")

#         # Save user with updated tokens
#         user.save()
#         logger.info(f"User {user.email} tokens updated successfully to {user.tokens}")

#         # Return success response
#         return Response({'status': 'success'}, status=status.HTTP_200_OK)

#     except stripe.error.InvalidRequestError as e:
#         logger.error(f"Stripe error: {e}")
#         return Response({'error': 'Invalid session ID'}, status=status.HTTP_400_BAD_REQUEST)
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe API error: {e}")
#         return Response({'error': 'Stripe API error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#     except Exception as e:
#         logger.exception(f"Unexpected error: {e}")
#         return Response({'error': 'Server error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def validate_session(request, session_id):
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        sub_id = session.get('subscription')
        if not sub_id:
            return Response({'error': 'No subscription on session'}, status=400)
        sub = stripe.Subscription.retrieve(sub_id)

        return Response({'status': 'success'})
    except stripe.error.InvalidRequestError:
        return Response({'error': 'Invalid session ID'}, status=400)
    except stripe.error.StripeError:
        return Response({'error': 'Stripe API error'}, status=500)
    except Exception as e:
        return Response({'error': str(e)}, status=500)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def validate_session_for_token(request, session_id):
    try:
        # Retrieve the Checkout session
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Get purchased line items (includes quantity)
        line_items = stripe.checkout.Session.list_line_items(session.id, limit=100)
        
        # Assume a single product purchase for tokens
        quantity = line_items.data[0].quantity

        # Get user
        try:
            user = CustomUser.objects.get(id=request.user.id)
        except CustomUser.DoesNotExist:
            return Response({'error': 'User not found'}, status=404)

     

        return Response({
            'status': 'success',
            'new_balance': user.tokens,
            'quantity_purchased': quantity
        })

    except stripe.error.InvalidRequestError:
        return Response({'error': 'Invalid session ID'}, status=400)
    except stripe.error.StripeError:
        return Response({'error': 'Stripe API error'}, status=500)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

from django.urls import path
from .views import current_user_view,validate_session,stripe_webhook,SignupView,RefreshView,LoginView,VerifyOTPView,ForgotPasswordView,ResendResetOTPView, VerifyResetOTPView, ResetPasswordView,CreditTokensView,LogoutView,create_checkout_session

urlpatterns = [
    path('signup/', SignupView.as_view(), name='signup'),
    path('verify-otp/', VerifyOTPView.as_view(), name='verify-otp'),
    path('forgot-password/', ForgotPasswordView.as_view(), name='forgot-password'),
    path('verify-reset-otp/', VerifyResetOTPView.as_view(), name='verify-reset-otp'),
    path('reset-password/', ResetPasswordView.as_view(), name='reset-password'),
    path('resend-reset-otp/', ResendResetOTPView.as_view(), name='resend-reset-otp'),
    path('login/', LoginView.as_view(), name='login'),
    path("me/", current_user_view, name="account-me"),
    path('refresh/', RefreshView.as_view(), name='token_refresh'),
    path('credits/',CreditTokensView.as_view(),name='user_credits'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('stripe/create-checkout-session/', create_checkout_session),
    path('stripe/webhook/', stripe_webhook, name='stripe-webhook'),
    path("stripe/validate-session/<str:session_id>/",validate_session, name="validate-session"),


    # path('create-model-checkout/<int:model_id>/', BuyModelCheckoutSession.as_view(), name='buy-model'),
    # path('webhooks/stripe/', StripeWebhookView.as_view(), name='stripe-webhook')
]

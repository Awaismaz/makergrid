from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    model = CustomUser

    list_display = (
        'username',
        'email',
        'full_name',
        'get_subscription_type',
        'is_subscription_active',
        'models_generated',
        'is_email_verified',
        'is_staff',
        'last_login',
    )

    list_filter = (
        'is_staff',
        'is_superuser',
        'is_active',
        'is_email_verified',
    )

    search_fields = ('username', 'email', 'full_name', 'organization')
    ordering = ('-date_joined',)

    readonly_fields = (
        'last_login',
        'get_subscription_type',
        'get_subscription_start',
        'get_subscription_end',
        'models_generated',
    )

    fieldsets = (
        (None, {'fields': ('username', 'email', 'password')}),
        ('Personal Info', {'fields': ('full_name', 'organization', 'profile_picture')}),
        ('Subscription', {
            'fields': (
                'get_subscription_type',
                'get_subscription_start',
                'get_subscription_end',
                'models_generated',
            )
        }),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important Dates', {'fields': ('last_login', 'date_joined')}),
        ('Verification', {'fields': ('is_email_verified',)}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2'),
        }),
    )

    # Safe proxy accessors for related Subscription model
    def get_subscription_type(self, obj):
        return getattr(obj.subscription, 'plan', 'free')
    get_subscription_type.short_description = 'Subscription Type'

    def get_subscription_start(self, obj):
        return getattr(obj.subscription, 'subscription_start', None)
    get_subscription_start.short_description = 'Start Date'

    def get_subscription_end(self, obj):
        return getattr(obj.subscription, 'subscription_end', None)
    get_subscription_end.short_description = 'End Date'

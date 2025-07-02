from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse

urlpatterns = [
    path("admin/", admin.site.urls),
    path('api/accounts/', include('accounts.urls')),
    path('api/community/', include('community.urls')),
    path('api/makers/',include('makers.urls')),
    path('api-auth/', include('rest_framework.urls')),
    path("test/", lambda r: JsonResponse({"status": "ok"}))
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


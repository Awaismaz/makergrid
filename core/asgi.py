import os
import django
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
from django.conf import settings
from starlette.staticfiles import StaticFiles
from starlette.responses import Response, FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

import community.routing  # ✅ safe after setup

# ✅ Ensure media folder exists
os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

MEDIA_DIR = Path(settings.MEDIA_ROOT).resolve()
print(f"📁 Using MEDIA_DIR: {MEDIA_DIR}")

media_app = StaticFiles(directory=str(MEDIA_DIR))

class MediaMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")

        if scope["type"] == "http" and path.startswith("/media/"):
            requested_file = path.replace("/media/", "")
            absolute_path = MEDIA_DIR / requested_file

            print(f"🔍 Requesting: {path}")
            print(f"📂 Resolved full path: {absolute_path}")
            print(f"📘 File exists: {absolute_path.exists()}")

            try:
                await media_app(scope, receive, send)
            except StarletteHTTPException as e:
                if e.status_code == 404:
                    response = Response("Media file not found", status_code=404)
                    await response(scope, receive, send)
                else:
                    raise
        else:
            await self.app(scope, receive, send)

# ✅ Wrap Django app
django_asgi_app = MediaMiddleware(get_asgi_application())

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(community.routing.websocket_urlpatterns)
    ),
})

from django.http import FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.clickjacking import xframe_options_exempt
import os
from django.conf import settings

@csrf_exempt
@xframe_options_exempt
def serve_media(request, path):
    full_path = os.path.join(settings.MEDIA_ROOT, path)
    print("Serving media via Django view:", path)
    if not os.path.exists(full_path):
        raise Http404("File not found.")
    return FileResponse(open(full_path, 'rb'))

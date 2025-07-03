
import os
import uuid
import requests
import time
from dotenv import load_dotenv
from django.conf import settings
from rest_framework import status, generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Asset
from .serializers import AssetSerializer
from core.authentication.authentication import JWTAuthentication
from .pagination import CustomPageNumberPagination
import replicate
from openai import OpenAI
import traceback
import httpx
import time
from celery.result import AsyncResult
from .tasks import generate_3d_model,generate_model_from_image
# from .tasks import generate_3d_model
load_dotenv(override=True)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRELLIS_KEY = os.getenv("TRELLIS_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

def save_uploaded_file_to_media(uploaded_file, filename):
    path = default_storage.save(f"uploads/{filename}", ContentFile(uploaded_file.read()))
    return settings.MEDIA_URL + path

def download_and_save_to_media(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        path = default_storage.save(f"models/{filename}", ContentFile(response.content))
        return settings.MEDIA_URL + path
    raise Exception("Failed to download file from external source.")

def normalize_text(text):
    return text.lower().replace("-", " ").replace("_", " ").strip()



class TextTo3DModelView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        print("Job started")
        start = time.time()
        
        # Extract the necessary information from the request
        user_prompt = request.data.get("prompt")
        style = request.data.get("style")
        complexity = request.data.get("complexity")
        optimize_printing = request.data.get("optimize_printing")

        # Validate that the required fields are provided
        if not user_prompt or not style or not complexity:
            return Response({"error": "Prompt, style, and complexity are required."}, status=400)
        
        # Get the user ID from the authenticated request user
        user_id = request.user.id
        
        # Call the Celery task with the relevant arguments
        task = generate_3d_model.apply_async(
            args=[user_id, user_prompt, style, complexity, optimize_printing]
        )

        print(f"Task {task.id} started for user {user_id}")

        # Return the task ID and status to the frontend
        return Response({
            "task_id": task.id,
            "status": "pending"
        })

class ImageTo3DModelView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        start = time.time()
        try:
            uploaded_file = request.FILES.get("image")
            if not uploaded_file:
                return Response({"error": "No image uploaded"}, status=400)

            filename = f"temp_{uuid.uuid4()}.png"
            image_url = save_uploaded_file_to_media(uploaded_file, filename)

            # Trigger Celery task asynchronously
            task = generate_model_from_image.apply_async(args=[request.user.id, image_url])

            print(f"Task {task.id} started for user {request.user.id}")

            # Return the task ID for status tracking
            return Response({
                "task_id": task.id,
                "status": "pending"
            })
        
        except Exception as e:
            print("🔥 Exception in ImageTo3DModelView:")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=500)


class CheckTaskStatusView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, task_id):
        task_result = AsyncResult(task_id)

        if task_result.state == 'PENDING':
            response_data = {
                'status': 'pending',
                'result': 'Task is still being processed.'
            }
        elif task_result.state == 'SUCCESS':
            response_data = {
                'status': 'completed',
                'result': task_result.result  # You can customize this based on the task result structure
            }
        elif task_result.state == 'FAILURE':
            response_data = {
                'status': 'failed',
                'result': str(task_result.info)  # The error message from the exception if the task failed
            }
        else:
            response_data = {
                'status': 'unknown',
                'result': 'Unknown state.'
            }
        
        print(f"Task {task_id} status checked: {response_data['status']}")

        return Response(response_data)
    
class AssetListCreateView(generics.ListCreateAPIView):
    serializer_class = AssetSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Asset.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class AssetRetrieveView(generics.RetrieveAPIView):
    queryset = Asset.objects.all()
    serializer_class = AssetSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]


class UserAssetsView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        assets = Asset.objects.filter(user=user).order_by('-created_at')

        paginator = CustomPageNumberPagination()
        paginated = paginator.paginate_queryset(assets, request)
        serializer = AssetSerializer(paginated, many=True)

        has_next_page = paginator.page.has_next() if paginator.page else False

        return Response({
            "items": serializer.data,
            "hasNextPage": has_next_page
        })

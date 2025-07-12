
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
from accounts.models import CustomUser
from .serializers import AssetSerializer
from core.authentication.authentication import JWTAuthentication
from .pagination import CustomPageNumberPagination
import replicate
from openai import OpenAI
import traceback
import httpx
import time
import logging
from celery.result import AsyncResult
from .tasks import generate_3d_model,generate_model_from_image
# from .tasks import generate_3d_model
load_dotenv(override=True)
logger = logging.getLogger(__name__)


REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRELLIS_KEY = os.getenv("TRELLIS_KEY")
BASEURL= os.getenv("BASEURL", "http://localhost:8000")

OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)



from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

def save_uploaded_file_to_media(uploaded_file, filename):
    """ Save the uploaded image file to the media directory and return the media URL. """
    try:
        # Save the file to the 'images/' directory in media folder
        path = default_storage.save(f"images/{filename}", uploaded_file)
        # Return the full URL of the file
        return path
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise


def save_image_to_media(uploaded_file_url, filename):
    # Download the image
    response = requests.get(uploaded_file_url)
    
    if response.status_code == 200:
        # Create a file-like object from the response content
        content = ContentFile(response.content)
        
        # Save the file to the media directory
        path = default_storage.save(f"images/{filename}", content)
        
        # Return the full media URL where the file is stored
        return path
    else:
        raise Exception(f"Failed to download the image. HTTP status code: {response.status_code}")


def download_and_save_to_media(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        path = default_storage.save(f"models/{filename}", ContentFile(response.content))
        return path
    raise Exception("Failed to download file from external source.")

def normalize_text(text):
    return text.lower().replace("-", " ").replace("_", " ").strip()



# class TextTo3DModelView(APIView):
#     authentication_classes = [JWTAuthentication]
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request):
#         print("Job started")
#         start = time.time()
        
#         # Extract the necessary information from the request
#         user_prompt = request.data.get("prompt")
#         style = request.data.get("style")
#         complexity = request.data.get("complexity")
#         optimize_printing = request.data.get("optimize_printing")

#         # Validate that the required fields are provided
#         if not user_prompt or not style or not complexity:
#             return Response({"error": "Prompt, style, and complexity are required."}, status=400)
        
#         # Get the user ID from the authenticated request user
#         user_id = request.user.id
        
#         # Call the Celery task with the relevant arguments
#         task = generate_3d_model.apply_async(
#             args=[user_id, user_prompt, style, complexity, optimize_printing]
#         )

#         print(f"Task {task.id} started for user {user_id}")

#         # Return the task ID and status to the frontend
#         return Response({
#             "task_id": task.id,
#             "status": "pending"
#         })





class TextTo3DModelView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        print("Job started")
        user_prompt = request.data.get("prompt")
        style = request.data.get("style")
        complexity = request.data.get("complexity")
        optimize_printing = request.data.get("optimize_printing")

        tokens = request.user.tokens

        print(f"tokens : {tokens}")

        if not user_prompt or not style or not complexity:
            return Response({"error": "Prompt, style, and complexity are required."}, status=400)

        complexity_map = {
            "simple": "dall-e-2",
            "medium": "dall-e-2",
            "complex": "dall-e-3",
            "very complex": "dall-e-3",
        }

        style_descriptions = {
            "realistic": "photorealistic precision with accurate textures and lighting",
            "stylized": "heavily stylized with exaggerated forms and bold colors",
            "low-poly": "minimalist low-poly style for games",
            "sci-fi": "futuristic, metallic, neon-lit design",
            "fantasy": "mythical elements and magical scenery",
            "miniature": "detailed miniatures for tabletop gaming",
            "cartoon": "cartoonish, bold outlines, expressive features"
        }

        model_type = complexity_map.get(normalize_text(complexity), "dall-e-2")
        style_instruction = style_descriptions.get(normalize_text(style), style_descriptions["realistic"])

        final_prompt = (
            f"Generate an image from: {user_prompt}. Style should be {style_instruction}. "
            "Use a pure black background with the subject centered."
        )

        if optimize_printing:
            final_prompt += " Ensure 3D printability with correct thickness and no fragile parts."

        image_size = "1792x1024" if normalize_text(complexity) == "very complex" else "1024x1024"

        try:
            # Assuming you are using a client to generate the image like OpenAI API or another image generation service
            image_response = OpenAIclient.images.generate(
                model=model_type,
                prompt=final_prompt,
                size=image_size,
                n=1,
            )
            image_url = image_response.data[0].url

            replicate_input = {
                "version": "e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c",
                "input": {
                    "images": [image_url],
                    "texture_size": 2048,
                    "mesh_simplify": 0.9,
                    "generate_model": True,
                    "save_gaussian_ply": True,
                    "ss_sampling_steps": 38,
                }
            }

            headers = {
                "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }

            with httpx.Client() as client:
                response = client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers=headers,
                    json=replicate_input,
                    timeout=300.0
                )
            print("job done")
            # Assuming the response contains the model prediction details
            if response.status_code == 201:
                replicate_response = response.json()
                print(f"Replicate response: {replicate_response}")
                
                # Extract the task ID and status
                task_id = replicate_response.get("id")
                status = replicate_response.get("status")
                
                # Return only task_id and status in the response
                return Response({
                    "task_id": task_id,
                    "status": status
                })

            else:
                return Response({"error": "Replicate API request failed."}, status=response.status_code)

        except Exception as e:
            print("🔥 Exception in TextTo3DModelView:")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=500)

class ImageTo3DModelView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        start = time.time()
        print("Job started image to 3D model")
        try:
            uploaded_file = request.FILES.get("image")
            if not uploaded_file:
                return Response({"error": "No image uploaded"}, status=400)

            filename = f"temp_{uuid.uuid4()}.png"
            image_url = save_uploaded_file_to_media(uploaded_file, filename)
            final_image_url = f"{BASEURL}/media/{image_url}"
            # temp_image_path = f"https://94bd-94-156-30-70.ngrok-free.app/media/{image_url}"
            replicate_input = {
                "version": "e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c",
                "input": {
                    "images": [final_image_url],
                    "texture_size": 2048,
                    "mesh_simplify": 0.9,
                    "generate_model": True,
                    "save_gaussian_ply": True,
                    "ss_sampling_steps": 38,
                }
            }

            headers = {
                "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }

            with httpx.Client() as client:
                response = client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers=headers,
                    json=replicate_input,
                    timeout=300.0
                )
            print("job done")
            # Assuming the response contains the model prediction details
            if response.status_code == 201:
                replicate_response = response.json()
                print(f"Replicate response: {replicate_response}")
                
                # Extract the task ID and status
                task_id = replicate_response.get("id")
                status = replicate_response.get("status")
                
                # Return only task_id and status in the response
                return Response({
                    "task_id": task_id,
                    "status": status
                })

            else:
                return Response({"error": "Replicate API request failed."}, status=response.status_code)
        
        except Exception as e:
            print("🔥 Exception in ImageTo3DModelView:")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=500)

class GetPredictionStatusView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, task_id):
        # Now, task_id is expected as part of the URL path (not the request body)
        if not task_id:
            return Response({"error": "task_id is required."}, status=400)
        
        user_prompt = request.data.get("prompt")
        style = request.data.get("style")
        complexity = request.data.get("complexity")
        optimize_printing = request.data.get("optimize_printing")

        try:
            # Prepare the headers for the request to Replicate API
            headers = {
                "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }

            # Construct the prediction URL using the task_id
            prediction_url = f"https://api.replicate.com/v1/predictions/{task_id}"

            # Initialize an HTTP client
            with httpx.Client() as client:
                # Fetch the current status of the prediction
                response = client.get(prediction_url, headers=headers)

                if response.status_code == 200:
                    prediction = response.json()
                    status = prediction.get("status")

                    # Perform additional operations (logging, processing, etc.)
                    print(f"Task {task_id} status: {status}")

                    # If the prediction succeeded, extract the URLs and return them
                    if status == "succeeded":
                        print(f"Prediction succeeded for task {task_id}")

                        # Extract URLs from the prediction result
                        output_data = prediction.get("output", {})
                        model_file = output_data.get("model_file")
                        color_video = output_data.get("color_video")
                        gaussian_ply = output_data.get("gaussian_ply")

                        print(f"Model file URL: {model_file}")
                        print(f"Color video URL: {color_video}")

                        if not model_file:
                            logger.error("Model file not found in output")
                            raise ValueError("Model file not found in output")

                        logger.info(f"Model file generated: {model_file}")

                        glb_filename = f"{uuid.uuid4()}.glb"
                        model_path = download_and_save_to_media(model_file, glb_filename)
                        logger.info(f"Model saved to media at: {model_path}")

                        input_data = prediction.get("input", {})
                        image_url = input_data.get("images", [])[0]

                        image_filename = f"{uuid.uuid4()}.jpg"
                        image_path = save_image_to_media(image_url,image_filename)
                        tokens = request.user.tokens

                        print(f"tokens : {tokens}")

                        update_tokens = CustomUser.objects.filter(id=request.user.id).update(tokens=tokens-20)

                        asset = Asset.objects.create(
                            user=request.user,
                            prompt=user_prompt,
                            model_file=model_path,
                            preview_image_url=image_path,
                            style=style,
                            complexity=complexity,
                            optimize_printing=optimize_printing
                        )
                        print(f"model path: {model_path}")
                        print(f'image path: {image_path}')
                        return Response({
                            "status": "completed",
                            "message": "3D model created.",
                            "asset_id": str(asset.id),
                            "model_file":  f"{model_file}",
                            "gaussian_ply": gaussian_ply,
                            "color_video": color_video,
                            "stored_path": model_path,
                            "preview_image_url": image_path,
                            "created_at": asset.created_at,
                        })

                    # If the prediction failed
                    elif status == "failed":
                        return Response({"error": "Prediction failed."}, status=400)

                    # If the prediction is still processing or starting, return the status
                    return Response({"status": status})

                else:
                    return Response({"error": "Failed to get prediction status."}, status=response.status_code)

        except Exception as e:
            print("🔥 Exception in GetPredictionStatusView:")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=500)

class GetImagePredictionStatusView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, task_id):
        if not task_id:
            return Response({"error": "task_id is required."}, status=400)

        retries = 5  # Number of retries for "starting" status
        try:
            headers = {
                "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }

            prediction_url = f"https://api.replicate.com/v1/predictions/{task_id}"

            with httpx.Client() as client:
                # Retry fetching the task status in case it's still starting
                for attempt in range(retries):
                    response = client.get(prediction_url, headers=headers)

                    if response.status_code == 200:
                        prediction = response.json()
                        status = prediction.get("status")

                        # Log the status
                        print(f"Task {task_id} status: {status}")

                        if status == "succeeded":
                            # Process the successful task
                            output_data = prediction.get("output", {})
                            model_file = output_data.get("model_file")
                            color_video = output_data.get("color_video")
                            gaussian_ply = output_data.get("gaussian_ply")

                            logger.info(f"Model file generated: {model_file}")

                            # Save model and return data
                            model_path = download_and_save_to_media(model_file, f"{uuid.uuid4()}.glb")
                            image_path = save_image_to_media(prediction.get("input", {}).get("images", [])[0], f"{uuid.uuid4()}.jpg")
                            tokens = request.user.tokens
                            update_tokens = CustomUser.objects.filter(id=request.user.id).update(tokens=tokens-20)
                            # Asset creation and response
                            asset = Asset.objects.create(
                                user=request.user,
                                prompt="Created from image",
                                model_file=model_path,
                                preview_image_url=image_path,
                                style="created from image",
                                complexity="created from image",
                                optimize_printing=False
                            )

                            return Response({
                                "status": "completed",
                                "message": "3D model created.",
                                "asset_id": str(asset.id),
                                "model_file": f"{model_file}",
                                "gaussian_ply": gaussian_ply,
                                "color_video": color_video,
                                "stored_path": model_path,
                                "preview_image_url": image_path,
                                "created_at": asset.created_at,
                            })

                        elif status == "failed":
                            # Log the failure and provide an error response
                            logger.error(f"Task {task_id} failed: {prediction.get('error_message', 'Unknown error')}")
                            return Response({"error": "Prediction failed."}, status=400)

                        # If the task is still starting, continue retrying
                        elif status == "starting" or status == "processing":
                            # logger.info(f"Task {task_id} still processing... Attempt {attempt + 1}/{retries}")
                            # if attempt == retries - 1:
                            #     return Response({"error": f"Task {task_id} is still processing after {retries} attempts."}, status=400)
                            # else:
                            #     time.sleep(5)  # Wait before retrying
                            #     continue
                            return Response({"status": status})
                    else:
                        # Log API failure
                        logger.error(f"Failed to get prediction status for task {task_id}. Status code: {response.status_code}")
                        return Response({"error": f"Failed to get prediction status. Status code: {response.status_code}"}, status=response.status_code)

        except Exception as e:
            logger.error(f"🔥 Exception in GetImagePredictionStatusView for task {task_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({"error": str(e)}, status=500)


    
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
        print(f"serializer.data: {serializer.data}")
        return Response({
            "items": serializer.data,
            "hasNextPage": has_next_page
        })

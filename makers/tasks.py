from celery import shared_task
from django.core.files.base import ContentFile
from django.conf import settings
from .models import Asset
from dotenv import load_dotenv
import time
import uuid
import os
import httpx
# from .utils import download_and_save_to_media, normalize_text
# from external_client import client, replicate  # Assuming this is an external service you're using
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import replicate
import requests
from openai import OpenAI
from accounts.models import CustomUser  # Import your custom user model
import logging

from makers.lib.functions import download_and_save_to_media, normalize_text

logger = logging.getLogger(__name__)
load_dotenv(override=True)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRELLIS_KEY = os.getenv("TRELLIS_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)

@shared_task(bind=True)
def generate_3d_model(self, user_id, user_prompt, style, complexity, optimize_printing):
    logger.info(f"Task started with user_id: {user_id}, prompt: {user_prompt}, style: {style}, complexity: {complexity}")

    try:
        # Log each function separately to know where it fails

        print(f"Received values: {user_id}, {user_prompt}, {style}, {complexity}, {optimize_printing}")
        
        # Check that all arguments are received correctly
        if not all([user_id, user_prompt, style, complexity, optimize_printing]):
            raise ValueError("Not all arguments are passed correctly.")
        
        logger.info("Fetching user data")
        try:
            user = CustomUser.objects.get(id=user_id)  # Reference your custom user model
            logger.info(f"User {user.username} fetched successfully.")
        except CustomUser.DoesNotExist:
            logger.error(f"User with ID {user_id} does not exist.")
            raise ValueError(f"User with ID {user_id} does not exist")

        logger.info("Processing complexity and style")
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

        model_type = complexity_map.get(complexity, "dall-e-2")
        style_instruction = style_descriptions.get(style, style_descriptions["realistic"])

        final_prompt = f"Generate an image from: {user_prompt}. Style should be {style_instruction}. Use a pure black background with the subject centered."
        if optimize_printing:
            final_prompt += " Ensure 3D printability with correct thickness and no fragile parts."

        logger.info(f"Final prompt: {final_prompt}")

        # Log the image generation request
        image_size = "1792x1024" if complexity == "very complex" else "1024x1024"
        image_response = client.images.generate(model=model_type, prompt=final_prompt, size=image_size, n=1)
        image_url = image_response.data[0].url
        logger.info(f"Image generated: {image_url}")

        replicate_input = {
            "images": [image_url],
            "texture_size": 2048,
            "mesh_simplify": 0.9,
            "generate_model": True,
            "save_gaussian_ply": True,
            "ss_sampling_steps": 38,
        }

        timeout = httpx.Timeout(300)
        output = replicate.run(settings.TRELLIS_KEY, input=replicate_input, timeout=timeout)

        model_file = output.get("model_file") and output["model_file"].url
        color_video = output.get("color_video") and output["color_video"].url
        gaussian_ply = output.get("gaussian_ply") and output["gaussian_ply"].url

        if not model_file:
            logger.error("Model file not found in output")
            raise ValueError("Model file not found in output")

        logger.info(f"Model file generated: {model_file}")

        glb_filename = f"{uuid.uuid4()}.glb"
        s3_url = download_and_save_to_media(model_file, glb_filename)
        logger.info(f"Model saved to media at: {s3_url}")

        asset = Asset.objects.create(
            user=user,
            prompt=user_prompt,
            model_file=s3_url,
            preview_image_url=image_url,
            style=style,
            complexity=complexity,
            optimize_printing=optimize_printing
        )
        logger.info(f"Asset created with ID: {asset.id}")

        response_data = {
            "status": "completed",
            "message": "3D model created.",
            "asset_id": str(asset.id),
            "model_file": model_file,
            "gaussian_ply": gaussian_ply,
            "color_video": color_video,
            "stored_path": s3_url,
            "preview_image_url": image_url,
            "created_at": asset.created_at,
        }

        return response_data

    except Exception as e:
        logger.exception("Error during 3D model generation")
        raise self.retry(exc=e)





@shared_task(bind=True)
def generate_model_from_image(self, user_id, image_url):
    try:
        replicate_input = {
            "images": [image_url],
            "texture_size": 2048,
            "mesh_simplify": 0.9,
            "generate_model": True,
            "save_gaussian_ply": True,
            "ss_sampling_steps": 38,
        }

        timeout = httpx.Timeout(300)
        output = replicate.run('YOUR_TRELLIS_KEY', input=replicate_input, timeout=300)

        model_file = output.get("model_file") and output["model_file"].url
        color_video = output.get("color_video") and output["color_video"].url
        gaussian_ply = output.get("gaussian_ply") and output["gaussian_ply"].url

        if not model_file:
            raise ValueError("model_file not found in output")

        # Upload model to S3
        glb_filename = f"{uuid.uuid4()}.glb"
        s3_url = download_and_save_to_media(model_file, glb_filename)

        # Create Asset in DB
        asset = Asset.objects.create(
            user_id=user_id,
            model_file=s3_url,
            preview_image_url=image_url,
        )

        return {
            "status": "completed",
            "asset_id": asset.id,
            "stored_path": s3_url,
            "model_file": model_file,
            "color_video": color_video,
            "gaussian_ply": gaussian_ply,
            "preview_image_url": image_url,
            "created_at": str(asset.created_at),
        }
    except Exception as e:
        raise self.retry(exc=e)

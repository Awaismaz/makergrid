from django.db import models
from django.conf import settings

class Asset(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="assets",
        null=True,
        blank=True
    )
    prompt = models.TextField()
    model_file = models.FileField(upload_to="assets/")
    preview_image_url = models.URLField(blank=True, null=True)
    style = models.TextField(default="realistic")  # or whatever default you prefer
    complexity = models.TextField(default="medium")  # sensible default
    optimize_printing = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Asset ({self.id}) - {self.user.username if self.user else 'anon'}"

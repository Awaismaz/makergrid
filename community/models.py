from django.db import models
from django.conf import settings


class Conversation(models.Model):
    participants = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='conversations')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        usernames = ", ".join(p.username for p in self.participants.all()[:2])
        return f"Conversation between {usernames}"

    def is_private(self):
        return self.participants.count() == 2




class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    sender_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)  # for relational filtering
    sender = models.CharField(max_length=150)  # store username as string
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    seen = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.sender}: {self.content[:20]}"



class Post(models.Model):
    VISIBILITY_CHOICES = [
        ('public', 'Public'),
        ('private', 'Private'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='posts')
    title = models.CharField(max_length=255, default='Untitled')
    caption = models.CharField(max_length=255, blank=True)
    prompts = models.TextField(help_text="Prompts used to generate the model (can be JSON, text, etc.)")
    model_file = models.FileField(upload_to='models/')
    image = models.ImageField(upload_to='model_images/', blank=True, null=True)  # ✅ New field
    preview_image = models.ImageField(upload_to='model_previews/', blank=True, null=True)
    visibility = models.CharField(max_length=7, choices=VISIBILITY_CHOICES, default='public')
    
    # ✅ Engagement metrics
    likes_count = models.PositiveIntegerField(default=0)
    views_count = models.PositiveIntegerField(default=0)
    hearts = models.PositiveIntegerField(default=0)
    lol = models.PositiveIntegerField(default=0)
    tips = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} by {self.user.username}"



class Comment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="comments")
    post = models.ForeignKey('Post', on_delete=models.CASCADE, related_name="comments")
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name="replies")
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Comment by {self.user.username} on Post {self.post.id}"

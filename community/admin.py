from django.contrib import admin
from .models import Post, Conversation, Message,Comment

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "caption", "visibility", "likes_count", "views_count", "created_at")
    search_fields = ("caption", "prompts", "user__username")
    list_filter = ("visibility", "created_at")


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "post", "parent", "created_at")
    search_fields = ("user__username", "content", "post__caption")


# Keep existing ones if not already present
admin.site.register(Conversation)
admin.site.register(Message)

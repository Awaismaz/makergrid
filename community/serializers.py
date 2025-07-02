from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Conversation, Message, Post, Comment

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'sender', 'sender_id', 'content', 'timestamp', 'seen']
        read_only_fields = ['sender', 'sender_id', 'timestamp']


class ConversationSerializer(serializers.ModelSerializer):
    other_participant = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'other_participant', 'last_message']

    def get_other_participant(self, obj):
        request = self.context.get('request')
        current_user = request.user
        other = obj.participants.exclude(id=current_user.id).first()
        return UserSerializer(other).data if other else None

    def get_last_message(self, obj):
        last = obj.messages.order_by('-timestamp').first()
        return MessageSerializer(last).data if last else None


# 🔹 Post Serializer


class PostSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    user_avatar = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            'id',
            'title',
            'image',
            'preview_image',
            'caption',
            'prompts',
            'model_file',
            'visibility',
            'likes_count',
            'views_count',
            'hearts',
            'lol',
            'tips',
            'created_at',
            'username',
            'user_avatar',
        ]
        read_only_fields = [
            'likes_count',
            'views_count',
            'hearts',
            'lol',
            'tips',
            'created_at',
            'username',
            'user_avatar',
        ]

    def get_user_avatar(self, obj):
        # Update based on your user model's avatar field
        return obj.user.avatar.url if hasattr(obj.user, 'avatar') and obj.user.avatar else "/avatar.jpg"

# 🔹 Comment Serializer
class CommentSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    replies = serializers.SerializerMethodField()

    class Meta:
        model = Comment
        fields = ['id', 'user', 'post', 'content', 'parent', 'created_at', 'replies']
        read_only_fields = ['user', 'created_at', 'replies']

    def get_replies(self, obj):
        if obj.parent is None:
            return CommentSerializer(obj.replies.all(), many=True).data
        return []


# serializers.py

class PostPreviewSerializer(serializers.ModelSerializer):
    likes = serializers.IntegerField(source='likes_count', read_only=True)

    class Meta:
        model = Post
        fields = [
            'id',
            'title',
            'image',
            'preview_image',
            'likes',
            'hearts',
            'lol',
            'tips',
            'created_at',
        ]

from rest_framework import generics, permissions, status
from django.contrib.auth import get_user_model
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.exceptions import ValidationError
from django.db.models import Q, Sum, Count
from django.utils.dateformat import format
import hashlib
from rest_framework.permissions import AllowAny
from .models import Conversation, Message, Post, Comment
from .serializers import ConversationSerializer, PostPreviewSerializer, MessageSerializer, PostSerializer, CommentSerializer
from core.authentication.authentication import JWTAuthentication
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
User = get_user_model()

from django.db.models import Count

User = get_user_model()

class PrivateConversationCreateView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user = request.user
        username = request.data.get("username")
        print(f"req : {request.data}")

        if not username:
            return Response({"error": "username is required."}, status=400)

        try:
            recipient = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({"error": "Recipient does not exist."}, status=404)

        if recipient == user:
            return Response({"error": "Cannot start a conversation with yourself."}, status=400)

        # Check if a private conversation already exists between these two users
        conversations = (
            user.conversations.annotate(num_participants=Count('participants'))
            .filter(num_participants=2, participants=recipient)
        )

        if conversations.exists():
            conversation = conversations.first()
        else:
            # Create new conversation
            conversation = Conversation.objects.create()
            conversation.participants.set([user, recipient])
            conversation.save()
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                f"user_{recipient.id}",  # group name specific to that user
                {
                    "type": "new_conversation_event",
                    "conversation_id": conversation.id,
                    "username": user.username,  # optional, to show who started it
                }
            )

        serializer = ConversationSerializer(conversation, context={'request': request})
        return Response(serializer.data)

# 🔹 Conversations
class ConversationListCreateView(generics.ListCreateAPIView):
    serializer_class = ConversationSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(participants=self.request.user)

    def perform_create(self, serializer):
        participants = self.request.data.get('participants', [])
        username = self.request.data.get('username')

        if username:
            try:
                user_obj = User.objects.get(username=username)
            except User.DoesNotExist:
                raise ValidationError("User does not exist.")
            participants = [self.request.user.id, user_obj.id]
        else:
            if not participants:
                raise ValidationError("Recipient(s) required.")
            if self.request.user.id not in participants:
                participants.append(self.request.user.id)

        participants = list(set(participants))
        sorted_ids = sorted(participants)
        hash_input = "-".join(map(str, sorted_ids))
        participant_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        existing = Conversation.objects.filter(participant_hash=participant_hash).first()
        if existing:
            self.instance = existing
            return

        conversation = serializer.save()
        conversation.participants.set(participants)
        conversation.participant_hash = participant_hash
        conversation.save(update_fields=["participant_hash"])


class MessageHistoryView(generics.ListAPIView):
    serializer_class = MessageSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        conversation_id = self.kwargs['conversation_id']
        return Message.objects.filter(
            Q(conversation_id=conversation_id),
            Q(conversation__participants=self.request.user)
        ).order_by('timestamp')


# 🔹 Posts
class PostListCreateView(generics.ListCreateAPIView):
    serializer_class = PostSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def get_queryset(self):
        qs = Post.objects.all()
        if self.request.user.is_authenticated:
            return qs.filter(Q(visibility='public') | Q(user=self.request.user))
        return qs.filter(visibility='public')


class PostDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = PostSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        qs = super().get_queryset()
        if self.request.user.is_authenticated:
            return qs.filter(Q(visibility='public') | Q(user=self.request.user))
        return qs.filter(visibility='public')


class LikeToggleView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, post_id):
        post = Post.objects.get(id=post_id)
        user = request.user
        session_key = f"liked_{post_id}"
        liked = request.session.get(session_key, False)

        if liked:
            post.likes_count = max(post.likes_count - 1, 0)
            request.session[session_key] = False
            action = False
        else:
            post.likes_count += 1
            request.session[session_key] = True
            action = True

        post.save()
        return Response({"liked": action, "likes_count": post.likes_count})


class ViewTrackView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, post_id):
        post = Post.objects.get(id=post_id)
        session_key = f"viewed_{post_id}"

        if not request.session.get(session_key):
            post.views_count += 1
            post.save()
            request.session[session_key] = True

        return Response({"views_count": post.views_count})


# 🔹 Categories
class TrendingPostsView(generics.ListAPIView):
    serializer_class = PostSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        return Post.objects.filter(visibility='public').order_by('-likes_count', '-created_at')[:20]


class MostViewedPostsView(generics.ListAPIView):
    serializer_class = PostSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        return Post.objects.filter(visibility='public').order_by('-views_count')[:20]


class MostLikedPostsView(generics.ListAPIView):
    serializer_class = PostSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        return Post.objects.filter(visibility='public').order_by('-likes_count')[:20]


class CommentListCreateView(generics.ListCreateAPIView):
    serializer_class = CommentSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        post_id = self.kwargs['post_id']
        return Comment.objects.filter(post_id=post_id, parent=None)

    def perform_create(self, serializer):
        post_id = self.kwargs['post_id']
        serializer.save(user=self.request.user, post_id=post_id)


class UserStatsView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, username):
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)

        posts = Post.objects.filter(user=user)

        post_count = posts.count()
        total_likes = posts.aggregate(total=Sum('likes_count'))['total'] or 0
        total_views = posts.aggregate(total=Sum('views_count'))['total'] or 0

        return Response({
            "username": username,
            "joined_date": format(user.date_joined, "M d, Y"),
            "post_count": post_count,
            "total_likes": total_likes,
            "total_views": total_views
        })


class UserPostPreviewListView(generics.ListAPIView):
    serializer_class = PostPreviewSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        username = self.kwargs['username']
        return Post.objects.filter(user__username=username, visibility='public').order_by('-created_at')

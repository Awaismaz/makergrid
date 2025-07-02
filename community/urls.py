from django.urls import path
from .views import PrivateConversationCreateView, ConversationListCreateView, MessageHistoryView,UserPostPreviewListView,  PostListCreateView, PostDetailView, LikeToggleView, ViewTrackView,TrendingPostsView, MostViewedPostsView, MostLikedPostsView,CommentListCreateView,UserStatsView
urlpatterns = [
    path('conversations/', ConversationListCreateView.as_view(), name='conversation-list'),
    path('conversations/<int:conversation_id>/history/', MessageHistoryView.as_view(), name='message-history'),
    path('conversations/private/create/', PrivateConversationCreateView.as_view(), name='create-private-conversation'),
    path('posts/', PostListCreateView.as_view(), name='post-list-create'),
    path('posts/<int:pk>/', PostDetailView.as_view(), name='post-detail'),
    path('posts/<int:post_id>/like/', LikeToggleView.as_view(), name='post-like-toggle'),
    path('posts/<int:post_id>/view/', ViewTrackView.as_view(), name='post-view-track'),
    path('posts/trending/', TrendingPostsView.as_view(), name='trending-posts'),
    path('posts/most-viewed/', MostViewedPostsView.as_view(), name='most-viewed-posts'),
    path('posts/most-liked/', MostLikedPostsView.as_view(), name='most-liked-posts'),
    path('posts/<int:post_id>/comments/', CommentListCreateView.as_view(), name='comment-list-create'),
    path("users/<str:username>/stats/", UserStatsView.as_view(), name="user-stats"),
    path("posts/<str:username>/preview/", UserPostPreviewListView.as_view(), name="user-post-preview"),
]

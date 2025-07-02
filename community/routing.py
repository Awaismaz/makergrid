from django.urls import re_path
from .consumers.chatConsumers import ChatConsumer
from .consumers.globalConsumer import GlobalSocketConsumer

websocket_urlpatterns = [
    re_path(r"ws/chat/(?P<conversation_id>\d+)/$", ChatConsumer.as_asgi()),
    re_path(r"ws/chat/user_(?P<user_id>\d+)/$", GlobalSocketConsumer.as_asgi()),  # 👈 Add this line

]

# consumers.py
import json
from urllib.parse import parse_qs
from channels.generic.websocket import AsyncWebsocketConsumer
class GlobalSocketConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user_id = self.scope['url_route']['kwargs']['user_id']
        self.group_name = f"user_{self.user_id}"

        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        print(f"[GlobalSocket] Connected to {self.group_name}")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)
        print(f"[GlobalSocket] Disconnected from {self.group_name}")

    async def new_conversation_event(self, event):
        await self.send(text_data=json.dumps({
            "type": "new_conversation",
            "conversation_id": event["conversation_id"],
            "username": event["username"],
        }))

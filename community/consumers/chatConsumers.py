import json
from urllib.parse import parse_qs
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework_simplejwt.exceptions import TokenError
from ..models import Conversation, Message
from django.contrib.auth import get_user_model
User = get_user_model()
class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.conversation_id = self.scope['url_route']['kwargs']['conversation_id']
        self.room_group_name = f'chat_{self.conversation_id}'
        print(f"[Connect] Attempt to join {self.room_group_name}")

        query = parse_qs(self.scope["query_string"].decode())
        token = query.get("token", [None])[0]

        if not token:
            print("[Connect] ❌ No token found")
            return await self.close(code=4001)

        try:
            user_info = self.get_user_info_from_access(token)
            self.scope["user_id"] = user_info["user_id"]
            self.scope["username"] = user_info["username"]
        except TokenError:
            await self.accept()
            await self.send(json.dumps({"type": "token_expired"}))
            return await self.close(code=4003)

        if not await self.is_participant():
            print(f"[Connect] ❌ User not in conversation {self.conversation_id}")
            return await self.close(code=4004)

        participants = await self.get_participant_usernames()
        print(f"[Room] Participants in {self.room_group_name}: {participants}")

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        print(f"[Connect] ✅ {self.scope['username']} joined {self.room_group_name}")

    def get_user_info_from_access(self, token):
        access = AccessToken(token)
        return {
            "user_id": access["user_id"],
            "username": access.get("username")
        }

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        print(f"[Disconnect] {self.scope.get('username')} left {self.room_group_name}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            event_type = data.get("type")
            print(f"[Receive] {event_type}: {data}")

            if event_type == "send_message":
                await self.handle_send_message(data)
            elif event_type == "mark_seen":
                await self.handle_mark_seen()
            elif event_type == "typing":
                await self.handle_typing()
        except Exception as e:
            await self.send(json.dumps({"error": str(e)}))

    async def handle_send_message(self, data):
        message = data.get("message")
        if not message:
            return await self.send(json.dumps({"error": "Message content required."}))

        saved = await self.save_message(
            self.conversation_id, self.scope["user_id"],
            self.scope["username"], message
        )

        # Do NOT broadcast to sender
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "chat_message",
                "message": saved["content"],
                "sender": saved["sender"],
                "timestamp": saved["timestamp"],
                "exclude_channel": self.channel_name  # Exclude this sender
            }
        )

    async def handle_typing(self):
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "typing_event",
                "user": self.scope["username"]
            }
        )

    async def handle_mark_seen(self):
        await self.mark_all_seen(self.conversation_id, self.scope["user_id"])
        await self.send(json.dumps({"type": "seen_confirmed"}))

    async def chat_message(self, event):
        if event.get("exclude_channel") == self.channel_name:
            return  # Don't send back to sender
        await self.send(json.dumps({
            "type": "chat_message",
            "message": event["message"],
            "sender": event["sender"],
            "timestamp": event["timestamp"],
        }))

    async def typing_event(self, event):
        await self.send(json.dumps({
            "type": "typing",
            "user": event["user"]
        }))

    async def new_conversation_event(self, event):
        await self.send(json.dumps({
            "type": "new_conversation",
            "conversation_id": event["conversation_id"],
            "username": event["username"],
        }))

    @database_sync_to_async
    def is_participant(self):
        conv = Conversation.objects.prefetch_related("participants").filter(id=self.conversation_id).first()
        return any(p.id == self.scope["user_id"] for p in conv.participants.all()) if conv else False

    @database_sync_to_async
    def get_participant_usernames(self):
        conversation = Conversation.objects.prefetch_related("participants").filter(id=self.conversation_id).first()
        if not conversation:
            return ["<invalid conversation>"]
        return [user.username for user in conversation.participants.all()]

    @database_sync_to_async
    def save_message(self, conversation_id, user_id, username, content):
        conversation = Conversation.objects.get(id=conversation_id)
        user = User.objects.get(id=user_id)  
        msg = Message.objects.create(
            conversation=conversation,
            sender_id=user,
            sender=username,
            content=content
        )
        return {
            "sender": username,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat()
        }

    @database_sync_to_async
    def mark_all_seen(self, conversation_id, user_id):
        Message.objects.filter(conversation_id=conversation_id, seen=False).exclude(sender_id=user_id).update(seen=True)

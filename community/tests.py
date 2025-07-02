import json
import pytest
import asyncio
from channels.testing import WebsocketCommunicator
from channels.routing import URLRouter
from community.consumers.chatConsumers import ChatConsumer
from community.routing import websocket_urlpatterns
from django.test import TransactionTestCase
from django.contrib.auth import get_user_model
from community.models import Conversation

User = get_user_model()

application = URLRouter(websocket_urlpatterns)

class ChatSocketTest(TransactionTestCase):
    """Requires database because it creates users and conversations"""

    async def asyncSetUp(self):
        self.user1 = User.objects.create_user(username='john', password='123456')
        self.user2 = User.objects.create_user(username='jane', password='123456')

        self.conversation = Conversation.objects.create()
        self.conversation.participants.set([self.user1, self.user2])

    @pytest.mark.asyncio
    async def test_socket_message_send(self):
        communicator = WebsocketCommunicator(application, f"/ws/chat/{self.conversation.id}/")
        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        # Send message
        await communicator.send_json_to({
            "sender": "john",
            "message": "Hello Jane!"
        })

        response = await communicator.receive_json_from()
        self.assertEqual(response["message"], "Hello Jane!")
        self.assertEqual(response["sender"], "john")

        await communicator.disconnect()

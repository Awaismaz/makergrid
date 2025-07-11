from django.urls import path
from .views import UserAssetsView,TextTo3DModelView,AssetListCreateView,AssetRetrieveView,ImageTo3DModelView,GetPredictionStatusView,GetImagePredictionStatusView

urlpatterns = [
    path("text-to-model/", TextTo3DModelView.as_view(), name="text-to-model"),
    path("image-to-model/", ImageTo3DModelView.as_view(), name="image-to-model"),
    path('check-task-status/<task_id>/', GetPredictionStatusView.as_view(), name='check-task-status'),
    path('check-image-task-status/<task_id>/', GetImagePredictionStatusView.as_view(), name='check-task-status'),

    path("assets/", UserAssetsView.as_view(), name="asset-list-create"),
    path("assets/<int:pk>/", AssetRetrieveView.as_view(), name="asset-retrieve")
]

from django.urls import path
from .views import UserAssetsView,TextTo3DModelView,AssetListCreateView,AssetRetrieveView,ImageTo3DModelView

urlpatterns = [
    path("text-to-model/", TextTo3DModelView.as_view(), name="text-to-model"),
    path("image-to-model/", ImageTo3DModelView.as_view(), name="text-to-model"),
    # path("model-job-status/<str:task_id>/", ModelJobStatusView.as_view()),
    path("assets/", UserAssetsView.as_view(), name="asset-list-create"),
    path("assets/<int:pk>/", AssetRetrieveView.as_view(), name="asset-retrieve")
]

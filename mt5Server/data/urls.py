from django.urls import path
import models

urlpatterns = [
    path('upload/', models.uploadForexData),
]
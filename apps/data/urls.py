from django.urls import re_path

from apps.data import views

urlpatterns = [
    re_path(r'^', views.DataView.as_view()),
]

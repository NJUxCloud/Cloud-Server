from django.urls import re_path

from apps.data import views

urlpatterns = [
    re_path(r'^data/$', views.DataView.as_view()),
]

from django.urls import re_path

from apps.preprocess import views

urlpatterns = [
    re_path(r'^type/$', views.PreprocessView.as_view())
]
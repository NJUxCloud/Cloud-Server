from django.urls import re_path

from apps.runtime import views

urlpatterns = [
    re_path(r'^kubernetes/$', views.KuberneteView.as_view()),
    re_path(r'^train/(?P<modelname>\w+)/(?P<iter>[0-9]+)/$', views.TensorResultView.as_view())
]

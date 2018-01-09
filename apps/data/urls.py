from django.urls import re_path

from apps.data import views

urlpatterns = [
    re_path(r'^(?P<pk>[0-9]+)/(?P<relative_path>.+)/$', views.DataDetail.as_view()),
    # re_path(r'^', views.DataView.as_view())
]

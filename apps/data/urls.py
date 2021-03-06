from django.urls import re_path

from apps.data import views

urlpatterns = [
    re_path(r'^(?P<pk>[0-9]+)/$', views.DataDetail.as_view()),
    re_path(r'^list/', views.DataView.as_view()),
    re_path(r'^create/', views.ModelCreation.as_view()),
    re_path(r'tag/', views.TagUpload.as_view()),
    # re_path(r'models/',views.ModelsList.as_view())
]

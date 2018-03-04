from django.urls import re_path

from apps.construction import views

urlpatterns = [
    re_path(r'^config/$', views.ConfigView.as_view()),
    re_path(r'^options/$', views.ConfigOptions.as_view()),
    re_path(r'^detail/(?P<modelname>\w+)/$', views.ConfigDetail.as_view()),
    re_path(r'^construction/(?P<userid>[0-9]+)/(?P<modelname>\w+)/(?P<datatype>\w+)/$', views.ConstructView.as_view()),
]

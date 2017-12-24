from django.urls import re_path

from demo import views

urlpatterns = [
    # path to showing all the bills
    re_path(r'^demo/$', views.ShowBills.as_view()),
    # path to searching bills by its name
    re_path(r'^demo/search/$', views.searchBillByName.as_view()),
    # path to checking bill by id
    re_path(r'^demo/(?P<pk>[0-9]+)/$', views.BillsDetail.as_view()),



]
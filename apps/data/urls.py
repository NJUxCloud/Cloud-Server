from django.urls import re_path

from apps.data.views import code_view

urlpatterns = [
    # 处理代码文件
    re_path(r'^code/$', code_view.CodeView.as_view()),

]
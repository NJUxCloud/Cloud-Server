from django.urls import re_path

from apps.data.views import code_view
from apps.data.views import doc_view

urlpatterns = [
    # 处理代码文件
    re_path(r'^code/$', code_view.CodeView.as_view()),
    re_path(r'^doc/$', doc_view.DocView.as_view()),
]

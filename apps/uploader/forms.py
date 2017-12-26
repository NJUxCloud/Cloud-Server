from django.forms import forms


class FileUploadForm(forms.Form):
    """
    处理文件上传的表单类
    """
    file = forms.FileField()

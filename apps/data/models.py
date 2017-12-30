from django.db import models


# Create your models here.

class RawData(models.Model):
    DOC = 'doc'
    AUDIO = 'audio'
    PICTURE = 'picture'
    CODE = 'code'
    FILE_TYPE_CHOICES = (
        (DOC, 'doc'),
        (AUDIO, 'audio'),
        (PICTURE, 'picture'),
        (CODE, 'code')
    )
    # id auto generated
    # create time
    created_at = models.DateTimeField(auto_now_add=True)
    # file/directory path
    file_path = models.CharField(max_length=100, blank=False)
    # file_type
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    # owner_id 其中定义了相关名，当查询用户的数据时可返回对应条目，否则只能查数据关联的用户
    owner = models.ForeignKey('auth.User', related_name="raw_data" ,on_delete=models.CASCADE)

    def __str__(self):
        return self.file_type + ' ' + self.file_path

    class Meta:
        ordering = ['created_at']

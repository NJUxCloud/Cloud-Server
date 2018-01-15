from django.db import models

# Create your models here.
class Preprocess(models.Model):
    DOC = 'doc'
    AUDIO = 'audio'
    PICTURE = 'picture'
    DATA_TYPE_CHOICES = (
        (DOC, 'doc'),
        (AUDIO, 'audio'),
        (PICTURE, 'picture'),
    )

    #preprocess name
    name=models.CharField(max_length=100, blank=False)

    #preprocess type
    data_type=models.CharField(max_length=10, choices=DATA_TYPE_CHOICES)

    class Meta:
        ordering = ['data_type']

class PreprocessDetail(models.Model):

    # 预处理操作id
    preprocess_id=models.IntegerField(blank=False)
    # 预处理操作参数
    parameter=models.CharField(max_length=100, blank=False)

    class Meta:
        ordering = ['preprocess_id']
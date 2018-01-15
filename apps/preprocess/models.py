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
# coding=utf-8
from django.db import models
# Create your models here.

# define a Bill model as an example
class Bills(models.Model):
    """
        所有自己定义的模型(想当于java的PO)
        要继承 models.Model,之后在下面添加字段
    """
    # create time
    created = models.DateTimeField(auto_now_add=True)
    # name of the goods
    goods = models.CharField(max_length=100, blank=False)
    # price of the goods
    price = models.FloatField()
    # amount of the goods
    amount = models.IntegerField(default=1)
    # description of the goods
    description = models.CharField(default='no description', max_length=100)
    # its owner 如果这个模型是由用户创建,或者不同实例属于不同用户,则要创建此字段
    owner = models.ForeignKey('auth.User', related_name='demo', on_delete=models.CASCADE)

    class Meta:
        """
             这个类应该是数据库要用的,暂时好像用不到
        """
        ordering = ('created', 'owner')

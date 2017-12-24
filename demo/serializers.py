# coding=utf-8
from rest_framework import serializers
from django.contrib.auth.models import User

from demo.models import Bills


class BillSerializer(serializers.ModelSerializer):
    """
    序列化,把模型转化成可传输序列,或者把可传输序列转化到python模型
    可以继承多种类型的 Serializer
    一般是两种
     一个是  ModelSerializer  也就是数据会会有一个主键
     还有一个 HyperlinkedModelSerializer , 是基于超链接的
     我觉得第一个更好理解,方便后面添加外键,以及符合mysql的模式,所以推荐使用第一种
    """
    owner = serializers.ReadOnlyField(source='owner.username')
    class Meta:
        # assign a model to this serializer
        model = Bills
        # declare fields id是必须有的
        fields = ('id', 'goods', 'price', 'amount', 'description', 'owner')


class UserSerializer(serializers.ModelSerializer):
    """
        指明user 和  user 所属类型的转化关系
        因为user的id 是 bill类型的id 所以把bill类作为一个field 加入到user中
        user 是 django rest-auth 框架自带的
    """
    demo = serializers.PrimaryKeyRelatedField(many=True, queryset=Bills.objects.all())

    class Meta:
        model = User
        fields = ('id', 'username', 'demo')

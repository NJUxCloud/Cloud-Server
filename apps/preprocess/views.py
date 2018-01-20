from django.http import Http404
from django.shortcuts import render

from rest_framework import status
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.preprocess.models import Preprocess
from apps.preprocess.serializers import PreprocessSerializer
from apps.preprocess import preprocess
import json


class PreprocessView(APIView):
    # use session
    # authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    # permission_classes = (IsAuthenticated,)

    def post(self, request):
        '''
         获取预处理操作类型
        '''

        """
        add by wsw
        预处理操作api说明
        request.data 传入的json:{
            type:{str} 说明此操作的类型 init:查询第一层预处理  search:查询某个预处理函数  call:调用某个预处理函数
            func:{str} 函数名称 (init 无需此项)
            param:{dict} 调用函数所需要的参数 (init 无需此项)
        }

        返回:
        对于search来说 本次操作为查询该函数需要哪些参数
        返回字典类参说明:back_data{
            "参数字段名":(显示在界面的中文名,参数类型(基础类型),下限,上限)
            "参数字段名":(显示在界面的中文名,参数类型(基础类型),下限,上限)gai
            ...
            "return":{boolean} 该函数是否为最终函数(最终函数是真正执行处理的函数)
        }
        对于init 或 call来说:
        1. 该函数是最终函数:
            {返回值是该函数返回值}
        3.  该函数不是最终函数:
            {
                'functions':[
                    {
                        name:{str}预处理方法函数名
                        name:{str}预处理方在界面现实的名字
                    },
                     {
                        name:{str}预处理方法函数名
                        name:{str}预处理方在界面现实的名字
                    },
                    ...
                ]

            }

        """
        # type=request.GET.get('type')
        # serializer=PreprocessSerializer(self.get_object(type),many=True)
        param = request.data
        print(request.data)
        back_data = {}
        obj = preprocess
        if param['type'] == 'init':
            func = getattr(obj, 'init')
            back_data = func()
        elif param['type'] == 'search':
            if hasattr(obj, param['func']):
                func = getattr(obj, param['func'])
                back_data = func.__annotations__
            else:
                back_data = {"error:""no such function"}

        elif param['type'] == 'call':
            if hasattr(obj, param['func']):

                func = getattr(obj, param['func'])
                p = json.loads(param['param'])
                print(p)
                back_data = func(**p)
            else:
                back_data = {"error:""no such function"}

        return Response(back_data, status=status.HTTP_200_OK)

    def get(self, request, format=None):

        return Response(status=status.HTTP_200_OK)

    def get_object(self, type):
        try:
            return Preprocess.objects.filter(data_type=type)
        except Preprocess.DoesNotExist:
            raise Http404

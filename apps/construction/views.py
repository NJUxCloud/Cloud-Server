from django.shortcuts import render
from rest_framework.views import APIView


# Create your views here.

class ConfigView(APIView):
    # def post(self, request):
    #     '''
    #     读取界面传来的json，生成配置文件, 这一步要直接放到运行时进行
    #     :param request:
    #     :return:
    #     '''
    #     pass

    def get(self, request, format=None):
        '''
        读取配置文件
        :param request:
        :param format:
        :return:
        '''
        pass


class ConstructView(APIView):
    def post(self, request):
        '''
        构造代码和配置文件，将代码、配置文件和数据交给master服务器运行
        :param request:
        :return:
        '''
        pass

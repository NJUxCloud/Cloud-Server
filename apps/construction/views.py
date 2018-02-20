from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from apps.construction.util import options
import json


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
        """
        读取配置文件
        :param request:
        :param format:
        :return:
        """
        pass


class ConstructView(APIView):
    def post(self, request):
        """
        构造代码和配置文件，将代码、配置文件和数据交给master服务器运行
        :param request:
        :return:
        """
        pass

    def create_file(self):
        """
        创建文件
        :return:
        """
        pass

    def write_params(self):
        """
        定义import和命令行参数
        :return:
        """
        pass

    def write_common_functions(self):
        """
        增添公用的方法
        :return:
        """
        pass

    def construct_net(self):
        """
        构建网络结构
        :return:
        """
        pass

    def construct_train(self):
        """
        构建训练过程（包括数据加载，模型创建，定义损失函数，优化器，评价指标，分布式代码）
        :return:
        """
        pass

    def construct_inference(self):
        """
        添加预测部分
        :return:
        """
        pass

    def write_main(self):
        """
        添加main函数
        :return:
        """
        pass


class ConfigOptions(APIView):
    def post(self, request):
        param = request.data
        back_data = {}
        obj = options

        if hasattr(obj, param['option']):
            func = getattr(obj, param['option'])
            try:
                back_data = func()
            except:
                back_data = {'error': 'wrong option'}

        return Response(data=back_data, status=status.HTTP_200_OK)

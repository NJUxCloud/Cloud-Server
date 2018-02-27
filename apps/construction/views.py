import os
import random
import string
import traceback

import shutil
from django.http import HttpResponse
from django.http import HttpResponseNotFound
from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from CloudServer import global_settings
from apps.construction.util import options
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
import json

class ConfigView(APIView):
    authentication_classes = (SessionAuthentication, TokenAuthentication)

    def get(self, request, format=None):
        """
        读取配置文件,因为到时候部署在同一台服务器上，所以保存在本地
        文件目录 NJUCloud/id/model/
        :param request:
        :param format:
        :return: 返回json的格式：["modelname2","modelname"]
        """
        userid = str(self.request.user.id)
        relative_path = 'NJUCloud/' + userid + '/model/'
        local_file_path = global_settings.LOCAL_STORAGE_PATH + relative_path
        list=getModelNameList(local_file_path)
        return HttpResponse(json.dumps(list), content_type='application/json')

def getModelNameList(path):
    '''
    获取所有模型的名字
    '''
    # 所有文件夹，第一个字段是次目录的级别
    dirList = []
    # 返回一个列表，其中包含在目录条目的名称(google翻译)
    files = os.listdir(path)
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            # 添加文件夹
            dirList.append(f)
    return dirList

class ConfigDetail(APIView):
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)

    def get(self, request, modelname, format=None):
        '''
        文件目录 NJUCloud/id/model/{modelname}/model.json
        将文件内容转化为json
        :param request:
        :param pk:
        :param format:
        :return:
        '''
        userid = str(self.request.user.id)
        relative_path = 'NJUCloud/' + userid + '/model/' + modelname + '/model.json'
        local_file_path = global_settings.LOCAL_STORAGE_PATH + relative_path
        print(local_file_path)
        try:
            with open(local_file_path, 'r') as load_f:
                load_dict = json.load(load_f)
                return HttpResponse(json.dumps(load_dict), content_type='application/json')
        except Exception as e:
            print(traceback.print_exc())
            response = HttpResponseNotFound()
        return response


class ConstructView(APIView):

    def post(self,request, userid):
        """
        构造代码和配置文件，将代码、配置文件和数据交给master服务器运行
        :param request:
        :return:
        """
        json_str = request.data
        userid = str(userid)
        self.save_model_file(json_str,userid)

        return Response(status=status.HTTP_200_OK)

    def create_file(self):
        """
        创建文件
        :return:
        """
        pass

    def save_model_file(self,json_str,userid):
        '''
        保存模型文件和脚本文件
        :param json_str:
        :param userid:
        :return:
        '''
        # 生成随机字符
        modelname = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        relative_path = 'NJUCloud/' + userid + '/model/' + modelname
        local_file_path = global_settings.LOCAL_STORAGE_PATH + relative_path
        if (not os.path.exists(local_file_path)):
            os.makedirs(local_file_path)

        #写model的json
        model_path =local_file_path+ "/model.json"
        with open(model_path, "w",encoding='UTF-8') as f:
            json.dump(json_str, f,ensure_ascii=False)

        #写ps脚本
        ps_path = local_file_path + "/ps.sh"
        json_str=json.dumps(json_str,ensure_ascii=False)
        print(json_str)
        shutil.copyfile("apps/construction/ps.sh", ps_path)
        with open(ps_path, 'a+',encoding='UTF-8') as f:
            f.write(json_str)
            f.write("'")

        # 写worker脚本
        worker_path=local_file_path+"/worker.sh"
        shutil.copyfile("apps/construction/worker.sh",worker_path)
        with open(worker_path, 'a+',encoding='UTF-8') as f:
            f.write(json_str)
            f.write("'")

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

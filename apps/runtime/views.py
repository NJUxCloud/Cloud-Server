import json

from django.http import HttpResponse
from django.shortcuts import render

# Create your views here
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.views import APIView

from CloudServer import global_settings
from apps.runtime.util.remote_operation import Linux


class TensorResultView(APIView):
    authentication_classes = (SessionAuthentication, TokenAuthentication)

    def get(self, request, modelname, format=None):
        """
        读取训练结果
        文件目录 NJUCloud/id/model/modelname
        :param request:
        :param format:
        :return: 返回json的格式
        """
        print(modelname)
        userid = str(self.request.user.id)
        host = Linux('119.23.51.139','root','NJUCloud145')
        host.connect()
        relative_dir='NJUCloud/' + userid + '/model/'+modelname
        relative_path =relative_dir +'/result.txt'
        cmd=global_settings.TRAIN_RESULT_ORDER % (relative_path, relative_dir)
        host.send(cmd)
        remote_file_path = relative_path
        local_file_path = global_settings.LOCAL_STORAGE_PATH+global_settings.LOCAL_TRAIN_RESULT_PATH
        host.download(remote_file_path,local_file_path)
        json_data=self.read_train_results(local_file_path)
        host.close()

        return HttpResponse(json_data, content_type='application/json')

    def read_train_results(self,filepath):
        '''
        读取训练结果
        :param filepath: NJUCloud/id/model/modelname/result.txt
        :return: json
        '''
        data = dict()
        with open(filepath, 'r') as f:
            line = f.readline()
            result = []
            accurary=[]
            while (line.startswith("step")):
                line_data = dict()
                strs = line.split(',')
                line_data['step'] = strs[0].split(':')[1]
                line_data['accuracy'] = strs[1].split(':')[1]
                accurary.append(float(line_data['accuracy']))
                line_data['duration'] = strs[2].split(':')[1].strip()
                result.append(line_data)
                line = f.readline()
            data["every_result"] = result
            if(line==None or len(line)<1):
                data["final_accuracy"]=float(sum(accurary)) / len(accurary)
            else:
                data["final_accuracy"] = line.split(':')[1].strip()
        json_data = json.dumps(data)
        return json_data



class KuberneteView(APIView):
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    def get(self, request, format=None):
        """
        读取kubernetes结果，返回给界面
        远程文件目录 /home/info.txt
        :param request:
        :param format:
        :return:返回json格式
        """
        host = Linux('120.78.176.4', 'root', 'NJUCloud175')
        host.connect()
        cmd=global_settings.KUBERNETES_RESULT_ORDER
        host.send(cmd)
        local_file_path=global_settings.LOCAL_STORAGE_PATH+global_settings.LOCAL_KUBERNETES_RESULT_PATH
        remote_file_path="/home/info.txt"
        host.download(remote_file_path,local_file_path)
        json_data=self.read_kubernetes_results(local_file_path);
        host.close()
        return HttpResponse(json_data, content_type='application/json')

    def read_kubernetes_results(self,filepath):
        '''
        读取kubernetes的结果
        :param filepath:
        :return:
        '''
        with open(filepath, 'r') as f:
            # 读取condition
            while (not f.readline().startswith("Condition")):
                continue
            f.readline()
            f.readline()

            data = dict()
            # 处理condition
            conditions = []
            line = f.readline()
            while (not line.startswith("Addresses")):
                # 过滤空格
                condition = dict()
                conditiondata = line.strip().split('  ')
                conditiondata = list(filter(lambda x: len(x) > 0, conditiondata))
                condition['Type'] = conditiondata[0].strip()
                condition['Status'] = conditiondata[1].strip()
                condition['LastHeartbeatTime'] = conditiondata[2].strip()
                condition['LastTransitionTime'] = conditiondata[3].strip()
                condition['Reason'] = conditiondata[4].strip()
                condition['Message'] = conditiondata[5].strip()
                conditions.append(condition)
                line = f.readline()
            data['Conditions'] = conditions

            # 处理addresses
            addr = dict()
            addr['InternalIP'] = f.readline().split(':')[1].strip()
            addr['Hostname'] = f.readline().split(':')[1].strip()
            data['Addresses'] = addr
            f.readline()

            # 处理Capacity
            ca = dict()
            ca['cpu'] = f.readline().split(':')[1].strip()
            ca['memory'] = f.readline().split(':')[1].strip()
            ca['pods'] = f.readline().split(':')[1].strip()
            data['Capacity'] = ca
            f.readline()

            # 处理Allocatable
            al = dict()
            al['cpu'] = f.readline().split(':')[1].strip()
            al['memory'] = f.readline().split(':')[1].strip()
            al['pods'] = f.readline().split(':')[1].strip()
            data['Allocatable'] = al
            f.readline()

            while (not f.readline().startswith(" Kernel Version")):
                continue
            # 处理System Info
            si = dict()
            si['OS Image'] = f.readline().split(':')[1].strip()
            si['Operating System'] = f.readline().split(':')[1].strip()
            si['Architecture'] = f.readline().split(':')[1].strip()
            si['Container Runtime Version'] = f.readline().split(': ')[1].strip()
            si['Kubelet Version'] = f.readline().split(':')[1].strip()
            si['Kube-Proxy Version'] = f.readline().split(':')[1].strip()
            data['System Info'] = si
            f.readline()

            while (not f.readline().startswith("Non-terminated Pods")):
                continue
            f.readline()
            f.readline()
            # 处理Non-terminated Pods
            conditions = []
            line = f.readline()
            while (not line.startswith("Allocated resources")):
                # 过滤空格
                condition = dict()
                conditiondata = line.strip().split('  ')
                conditiondata = list(filter(lambda x: len(x) > 0, conditiondata))
                condition['Namespace'] = conditiondata[0].strip()
                condition['Name'] = conditiondata[1].strip()
                condition['CPU Requests'] = conditiondata[2].strip()
                condition['CPU Limits'] = conditiondata[3].strip()
                condition['Memory Requests'] = conditiondata[4].strip()
                condition['Memory Limits'] = conditiondata[5].strip()
                conditions.append(condition)
                line = f.readline()
            data['Non-terminated Pods'] = conditions
        json_data = json.dumps(data)
        return json_data
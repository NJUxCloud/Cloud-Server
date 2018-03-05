# coding=utf-8
import shutil
import time
import traceback
import os
import urllib.request
import zipfile

from idna import unicode
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from apps.data.models import RawData
from apps.data.util.remote_operation import Linux
from apps.data.serializers import RawDataSerializer

from apps.data.util.csv_handler import Parser
from apps.data.util.file_walker import FileWalker
from django.http import HttpResponse, HttpResponseNotFound
import mimetypes
import CloudServer.global_settings as global_settings


class DataView(APIView):
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        """
        处理上传的文件，情况包括：url， 单个文件、一个zip压缩文件
        :param request:
        :return:
        """
        # 文件类型(url, single, zip)
        file_type = request.POST.get('file_type')
        try:
            if file_type == 'single':
                data_id = self.upload_and_save(request, need_unzip=False)
            elif file_type == 'zip':
                data_id = self.upload_and_save(request, need_unzip=True)
            elif file_type == 'url':
                data_id = self.handle_url(request)
            else:
                return Response(status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            print(traceback.print_exc())
            return Response(status=status.HTTP_400_BAD_REQUEST)
        return Response(data={'data_id': data_id}, status=status.HTTP_200_OK)

    def get(self, request, format=None):
        """
        获得用户所有的数据简介
        :param request:
        :return:
        """
        data = RawData.objects.filter(owner=self.request.user.id)
        serializer = RawDataSerializer(data, many=True)
        return Response(serializer.data)

    def format_name(self, ori_name):
        """
        根据时间生成文件名，防止用户多次上传的文件名一样
        在后缀前加上时间
        TODO 还没考虑后缀名为.tar.gz这种多个组成的情况
        :param ori_name:
        :return:
        """
        name_parts = ori_name.split('.')
        if len(name_parts) == 1:
            filename = name_parts[0] + '_' + time.strftime('%Y%m%d%H%M%S',
                                                           time.localtime(time.time()))
        elif len(name_parts) == 2:
            filename = name_parts[0] + '_' + time.strftime('%Y%m%d%H%M%S',
                                                           time.localtime(time.time())) + '.' + name_parts[1]
        else:
            filename = '.'.join(name_parts[:-1]) + '_' + time.strftime('%Y%m%d%H%M%S',
                                                                       time.localtime(time.time())) + '.' + name_parts[
                           -1]

        return filename

    def upload_and_save(self, request, need_unzip):
        """
        上传文件并保存到数据库
        :param request:
        :return:
        """
        # 文件类别(doc, code, audio, picture)
        file_class = request.POST.get('file_class')
        userid = str(self.request.user.id)
        file = request.FILES.get('file')
        # 根据时间随机生成文件名，防止用户多次上传的文件名一样
        filename = self.format_name(file.name)
        dir_path = 'NJUCloud/' + userid + '/data/' + file_class + '/'
        file_path = dir_path + filename
        self.save_to_local(file=file, dir_path=dir_path, file_path=file_path, need_unzip=need_unzip)
        # self.upload_file(dir_path=dir_path, file_path=file_path, need_unzip=need_unzip)
        data_id = self.save_to_db(file_type=file_class, file_path=file_path, need_unzip=need_unzip)
        return data_id

    def save_to_db(self, file_path, file_type, need_unzip):
        """
        存储到数据库
        :param user_id:
        :param file_path:
        :param file_type: RawData.DOC, RawData.CODE, ...
        :return:
        """
        # 去掉.zip的后缀
        if (need_unzip):
            file_path = file_path.split('.')[0]
        raw_data = RawData(file_path=file_path, file_type=file_type, owner=self.request.user)
        raw_data.save()

        return raw_data.id

    # def upload_file(self, dir_path, file_path, need_unzip):
    #     """
    #     上传文件，如果需要解压则最后need_unzip会是true
    #     """
    #     host = Linux()
    #     host.connect()
    #     host.sftp_upload_file(dir_path, file_path, need_unzip)
    #     host.close()

    def save_to_local(self, file, dir_path, file_path, need_unzip):
        '''
        将文件保存到本地，如果是压缩文件就进行解压缩
        :param file: 文件
        :param dir_path: 文件夹目录 NJUCloud/id/data/type/
        :param file_path: 文件目录 NJUCloud/id/data/type/xxx
        :param need_unzip: 是否需要解压缩
        :return:
        '''
        try:
            dir_path = global_settings.LOCAL_STORAGE_PATH + file_path.split('.')[0]
            file_path = global_settings.LOCAL_STORAGE_PATH + file_path
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            if need_unzip:
                f = zipfile.ZipFile(file_path, 'r')
                for fname in f.namelist():
                    # 解决文件名中文乱码问题
                    filename = fname.encode('cp437').decode('utf8')
                    output_filename = os.path.join(dir_path, filename)
                    output_file_dir = os.path.dirname(output_filename)
                    if (filename.endswith('/')):
                        if not os.path.exists(output_file_dir):
                            os.makedirs(output_file_dir)
                    else:
                        with open(output_filename, 'wb') as output_file:
                            shutil.copyfileobj(f.open(fname), output_file)
                f.close()

        except Exception as e:
            print(traceback.print_exc())
            return Response(status=status.HTTP_400_BAD_REQUEST)

    def handle_url(self, request):
        """
        在远程服务器执行命令进行下载
        通过url下载数据集到指定位置
        如果url有多个用;隔离开
        :param request:
        :return:
        """
        # 文件类别(doc, code, audio, picture)
        file_class = request.POST.get('file_class')
        userid = str(self.request.user.id)
        urls = request.POST.get('url').split(';')
        filename = self.format_name("url")
        dir_path = 'NJUCloud/' + userid + '/data/' + file_class + '/'
        file_path = dir_path + filename
        # host = Linux()
        # host.connect()
        # for url in urls:
        #     command = 'wget -c -P ./' + file_path + ' ' + url
        #     host.send(cmd=command)
        # host.close()
        self.handle_url_local(urls=urls, file_path=file_path)
        data_id = self.save_to_db(file_type=file_class, file_path=file_path, need_unzip=False)
        return data_id

    def handle_url_local(self, urls, file_path):
        '''
        将url文件缓存到本地
        :param request:
        :return:
        '''
        local_file_path = global_settings.LOCAL_STORAGE_PATH + file_path
        if not os.path.exists(local_file_path):
            os.makedirs(local_file_path)
        for url in urls:
            try:
                file_name = local_file_path + '/' + str(url).split('/')[-1]
                urllib.request.urlretrieve(url, file_name)
            except Exception:
                print(traceback.print_exc())
                print('\tError retrieving the URL:', url)


class DataDetail(APIView):
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def get(self, request, pk, format=None):
        relative_path = request.GET.get('relative_path')
        return self.get_object(pk=pk, relative_path=relative_path)

    def delete(self, request, pk, format=None):
        data = RawData.objects.get(pk=pk)
        if data is None:
            return Response(status=status.HTTP_204_NO_CONTENT)
        data.delete()
        return Response(status=status.HTTP_200_OK, data={'message': 'success'})

    def get_object(self, pk, relative_path=None):
        """
        获得一份数据全部内容
        :param relative_path: 目录中文件的相对路径
        :param pk: 数据文件id
        :return:
        """
        raw_data = RawData.objects.get(pk=pk)
        if relative_path is not None:
            userid = str(raw_data.owner.id)
            # 文件类别(doc, code, audio, picture)
            file_class = raw_data.file_type
            relative_path = global_settings.LOCAL_STORAGE_PATH + 'NJUCloud/' + userid + '/data/' + file_class + '/' + relative_path
            local_file_path = global_settings.LOCAL_STORAGE_PATH + relative_path
            if local_file_path.endswith('.csv'):
                file_type = RawData.DOC
            else:
                file_type = None
        else:
            local_file_path = raw_data.file_path
            # filename = remote_file_path.split('/')[-1]
            # TODO 本地文件存储路径
            # local_file_path = global_settings.LOCAL_STORAGE_PATH + remote_file_path
            file_type = raw_data.file_type

            # 先判断在本地是否存在这一份数据的缓存
            # if not os.path.exists(local_file_path):
            # 如果不存在，将数据从远程加载到本地
            # host = Linux()
            # host.connect()
            # host.download(remote_path=remote_file_path, local_path=local_file_path)
            # host.close()

        # csv 将csv转换为json
        if file_type == RawData.DOC and local_file_path.endswith('csv'):
            parser = Parser()
            json_data = parser.csv_to_json(local_file_path=local_file_path)
            return HttpResponse(json_data, content_type='application/json')

        # 单个图片，或单个音频文件，或单个代码文件，直接返回
        if os.path.isfile(local_file_path):
            try:
                mimetypes.init()
                fsock = open(local_file_path, "rb")
                filename = os.path.basename(local_file_path)
                mime_type_guess = mimetypes.guess_type(filename)
                print(mime_type_guess[0])
                if mime_type_guess is not None:
                    response = HttpResponse(content=fsock.read(), content_type=mime_type_guess[0])
                else:
                    response = HttpResponse(fsock)
                response['Content-Disposition'] = 'attachment; filename=' + filename
            except IOError:
                response = HttpResponseNotFound()
            return response
        else:
            # 以目录形式存储的，返回目录
            walker = FileWalker()
            json_data = walker.get_dir_tree_json(local_file_path)
            return HttpResponse(json_data, content_type='application/json')


class ModelCreation(APIView):
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        user_id = self.request.user.id
        if user_id is None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        dir_path = global_settings.LOCAL_STORAGE_PATH + 'NJUCloud/' + user_id + '/model/'

        if not os.path.exists(dir_path):
            os.makedirs(path=dir_path)

        try:
            sub_dir_path = dir_path + '/' + request.POST.get('modelName') + '/'
            os.mkdir(sub_dir_path)
            return Response(status=status.HTTP_200_OK)
        except:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TagUpload(APIView):
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        user_id = self.request.user.id
        model_name = request.POST.get('modelName')
        tag_file = request.FILES.get('file')

        try:
            file_path = global_settings.LOCAL_STORAGE_PATH + 'NJUCloud/' + user_id + '/model/' + model_name + '/' + 'tag.json'

            with open(file_path, 'wb+') as destination:
                for chunk in tag_file.chunks():
                    destination.write(chunk)

            return Response(status=status.HTTP_200_OK)
        except:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class ModelsList(APIView):
#     authentication_classes = (SessionAuthentication, TokenAuthentication)
#     # use permission, in this case, we use the permission subclass from framework
#     permission_classes = (IsAuthenticated,)
#
#     def get(self, request, format=None):
#         user_id = str(self.request.user.id)
#
#         model_dir_loc = global_settings.LOCAL_STORAGE_PATH + 'NJUCloud/' + user_id + '/model/'
#
#         model_dirs = [{os.path.join(model_dir_loc, o): o} for o in os.listdir(model_dir_loc) if
#                       os.path.isdir(os.path.join(model_dir_loc, o))]
#
#         models_list = []
#         for model_dir in model_dirs:
#             for k, v in model_dir.items():
#                 cur = dict()
#                 cur['modelName'] = v
#                 cur['creationTime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(k).st_ctime))
#                 models_list.append(cur)
#
#         res = dict()
#         res['userId'] = user_id
#         res['models'] = models_list
#
#         return Response(data=res, status=status.HTTP_200_OK)

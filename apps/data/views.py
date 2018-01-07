# coding=utf-8
import time
import traceback
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from apps.data.models import RawData
from django.contrib.auth.models import User
from apps.data.util.remote_operation import Linux


class DataView(APIView):

    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        '''
        处理上传的文件，情况包括：url， 单个文件、一个zip压缩文件
        :param request:
        :return:
        '''
        # 文件类型(url, single, zip)
        file_type = request.POST.get('file_type')
        try:
            if file_type == 'single':
                self.upload_and_save(request,need_unzip=False)
            elif file_type == 'zip':
                self.upload_and_save(request,need_unzip=True)
            elif file_type == 'url':
                pass

        except Exception as e:
            print(traceback.print_exc())
            return Response(status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_200_OK)

    def get(self, request):
        return

    def format_name(self, ori_name):
        """
        根据时间生成文件名，防止用户多次上传的文件名一样
        在后缀前加上时间
        TODO 还没考虑后缀名为.tar.gz这种多个组成的情况
        :param ori_name:
        :return:
        """
        name_parts = ori_name.split('.')
        if len(name_parts) == 2:
            filename = name_parts[0] + '_' + time.strftime('%Y%m%d%H%M%S',
                                                           time.localtime(time.time())) + '.' + name_parts[1]
        else:
            filename = '.'.join(name_parts[:-1]) + '_' + time.strftime('%Y%m%d%H%M%S',
                                                                       time.localtime(time.time())) + '.' + name_parts[
                           -1]

        return filename

    def save_to_db(self, file_path, file_type):
        """
        存储到数据库
        :param user_id:
        :param file_path:
        :param file_type: RawData.DOC, RawData.CODE, ...
        :return:
        """
        raw_data = RawData(file_path=file_path, file_type=file_type, owner=self.request.user)
        raw_data.save()

    def upload_and_save(self,request,need_unzip):
        '''
        上传文件并保存到数据库
        :param request:
        :return:
        '''
        # 文件类别(doc, code, audio, picture)
        file_class = request.POST.get('file_class')
        userid = str(self.request.user.id)
        file = request.FILES.get('file')
        # 根据时间随机生成文件名，防止用户多次上传的文件名一样
        filename = self.format_name(file.name)
        dir_path = 'NJUCloud/' + userid + '/data/' + file_class + '/'
        file_path = dir_path + filename
        self.upload_file(file=file, dir_path=dir_path, file_path=file_path,need_unzip=need_unzip)
        self.save_to_db(file_type=file_class, file_path=file_path)

    def upload_file(self, file, dir_path, file_path,need_unzip):
        '''
        上传文件，如果需要解压则最后need_unzip会是true
        :param file:
        :param dir_path:
        :param file_path:
        :param need_unzip:
        :return:
        '''
        host = Linux()
        host.connect()
        host.sftp_upload_file(file, dir_path, file_path)
        if need_unzip:
            host.unzip_file(file_path)
        host.close()


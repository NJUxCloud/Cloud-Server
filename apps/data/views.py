# coding=utf-8
import time
import traceback
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from apps.data.models import RawData
from django.contrib.auth.models import User
from apps.data.util.remote_operation import Linux


class DataView(APIView):
    def post(self, request):
        '''
        处理上传的文件，情况包括：url， 单个文件、一个zip压缩文件
        :param request:
        :return:
        '''
        # 文件类别(doc, code, audio, picture)
        file_class = request.POST.get('file_class')
        # 文件类型(url, single, zip)
        file_type = request.POST.get('file_type')
        try:
            if file_class == 'doc' and file_type == 'single':
                userid = request.POST.get('userid')
                file = request.FILES.get('file')
                # 根据时间随机生成文件名，防止用户多次上传的文件名一样
                filename = self.format_name(file.name)
                dir_path = 'NJUCloud/' + userid + '/data/doc/'
                file_path = dir_path + filename
                self.upload_file(file=file, dir_path=dir_path, file_path=file_path)
                self.save_to_db(user_id=userid, file_type=RawData.DOC, file_path=file_path)
            elif file_class == 'code':
                pass
            elif file_class == 'audio':
                pass
            elif file_class == 'picture':
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

    def save_to_db(self, user_id, file_path, file_type):
        """
        存储到数据库
        :param user_id:
        :param file_path:
        :param file_type: RawData.DOC, RawData.CODE, ...
        :return:
        """
        user = User.objects.get(pk=user_id)
        raw_data = RawData(file_path=file_path, file_type=file_type, owner=user)
        raw_data.save()

    def upload_file(self, file, dir_path, file_path):
        host = Linux()
        host.connect()
        host.sftp_upload_file(file, dir_path, file_path)
        host.close()

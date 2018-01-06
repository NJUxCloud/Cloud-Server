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
            if file_class == 'doc':
                userid = request.POST.get('userid')
                file = request.FILES.get('file')
                # 根据时间随机生成文件名，防止用户多次上传的文件名一样
                name_parts = file.name.split('.')
                if len(name_parts) == 2:
                    filename = name_parts[0] + '_' + time.strftime('%Y%m%d%H%M%S',
                                                                   time.localtime(time.time())) + '.' + name_parts[1]
                else:
                    filename = '.'.join(name_parts[:-1]) + '_' + time.strftime('%Y%m%d%H%M%S',
                                                                               time.localtime(time.time())) + '.' + \
                               name_parts[-1]
                dir_path = 'NJUCloud/' + userid + '/data/doc/'
                file_path = dir_path + filename
                host = Linux()
                host.connect()
                host.sftp_upload_file(file, dir_path, file_path)
                host.close()

                # save file_path to database
                user = User.objects.get(pk=userid)
                raw_data = RawData(file_path=file_path, file_type=RawData.DOC, owner=user)

                raw_data.save()
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


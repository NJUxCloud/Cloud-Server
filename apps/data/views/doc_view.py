# coding=utf-8
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
import time

from apps.data.util.remote_operation import Linux

from apps.data.models import RawData


class DocView(APIView):
    def post(self, request):
        '''
        处理上传的文本文件，默认用户只上传一个
        :param request:
        :return:
        '''
        try:
            userid = request.POST.get('userid')
            file = request.FILES.get('file')
            # TODO 考虑可以根据时间随机生成文件名, 否则如果用户多次上传的文件名一样怎么办?
            filename = file.name + '_' + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            dir_path = 'NJUCloud' + userid + '/data/doc'
            file_path = dir_path + filename
            host = Linux()
            host.connect()
            host.sftp_upload_file(file, dir_path, file_path)
            host.close()

            # save to file_path to database
            new_raw_data = RawData(file_path=file_path, file_type=RawData.DOC, owner=userid)
            new_raw_data.save()
        except Exception as e:
            print(e)
            return Response(status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_200_OK)

    def get(self, request):
        return

# coding=utf-8
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.data.util.remote_operation import Linux


class CodeView(APIView):
    def get(self, request):
        return

    def post(self, request):
        '''
           处理上传的项目代码，默认为压缩文件zip格式
        '''
        try:
            userid = request.POST.get('userid')
            file = request.FILES.get('file')
            filename = file.name
            dir_path='NJUCloud/'+userid + '/data/code/'
            file_path = dir_path + filename
            host=Linux()
            host.connect()
            host.sftp_upload_file(file,dir_path,file_path)
            host.close()

            # if not os.path.exists(dir_path):
            #     os.makedirs(dir_path)
            #
            # f = open(file_path, 'wb')
            # for chunk in file.chunks():
            #     f.write(chunk)
            #     f.close()
        except Exception as e:
            print(e)
            return Response( status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_200_OK)
from rest_framework import status
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from CloudServer import global_settings

import json
import os

from apps.data.models import RawData
import apps.preprocess.preprocess as preprocess

op_map = {
    "上下翻转": "flip_up_down",
    "左右翻转": "flip_left_right",
    "对角线翻转": "transpose_image",
    "亮度调整": "adjust_brightness",
    "随机亮度调整": "random_brightness",
    "对比度调整": "adjust_contrast",
    "随机对比度调整": "random_contrast",
    "色相调整": "adjust_hue",
    "随机色相调整": "random_hue",
    "饱和度调整": "adjust_saturation",
    "随机饱和度调整": "random_saturation",
    "标准归一化": "standardize",
    "均值滤波": "mean_filter",
    "高斯模糊": "gaussian_blur",
    "中值滤波": "median_filter",
    "灰度非局部平均值去噪": "nl_denoise_gray",
    # "彩色非局部平均值去噪": "nl_denoise_colored",
    "添加椒盐噪声": "add_salt_pepper_noise"
}


class PreprocessView(APIView):
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        """
        数据预处理
        :param request:
        :return:
        """
        try:
            params = request.data

            data_id = params['dataId']
            model_name = params['modelName']
            operations = params['operations']

            # 将数据从data文件夹拷贝到模型文件夹
            data = RawData.objects.get(pk=data_id)
            data_location = data.file_path
            # data_location = '/Users/keenan/Downloads/NJUCloud/3/data/pics'

            model_data_dir = global_settings.LOCAL_STORAGE_PATH + 'NJUCloud/' + \
                             str(self.request.user.id) + '/model/' + model_name + '/data'

            if not os.path.exists(model_data_dir):
                os.mkdir(model_data_dir)

            cmd = 'cp -R ' + data_location + '/*' + ' ' + model_data_dir
            os.system(cmd)

            tag_location = global_settings.LOCAL_STORAGE_PATH + 'NJUCloud/' + \
                           str(self.request.user.id) + '/model/' + model_name + '/tag.json'

            # 读取预处理操作，并按顺序进行处理
            for operation in operations:
                self.execute(operation, model_data_dir + '/', tag_location)

            return Response(data={'message': 'success'}, status=status.HTTP_200_OK)
        except:
            return Response(data={'message': 'error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request, format=None):
        return Response(data={'message': 'error'}, status=status.HTTP_501_NOT_IMPLEMENTED)

    def get_object(self, type):
        pass

    def execute(self, operation, data_dir, tag_location):
        """
        {
        "operationName":"中值滤波",
        "overlap":false,
        "value1":5,
        "value2":5
      }
        :param operation:
        :param data_dir:
        :param tag_location:
        :return:
        """
        print(type(operation))
        print(operation)

        pp = preprocess
        if hasattr(pp, op_map.get(operation['operationName'])):
            with open(tag_location, 'r') as f:
                tags = json.load(f)
                tags_edit = tags.copy()
                f.close()

            func = getattr(pp, op_map.get(operation['operationName']))

            for data_relative_loc, data_tag in tags.items():
                data_loc = data_dir + data_relative_loc
                # 将图像大小变为 28*28
                preprocess.resize(data_loc)

                if operation.get('overlap') is 'true' or operation.get('overlap') is True:
                    func(data_loc, True, operation.get('value1'), operation.get('value2'))
                else:
                    func(data_loc, False, operation.get('value1'), operation.get('value2'))

                # 在tag文件中添加新的
                if operation.get('overlap') is 'true' or operation.get('overlap') is True:
                    parts = os.path.splitext(data_relative_loc)
                    new_relative_loc = parts[0] + '_copy' + parts[1]
                    tags_edit[new_relative_loc] = data_tag

            with open(tag_location, 'w') as f:
                jsObj = json.dumps(tags_edit)
                f.write(jsObj)
                f.close()

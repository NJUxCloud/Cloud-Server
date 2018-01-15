from django.http import Http404
from django.shortcuts import render

from rest_framework import status
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.preprocess.models import Preprocess
from apps.preprocess.serializers import PreprocessSerializer


class PreprocessView(APIView):
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        return Response(status=status.HTTP_200_OK)

    def get(self, request, format=None):
        '''
        获取预处理操作类型
        '''
        type=request.GET.get('type')
        serializer=PreprocessSerializer(self.get_object(type),many=True)
        return Response(serializer.data,status=status.HTTP_200_OK)

    def get_object(self, type):
        try:
            return Preprocess.objects.filter(data_type=type)
        except Preprocess.DoesNotExist:
            raise Http404

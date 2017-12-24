# coding=utf-8
from django.http import Http404
from rest_framework import status
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from demo.models import Bills
from demo.permission import IsOwnerOrReadOnly
from demo.serializers import BillSerializer

# Create your views here.


class ShowBills(APIView):
    """
     展示账单
     相当于一个API(前端需要的方法, 要继承APIView)
     也可以用装饰器或者 viewset来实现

     在这个例子中,所有的返回都是数据库中的数据
     但在实际开发中,可以返回自己定义的json
    """
    # use session
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # use permission, in this case, we use the permission subclass from framework
    permission_classes = (IsAuthenticated,)

    def get(self, request, format=None):
        """
        ShowBills 的get方法
        将数据库中所有的bills按照format格式返回
        :param request:
        :param format:
        :return:
        """
        bills = Bills.objects.all()
        serializer = BillSerializer(bills, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        """
        创建一个新的 bills (前提是用户登录)
        :param request:  request from frontend
        :param format:   respone formate
        :return:     {json} or serializer in proper format
        """
        serializer = BillSerializer(data=request.data)
        if serializer.is_valid():
            self.perform_create(serializer)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class searchBillByName(APIView):
    """
      通过bill中货物的名字,查找bill
    """
    authentication_classes = (SessionAuthentication, TokenAuthentication)

    def get_object(self, name):
        print(name)
        try:
            return Bills.objects.filter(goods__contains=name)
        except Bills.DoesNotExist:
            raise Http404

    def get(self, request , format=None):
        snippet = self.get_object(request.data.get('name'))
        serializer = BillSerializer(snippet, many=True)
        return Response(serializer.data)


class BillsDetail(APIView):
    """
     根据id 查看bill
    """
    authentication_classes = (SessionAuthentication, TokenAuthentication)
    # # use permission, in this case, we use the permission subclass defined by ourselves
    permission_classes = (IsOwnerOrReadOnly,)

    def get_object(self, pk):
        try:
            return Bills.objects.get(pk=pk)
        except Bills.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        snippet = self.get_object(pk)
        serializer = BillSerializer(snippet)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        snippet = self.get_object(pk)
        serializer = BillSerializer(snippet, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        snippet = self.get_object(pk)
        snippet.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

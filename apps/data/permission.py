# coding=utf-8
from rest_framework import permissions


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    只有用户自己能访问自己的数据
    """

    def has_object_permission(self, request, view, obj):

        return obj.owner == request.user
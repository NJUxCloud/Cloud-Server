# coding=utf-8
from rest_framework import permissions


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    使每个Bill只允许其创建者编辑它
    任何用户或者游客都可以访问任何Bill
    自定义权限,
    操作分为安全操作和非安全操作
    安全操作: 不会修改数据,比如GRT
    非安全操作: 可以修改数据 比如delete 和 post等
    所以可以根据不同类型的操作对不同的用户规定权限
    要继承permissions.BasePermission
    也可以使用框架自己写好的一些权限方式
    """

    def has_object_permission(self, request, view, obj):
        # 任何用户或者游客都可以访问任何Bills，所以当请求动作在安全范围内，
        # 也就是GET，HEAD，OPTIONS请求时，都会被允许
        if request.method in permissions.SAFE_METHODS:
            return True

        # 而当请求不是上面的安全模式的话，那就需要判断一下当前的用户
        # 如果Snippet所有者和当前的用户一致，那就允许，否则返回错误信息
        return obj.owner == request.user
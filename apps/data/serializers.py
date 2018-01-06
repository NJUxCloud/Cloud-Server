from rest_framework import serializers
from django.contrib.auth.models import User

from apps.data.models import RawData


class RawDataSerializer(serializers.ModelSerializer):
    # owner = serializers.ReadOnlyField(source='owner.id')
    class Meta:
        model = RawData
        fields = ('id', 'created_at', 'file_path', 'file_type', 'owner')


class UserSerializer(serializers.ModelSerializer):
    relation = serializers.PrimaryKeyRelatedField(many=True, queryset=RawData.objects.all())

    class Meta:
        model = User
        fields = ('id', 'username', 'relation')

from rest_framework import serializers
from django.contrib.auth.models import User

from apps.data.models import RawData


class RawDataSerializer(serializers.ModelSerializer):
    # owner = serializers.ReadOnlyField(source='owner.id')
    file_name = serializers.SerializerMethodField()

    class Meta:
        model = RawData
        fields = ('id', 'created_at', 'file_type', 'file_name', 'owner')

    def get_file_name(self, obj):
        path_parts = obj.file_path.split('/')
        return path_parts[-1]


class UserSerializer(serializers.ModelSerializer):
    relation = serializers.PrimaryKeyRelatedField(many=True, queryset=RawData.objects.all())

    class Meta:
        model = User
        fields = ('id', 'username', 'relation')

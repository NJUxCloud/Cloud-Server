from rest_framework import serializers

from apps.preprocess.models import Preprocess


class PreprocessSerializer(serializers.ModelSerializer):

    class Meta:
        model = Preprocess
        fields = ('id', 'name', 'data_type')



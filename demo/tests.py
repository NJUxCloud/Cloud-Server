# Create your tests here.
# ignore the unresolved problem
from rest_framework.test import APITestCase
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status

from demo import views
from demo.models import Bills


class SearchByNameTestCase(APITestCase):
    def setUp(self):
        self.superuser = User.objects.create_superuser('testman', 'test@man.com', 'passw123')
        self.client.force_authenticate(user=self.superuser)
        bill = Bills.objects.create(goods='banana', amount=10, price=6, owner=self.superuser)
        bill.save()
        bill = Bills.objects.create(goods='apple', amount=9, price=5, owner=self.superuser)
        bill.save()

    def test_get_bills(self):
        response = self.client.get(reverse('bill-list'))
        self.assertEqual(response.data[0]['goods'],'banana')
        self.assertEqual(response.data[1]['price'], 5)


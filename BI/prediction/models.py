from django.db import models
import os, json
from django.conf import settings

# Create your models here.
class Data:
    @staticmethod
    def get_data():
        filepath = os.path.join(settings.BASE_DIR, 'visualization/data.json')
        #filepath = os.path.join('/home/saulopez/DjangoBI/BI/visualization/data.json')
        with open(filepath, 'r') as file_object:
            data = json.load(file_object)
        return data
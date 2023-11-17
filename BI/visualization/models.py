from django.db import models
from django.conf import settings
import os
import json

# Create your models here.

class Data:
    @staticmethod
    def get_data():
        filepath = os.path.join(settings.BASE_DIR, 'visualization/data.json')
        with open(filepath, 'r') as file_object:
            data = json.load(file_object)
        return data
from django.db import models
import pickle

# Create your models here.




class ImageData(models.Model):
    name = models.CharField(max_length=256)
    path = models.CharField(max_length=1024, default="")
    date = models.DateField()
    rate = models.FloatField()
    _value = models.BinaryField()

    def set_data(self, data):
        self._value = pickle.dumps(data)

    def get_data(self):
        return pickle.loads(self._value)

    #value = property(get_data, set_data)
"""
class ImageRate(models.Model):
    name = models.CharField(max_length=256)
    path = models.CharField(max_length=1024, default="")
    date = models.DateField()
    rate = models.FloatField()
    uuid = models.UUIDField()

class ImageCont(models.Model):
    uuid = models.UUIDField()
    _cont = models.BinaryField()
    def set_data(self, data):
        self._cont = pickle.dumps(data)
    def get_data(self):
        return pickle.loads(self._cont)

class ImageArea(models.Model):
    uuid = models.UUIDField()
    _area = models.BinaryField()
    def set_data(self, data):
        self._area = pickle.dumps(data)
    def get_data(self):
        return pickle.loads(self._area)
"""

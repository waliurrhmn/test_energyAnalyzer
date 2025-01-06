# models.py
from django.db import models
from django.contrib.auth.models import User

class UserSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    ceiling_kwh = models.FloatField(help_text="Quarterly energy ceiling in KWH")
    high_price_start = models.TimeField()
    high_price_end = models.TimeField()
    price_high = models.DecimalField(max_digits=6, decimal_places=2)
    price_low = models.DecimalField(max_digits=6, decimal_places=2)
    solar_return_high = models.DecimalField(max_digits=6, decimal_places=2, null=True, blank=True)
    solar_return_low = models.DecimalField(max_digits=6, decimal_places=2, null=True, blank=True)
    has_solar = models.BooleanField(default=False)
    solar_setup_type = models.CharField(max_length=50, null=True, blank=True)

class EnergyData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='uploads/')
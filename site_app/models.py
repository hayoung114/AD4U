from django.db import models


class Advertise(models.Model):
    ad_num = models.AutoField(primary_key=True)
    link = models.TextField(blank=True, null=True)
    title = models.CharField(max_length=45, blank=True, null=True)
    cnt = models.IntegerField(blank=True, null=True)
    length = models.IntegerField(blank=True, null=True)
    tg_num = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'advertise'


class Slot(models.Model):
    slot_num = models.AutoField(primary_key=True)
    time_from = models.IntegerField(blank=True, null=True)
    time_to = models.IntegerField(blank=True, null=True)
    cnt = models.IntegerField(blank=True, null=True)
    tg_num = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'slot'


class Target(models.Model):
    target_num = models.AutoField(primary_key=True)
    sex = models.CharField(max_length=45, blank=True, null=True)
    age_from = models.IntegerField(blank=True, null=True)
    age_to = models.IntegerField(blank=True, null=True)
    total_cnt = models.IntegerField()
    slot_cnt = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'target'


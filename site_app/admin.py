from django.contrib import admin

from site_app.models import Advertise, Slot, Target


@admin.register(Target)
class TargetAdmin(admin.ModelAdmin):
    list_display = ('target_num', 'sex', 'age_from', 'age_to', 'total_cnt', 'slot_cnt')

@admin.register(Advertise)
class AdvertiseAdmin(admin.ModelAdmin):
    list_display = ('ad_num', 'link', 'title', 'cnt', 'length', 'tg_num')

@admin.register(Slot)
class SlotAdmin(admin.ModelAdmin):
    list_display = ('slot_num', 'time_from', 'time_to', 'cnt', 'tg_num')


# Register your ai_models here.

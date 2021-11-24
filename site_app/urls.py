from django.urls import path

from site_app import views


app_name = 'site_app'
urlpatterns = [
    path('index/', views.index, name='index'),
    path('chart/', views.chart, name='charts'),
    path('table/', views.table, name='tables'),
]
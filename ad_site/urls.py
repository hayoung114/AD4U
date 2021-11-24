"""ad_site URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from xml.etree.ElementInclude import include

from django.contrib import admin
from django.urls import path, include
import threading
from ai_project import main_OpenCV_SSD


#웹캠 띄우는 함수 Webcam()을 thread로 실행
thr = threading.Thread(target=main_OpenCV_SSD.Webcam)
print("thread1-webcam")
thr.start()
#ai모델 사용하는 함수 project()을 thread로 실행
t = threading.Thread(target=main_OpenCV_SSD.Project)
print("thread2-project")
t.start()


urlpatterns = [
    path('admin/', admin.site.urls),

    path('', include('site_app.urls'), name='home'),


]

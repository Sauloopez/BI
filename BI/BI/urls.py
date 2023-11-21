"""BI URL Configuration

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
from django.contrib import admin
from django.urls import path
from home.views import index, signup, login_view, logout
from prediction.views import index as prediction_index
from visualization.views import index as visualization_index
from visualization.views import NASADataView
from prediction.views import PredictionSVR
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('accounts/signup/', signup, name='signup'),
    path('accounts/login/', login_view, name='login'),
    path('accounts/logout/', LogoutView.as_view(), name='logout'),
    path('prediction/', prediction_index, name='prediction'),
    path('nasa-data/', NASADataView.as_view(), name='nasa-data'),
    path('predictions-svr/', PredictionSVR.as_view(), name='predictions-svr'),
    path('visualization/', visualization_index, name='visualization'),
]

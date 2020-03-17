from django.urls import path, re_path
from django.urls import path
from django.conf.urls import url, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

from . import views

urlpatterns = [
    url(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATICFILES_DIRS[0]}),

    path("", views.indexView, name="indexView"),
    path('video_feed', views.video_feed, name = "video_feed"),
    path('camera', views.camera, name = 'camera'),
    path('capture', views.capture, name = 'capture'),
    path('search', views.search, name = 'search'),
    path('progress', views.progress, name = 'progress'),
    path('get_tmpl_names', views.get_tmpl_names, name = 'get_tmpl_names'),
    path('get_file_info', views.get_file_info, name = 'get_file_info'),
    path('set_file_info', views.set_file_info, name = 'set_file_info'),
    #path('convert_directories', views.convert_directories, name = 'convert_directories'),
    path('save_captured_img', views.save_captured_img, name = 'save_captured_img')
]
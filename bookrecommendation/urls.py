
from django.urls import path, include
from bookrecommendation import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signin/', views.signin, name='signin'),
    path('signup/', views.signup, name='signup'),
    path('register/', views.register, name='register'),
    path('authenticate/', views.authenticate_user, name='authenticate_user'),
    path('profile/', views.profile, name='profile'),
    path('logout/', views.logout_view, name='logout_view'),
    path('search_items/', views.search_items, name='search_items'),
    path('book_info/<int:book_id>/', views.book_info, name='book_info'),
    path('success_payment/<int:book_id>/', views.success_payment, name='success_payment'),



]

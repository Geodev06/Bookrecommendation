from django.shortcuts import render, redirect, get_object_or_404
import os
import csv
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout
from django.shortcuts import HttpResponseRedirect
from .models import Book
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Avg, Count
import random
from .forms import UserRegistrationForm, UserLoginForm
from .models import CustomUser
from django.contrib import messages
from django.http import JsonResponse

# Assuming your Django app name is 'yourapp'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
   random_books = list(Book.objects.order_by('?')[:50])
    # Fetch top 10 books with average rating and total reviews pre-calculated
   top_books = Book.objects.annotate(
        avg_rating=Avg('rating'),
        total_reviews=Count('reviews')
    ).order_by('-avg_rating', '-total_reviews')[:10]
   return render(request, 'index.html', {'selected_books': random_books, 'top_books': top_books})

def signin(request):
    return render(request, 'auth/signin.html')


def signup(request):
    return render(request, 'auth/signup.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)  # Save the user object but don't commit to the database yet
            user.set_password(form.cleaned_data['password'])  # Hash the password
            user.save()  # Now save the user with the hashed password
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=raw_password)  # Authenticate the user
            if user is not None:
                login(request, user)  # Log the user in
                print(user)
                return redirect('/')  # Redirect to the user's profile page
            else:
                print("User authentication failed.")
        else:
            print("Form is invalid.")
    else:
        form = UserRegistrationForm()
    return render(request, 'auth/signup.html', {'form': form})

def authenticate_user(request):

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)  # Authenticate the user
            
        if user is not None:
            login(request, user)  # Log the user in
            
            return redirect('/')  # Redirect to the desired page after successful login
        else:
            return render(request, 'auth/signin.html',{'msg':'Incorrect Username or password'})
    else:
        # form = UserLoginForm()
        return render(request, 'auth/signin.html')

def profile(request):
    return render(request, 'auth/profile.html')

def logout_view(request):
    logout(request)
    # Redirect to home page
    return redirect('/')


def search_items(request):
    query = request.GET.get('q')
    if query:
        # Filter items based on the query (case-insensitive match on the 'title' field)
        results = Book.objects.filter(title__icontains=query)[:10]  # Limiting to 10 results
    else:
        # If no query is provided, return all items
        results = Book.objects.all()

    # Serialize the queryset to JSON
    serialized_results = [{'id': item.id, 'title': item.title, 'img': item.img} for item in results]

    # Return the JSON response as an array
    return JsonResponse(serialized_results, safe=False)

def book_info(request, book_id):
    # Retrieve the book based on book_id or return 404 if not found
    book = get_object_or_404(Book, id=book_id)
    
    context = {
        'book': book,
    }
    
    return render(request, 'partials/book_info.html', context)
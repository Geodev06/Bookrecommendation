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


#

import torch
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity
from torchtext.vocab import FastText


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


#FASTTEXT MODEL
class BookRecommendationModel:
    def __init__(self, embeddings_path, data_path, max_keywords=10):
        self.book_embeddings = np.load(embeddings_path)
        self.book_data = pd.read_csv(data_path)
        self.fasttext = FastText(language='en')  # Load FastText embeddings
        self.embedding_dim = self.fasttext.dim
        self.user_search_history = {}  # Dictionary to store user search history with keyword counts
        self.max_keywords = max_keywords  # Maximum number of keywords to consider in search history

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    def get_average_embedding(self, tokens):
        embeddings = []
        for token in tokens:
            if token in self.fasttext.stoi:
                embeddings.append(self.fasttext.vectors[self.fasttext.stoi[token]])
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)

    def update_search_history(self, search_query):
        preprocessed_query = self.preprocess_text(search_query)
        for keyword in preprocessed_query:
            if keyword in self.user_search_history:
                self.user_search_history[keyword] += 1
            else:
                self.user_search_history[keyword] = 1
        
        # Remove older keywords if the total count exceeds the maximum limit
        if len(self.user_search_history) > self.max_keywords:
            sorted_keywords = sorted(self.user_search_history.items(), key=lambda x: x[1], reverse=True)
            self.user_search_history = dict(sorted_keywords[:self.max_keywords])

    def recommend_books(self, query, k=15):
        # Preprocess the query text
        preprocessed_query = self.preprocess_text(query)
        
        # Combine the preprocessed query with user search history keywords
        combined_query = preprocessed_query + list(self.user_search_history.keys())
        
        # Compute the average embedding for the combined query
        combined_embedding = self.get_average_embedding(combined_query)
        
        # Calculate cosine similarity between the combined query embedding and book embeddings
        similarities = cosine_similarity(combined_embedding.reshape(1, -1), self.book_embeddings).flatten()
        
        # Get the indices of the top k books based on similarity scores
        top_indices = similarities.argsort()[::-1][:k]
        
        # Create a list of recommended books as dictionaries
        recommendations = [
            {
                'title': self.book_data.iloc[i]['title'], 
                'desc': self.book_data.iloc[i]['desc'], 
                'isbn': self.book_data.iloc[i]['isbn'], 
                'similarity_score': similarities[i]
            } 
            for i in top_indices
        ]
        
        return recommendations


def book_info(request, book_id):
     # Retrieve the book object based on book_id
    book = get_object_or_404(Book, id=book_id)
    
    embeddings_path = 'bookrecommendation/static/book_recommendation_embeddings.npy'
    data_path = 'bookrecommendation/static/book_recommendation_data.csv'

    # Initialize BookRecommendationModel
    recommendation_model = BookRecommendationModel(embeddings_path=embeddings_path, data_path=data_path, max_keywords=15)
    
  # Search history
    recommendation_model.update_search_history("Comedy, Batman Joker Spiderman")
    # Generate book recommendations based on book description
    recommendations = recommendation_model.recommend_books(query=book.desc, k=15)
    
    # Retrieve detailed information for each recommended book
    detailed_recommendations = []
    for recommendation in recommendations:
        isbn = recommendation['isbn']  # Assuming 'isbn' is the key for ISBN in the recommendation
        
        # Check if the recommended ISBN is not the same as the current book's ISBN
        if isbn != book.isbn:
            recommended_book = Book.objects.filter(isbn=isbn).first()
            if recommended_book:
                detailed_recommendations.append(recommended_book)
    
    # Render the book information template with the book object and detailed recommendations
    return render(request, 'partials/book_info.html', {'book': book, 'recommendations': detailed_recommendations})


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


#BERT
import re
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

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

class BookRecommendationModel:
    def __init__(self, model_path, embeddings_path, data_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.load(model_path, embeddings_path, data_path)
        self.history_keywords = {}

    def load(self, model_path, embeddings_path, data_path):
        # Load the model state dict from file
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.book_embeddings = np.load(embeddings_path)
        self.book_data = pd.read_csv(data_path)

    def preprocess_text(self, text):
        # Implement text preprocessing if needed
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.lower()

    def extract_keywords(self, text):
        # Extract keywords from text
        tokens = self.tokenizer.tokenize(text)
        keywords = set()
        for token in tokens:
            if token in self.tokenizer.vocab:
                keywords.add(token)
        return keywords

    def recommend_books(self, query, user_history, k=10):
        # Tokenize the query
        tokenized_query = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)

        # Get embeddings for the query
        with torch.no_grad():
            outputs = self.model(**tokenized_query)
            query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Calculate cosine similarity between the query and book embeddings
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.book_embeddings).flatten()

        # Update history_keywords with keywords from the user's history
        for book in user_history:
            normalized_book = book.strip().lower()  # Normalize book title
            if normalized_book not in self.history_keywords:
                book_description = self.book_data[self.book_data['title'].str.lower() == normalized_book]['desc'].iloc[0]
                self.history_keywords[normalized_book] = self.extract_keywords(book_description)

        # Calculate historical similarities based on keywords
        history_similarities = np.zeros(len(self.book_embeddings))
        for book in user_history:
            if book.strip().lower() in self.history_keywords:
                book_keywords = self.history_keywords[book.strip().lower()]
                for idx, desc in enumerate(self.book_data['desc']):
                    if any(keyword.lower() in desc.lower() for keyword in book_keywords):
                        history_similarities[idx] += 1

        # Normalize history similarities
        if len(user_history) > 0:
            history_similarities /= len(user_history)

        # Combine similarities from query and user history
        combined_similarities = 0.5 * similarities + 0.5 * history_similarities

        # Get indices of top k similar books
        top_indices = combined_similarities.argsort()[-k:][::-1]

        # Extract information for top k books
        recommendations = []
        for index in top_indices:
            title = self.book_data.iloc[index]['title']
            description = self.book_data.iloc[index]['desc']
            isbn = self.book_data.iloc[index]['isbn']
            similarity = combined_similarities[index]
            recommendations.append((title, description, similarity,isbn))

        return recommendations

def book_info(request, book_id):
    # Retrieve the book based on book_id or return 404 if not found
    book = get_object_or_404(Book, id=book_id)
    
    # Initialize BookRecommendationModel
    # Example usage
    model_path = "bookrecommendation/static/book_recommendation_model.pth"
    embeddings_path = "bookrecommendation/static/book_recommendation_embeddings.npy"
    data_path = "bookrecommendation/static/book_recommendation_data.csv"
    recommendation_model = BookRecommendationModel(model_path, embeddings_path, data_path)
    
    # Get user history (assuming it's a list of book titles)
    user_history = ['Wild Indiana']  # Replace with actual user history
    
    # Generate book recommendations based on the current book's description and user history
    recommendations = recommendation_model.recommend_books(book.desc, user_history)

    # Convert each tuple to a dictionary
    formatted_recommendations = [
        {
            'title': title,
            'desc': desc,
            'similarity': similarity,
            'isbn': isbn
        }
        for title, desc, similarity, isbn in recommendations
    ]
    # List to store book info
    recommended_books = []

    # Fetch book info for each ISBN
    for reco in formatted_recommendations:
        isbn = reco['isbn']
        try:
            cur = Book.objects.get(isbn=isbn)
            book_info = {
                'title': cur.title,
                'author': cur.author,
                'desc': cur.desc,
                'isbn': cur.isbn,
                'img': cur.img,  # Assuming your Book model has an 'img' field
                'id': cur.id,    # Assuming your Book model has an 'id' field
                'link': cur.link # Assuming your Book model has a 'link' field
                # Add other fields as needed
            }
            recommended_books.append(book_info)
        except Book.DoesNotExist:
            print(f"No book found for ISBN: {isbn}")

    context = {
        'book': book,  # Assuming you have a 'book' object you want to display details for
        'recommendations': recommended_books  # Pass recommendations list to the template
    }
    # print(recommendations)
    return render(request, 'partials/book_info.html', context)
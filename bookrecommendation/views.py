from django.shortcuts import render, redirect, get_object_or_404
import os
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from .models import Book, UserBook, Payment, CustomUser, History
from django.db.models import Avg, Count
from .forms import UserRegistrationForm
from django.http import JsonResponse
import stripe
from django.urls import reverse
from django.conf import settings

from django.core.serializers import serialize

#
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torchtext.vocab import FastText

class BookRecommendationModel:
    def __init__(self, embeddings_path, data_path, max_keywords=10):
        self.book_embeddings = np.load(embeddings_path)
        self.book_data = pd.read_csv(data_path)
        self.fasttext = FastText(language='en')
        self.embedding_dim = self.fasttext.dim
        self.user_search_history = {}
        self.max_keywords = max_keywords

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    def get_average_embedding(self, tokens):
        embeddings = [self.fasttext.vectors[self.fasttext.stoi[token]] for token in tokens if token in self.fasttext.stoi]
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(self.embedding_dim)

    def update_search_history(self, search_query):
        preprocessed_query = self.preprocess_text(search_query)
        for keyword in preprocessed_query:
            self.user_search_history[keyword] = self.user_search_history.get(keyword, 0) + 1

        if len(self.user_search_history) > self.max_keywords:
            sorted_keywords = sorted(self.user_search_history.items(), key=lambda x: x[1], reverse=True)[:self.max_keywords]
            self.user_search_history = dict(sorted_keywords)

    def recommend_books(self, query, k=15):
        preprocessed_query = self.preprocess_text(query)
        combined_query = preprocessed_query + list(self.user_search_history.keys())
        combined_embedding = self.get_average_embedding(combined_query)
        
        similarities = cosine_similarity(combined_embedding.reshape(1, -1), self.book_embeddings).flatten()
        
        top_indices = similarities.argsort()[::-1][:k]
        
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




# Set your Stripe API key
stripe.api_key = settings.STRIPE_SECRET_KEY

# Assuming your Django app name is 'yourapp'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def index(request):
    random_books = list(Book.objects.order_by('?')[:20])

    # Fetch top 10 books with average rating and total reviews pre-calculated
    top_books = Book.objects.annotate(
        avg_rating=Avg('rating'),
        total_reviews=Count('reviews')
    ).order_by('-avg_rating', '-total_reviews')[:6]

    popular_books = Book.objects.order_by('-totalratings')[:10]


    return render(request, 'index.html', {'selected_books': random_books, 'top_books': top_books,'popular_books':popular_books})



def get_recommendation_index(request):

    recommended_books = []

    if request.user.is_authenticated:
        embeddings_path = 'bookrecommendation/static/book_recommendation_embeddings.npy'
        data_path = 'bookrecommendation/static/book_recommendation_data.csv'

        user_id = request.user.id
        # Step 1: Retrieve the latest timestamps for the user's books
        latest_timestamps = list(UserBook.objects.filter(user_id=user_id).order_by('-created_at')[:3].values_list('created_at', flat=True))

        # Step 2: Use the latest timestamps to fetch the corresponding book IDs
        my_book_ids = UserBook.objects.filter(user_id=user_id, created_at__in=latest_timestamps).values_list('book_id', flat=True)


        latest_timestamps2 = list(History.objects.filter(user_id=user_id).order_by('-created_at')[:3].values_list('created_at', flat=True))
        last_3_title_history = History.objects.filter(user_id=user_id, created_at__in=latest_timestamps2).values_list('book_id', flat=True)

        # Fetch book titles based on user's book IDs
        book_titles = Book.objects.filter(id__in=my_book_ids).values_list('title', flat=True)

        history_book_titles =  Book.objects.filter(id__in=last_3_title_history).values_list('title', flat=True)

        all_book_titles = list(book_titles) + list(history_book_titles)

        print(all_book_titles)
        # Concatenate all titles into a single string
        concatenated_titles = ', '.join(all_book_titles)
        # Print the concatenated titles

        # Initialize BookRecommendationModel
        recommendation_model = BookRecommendationModel(embeddings_path=embeddings_path, data_path=data_path, max_keywords=20)
        recommendation_model.update_search_history(concatenated_titles)

        recommendations = recommendation_model.recommend_books(query='', k=15)

        my_books = Book.objects.filter(id__in=my_book_ids)

        # Fetch recommended books based on recommendations' ISBN
        isbn_list = [rec.get('isbn') for rec in recommendations]

        # Exclude books with ISBNs that exist in my_books
        recommended_books = Book.objects.filter(isbn__in=isbn_list).exclude(id__in=my_books.values_list('id', flat=True))
    
    recommended_books2 = serialize('json', recommended_books)

    return JsonResponse(recommended_books2,safe=False)


def get_users(request):
    # Fetch all users from CustomUser model
    users = CustomUser.objects.filter(is_superuser=0)

    # Serialize user data
    user_data = serialize('json', users)

    # Return serialized data as JSON response
    return JsonResponse(user_data, safe=False)

def get_user_payments(request):

  # Initialize an empty list to store the combined data
    data = []

    # Retrieve all payments from the Payment model
    payments = Payment.objects.all()

    # Retrieve all users from the CustomUser model
    users = CustomUser.objects.all()

    # Retrieve all books from the Book model
    books = Book.objects.all()

    # Loop through each payment
    for payment in payments:
        # Retrieve the user associated with the payment
        user = users.filter(id=payment.user_id).first()

        # Retrieve the book associated with the payment
        book = books.filter(id=payment.book_id).first()

        # Construct a dictionary with the combined data
        payment_data = {
            'payment_id': payment.id,
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'book_id': book.id,
            'book_title': book.title,
            'price': payment.price,
            'created_at' : payment.created_at
        }

        # Append the dictionary to the data list
        data.append(payment_data)

    # Return the combined data as a JSON response
    return JsonResponse(data, safe=False)

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
            
        if user is not None and user.is_active == 1:
            login(request, user)  # Log the user in
            
            return redirect('/')  # Redirect to the desired page after successful login
        else:
            return render(request, 'auth/signin.html',{'msg':'Incorrect Username or password'})
    else:
        # form = UserLoginForm()
        return render(request, 'auth/signin.html')

def profile(request):
    user_id = request.user.id
    my_book_ids = UserBook.objects.filter(user_id=user_id)

    # Iterate over each UserBook object in my_book_ids
    my_books = []
    for user_book in my_book_ids:
        book_data = Book.objects.get(id=user_book.book_id)  # Assuming UserBook has a field named book_id
        my_books.append(book_data)

    my_books.reverse()

    superuser_books = Book.objects.filter(status='Y') # GET ALL ACTIVE BOOKS ONLY

    return render(request, 'auth/profile.html', {'my_books': my_books, 'superuser_books': superuser_books})

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
     # Retrieve the book object based on book_id
    book = get_object_or_404(Book, id=book_id)
    book_exists = UserBook.objects.filter(book_id=book_id, user_id=request.user.id).exists()

    user_id = request.user.id

    # Retrieve all records matching the criteria
    records_to_delete = History.objects.filter(user_id=user_id, book_id=book_id)

    # Delete each record
    for record in records_to_delete:
        record.delete()

    #insert new
    History.objects.create(
        book_id=book_id,
        user_id=user_id
    )

    embeddings_path = 'bookrecommendation/static/book_recommendation_embeddings.npy'
    data_path = 'bookrecommendation/static/book_recommendation_data.csv'

    # Initialize BookRecommendationModel
    recommendation_model = BookRecommendationModel(embeddings_path=embeddings_path, data_path=data_path, max_keywords=15)
    
#   # Search history
#     recommendation_model.update_search_history("Comedy, Batman Joker Spiderman")
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
    
    # Create a Stripe Checkout session
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[
            {
                'price_data': {
                    'currency': 'php',
                    'product_data': {
                        'name': book.title,
                        'images': [book.img],  # Replace with the URL of the book image
                    },
                    'unit_amount': int(book.price * 100),  # Stripe requires amount in cents
                },
                'quantity': 1,
            },
        ],
        mode='payment',
        success_url=request.build_absolute_uri(reverse('success_payment', args=[book_id])),  # Replace 'success_url' with your actual success URL
        cancel_url = request.build_absolute_uri(reverse('book_info', args=[book_id])) # Replace 'cancel_url' with your actual cancel URL
    )
    
    # Pass the session ID to the template
    context = {
        'session_id': session.id,
        'book': {
            'id': book.id,
            'title': book.title,
            'author': book.author,
            'img': book.img,
            'price': book.price
        }
    }

    # Render the book information template with the book object and detailed recommendations
    return render(request, 'partials/book_info.html', {'book': book,'book_exists':book_exists, 'recommendations': detailed_recommendations, 'context':context })

def success_payment(request, book_id):

    book = get_object_or_404(Book, id=book_id)
    user_id = request.user.id
     # Create a new Payment object
    payment = Payment.objects.create(
        book_id=book_id,
        user_id=user_id,
        price=book.price
    )
    
    # Save the Payment object to the database
    payment.save()

    user_book = UserBook.objects.create(
        book_id=book_id,
        user_id=user_id
    )

    user_book.save()

    return redirect('/profile')  


def togglestatus(request, user_id):

    try:
        user = CustomUser.objects.get(pk=user_id)
        
        user.is_active = not user.is_active
        user.save()
        user_data = serialize('json', [user])
        return JsonResponse(user_data, safe=False)
    except CustomUser.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)

def addbook(request, book_id):
    book = Book.objects.get(id=book_id)
    book.status = 'Y' if book.status == 'N' else 'N'  # Toggle status between 'Y' and 'N'
    book.save()

    return redirect('book_info',book_id)
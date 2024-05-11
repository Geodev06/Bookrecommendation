from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser
from decimal import Decimal
import random

class Book(models.Model):
    author = models.TextField()
    bookformat = models.TextField()
    desc = models.TextField(null=True)
    genre = models.TextField(null=True)
    img = models.TextField(null=True)
    isbn = models.TextField(null=True)
    isbn13 = models.TextField(null=True)
    link = models.TextField(null=True)
    pages = models.PositiveIntegerField(null=True)
    rating = models.FloatField(null=True)
    reviews = models.PositiveIntegerField(null=True)
    title = models.TextField()
    totalratings = models.PositiveIntegerField(null=True)
    price = models.DecimalField(max_digits=6, decimal_places=2, default=Decimal(random.randint(100000, 200000)/100))
    status = models.TextField(null=True)
    created_at = models.DateTimeField(default=timezone.now)  # Use timezone.now without parentheses

    class Meta:
        verbose_name = "Book"
        verbose_name_plural = "Books"

class CustomUser(AbstractUser):
    SEX_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    )
    
    sex = models.CharField(max_length=1, choices=SEX_CHOICES)
    
    # Add or change related_name for groups and user_permissions fields
    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_users',
        blank=True,
        verbose_name='groups',
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_users',
        blank=True,
        verbose_name='user permissions',
        help_text='Specific permissions for this user.',
    )

    def __str__(self):
        return self.username

class Payment(models.Model):
    user_id = models.PositiveIntegerField(null=False)
    book_id = models.PositiveIntegerField(null=False)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    created_at = models.DateTimeField(default=timezone.now)  # Use timezone.now without parentheses

    class Meta:
        verbose_name = "Payment"
        verbose_name_plural = "Payments"


class UserBook(models.Model):
    user_id = models.PositiveIntegerField(null=False)
    book_id = models.PositiveIntegerField(null=False)
    created_at = models.DateTimeField(default=timezone.now)  # Use timezone.now without parentheses

    class Meta:
        verbose_name = "UserBook"
        verbose_name_plural = "UserBooks"


class History(models.Model):
    user_id = models.PositiveIntegerField(null=False)
    book_id = models.PositiveIntegerField(null=False)
    created_at = models.DateTimeField(default=timezone.now)  # Use timezone.now without parentheses

    class Meta:
        verbose_name = "Histories"
        verbose_name_plural = "Histories"
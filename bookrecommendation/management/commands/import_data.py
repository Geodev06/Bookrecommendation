import csv
import random
from django.core.management.base import BaseCommand
from bookrecommendation.models import Book

class Command(BaseCommand):
    help = 'Import data from CSV file to MySQL'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str)

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        
        batch_size = 1000  # Adjust batch size as needed
        
        with open(csv_file, 'r', encoding='utf-8') as f:  # Specify the encoding
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            
            batch = []
            for row in reader:
                # Generate random price between 1000 and 2500 with 2 decimal places
                price = round(random.uniform(1000, 2500), 2)
                
                # Assuming the CSV columns are in the same order as model fields
                obj = Book(
                    author=row[0], 
                    bookformat=row[1], 
                    desc=row[2], 
                    genre=row[3], 
                    img=row[4], 
                    isbn=row[5], 
                    isbn13=row[6], 
                    link=row[7], 
                    pages=row[8], 
                    rating=row[9], 
                    reviews=row[10], 
                    title=row[11], 
                    price=price,  # Set the random price here
                    totalratings=row[12],
                    status='N'
                )
                batch.append(obj)
                
                # Insert batch into the database when it reaches the batch size
                if len(batch) >= batch_size:
                    Book.objects.bulk_create(batch)
                    batch = []
            
            # Insert any remaining objects
            if batch:
                Book.objects.bulk_create(batch)

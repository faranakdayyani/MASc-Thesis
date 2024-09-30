import os
import csv
import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.retry import Retry

# Initialize Firebase Admin
root = '/home/ali/PycharmProjects/si-exploratory'
file = 'cbdstudy-47927-firebase-adminsdk-4beij-8381d7534e.json'
if not firebase_admin._apps:
    cred = credentials.Certificate(os.path.join(root, file))
    firebase_admin.initialize_app(cred)

# Access Firestore
db = firestore.client()


def save_document_fields_to_csv(collection_path, document_id, data):
    # Ensure the collection_path directory exists
    os.makedirs(collection_path, exist_ok=True)
    csv_file_path = os.path.join(collection_path, f'{document_id}.csv')

    # Writing data to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)


def fetch_and_save_collection_data(collection_ref, parent_path=''):
    # Iterate through all documents in the collection

    # docs = collection_ref.stream()
    docs = collection_ref.stream(retry=Retry())

    for doc in docs:
        document_data = doc.to_dict()
        document_id = doc.id

        print(doc.id)

        document_path = os.path.join(parent_path, collection_ref.id)
        # Save the document data to a CSV file
        save_document_fields_to_csv(document_path, document_id, document_data)

        # Check for subcollections
        for subcollection in doc.reference.collections():
            fetch_and_save_collection_data(subcollection, os.path.join(document_path, document_id))




def main():

    parent_path = '/data/p08/raw'

    # List all collections (top-level)
    collections = db.collections()
    for collection in collections:

        if collection.id == 'participantuoft07@gmail.com':
            print(collection.id)
            fetch_and_save_collection_data(collection, parent_path=parent_path)


if __name__ == '__main__':
    main()

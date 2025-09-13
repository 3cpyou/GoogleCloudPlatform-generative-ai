import json

from google.api_core.exceptions import NotFound
from google.cloud import firestore

# --- Configuration ---
# The name of your Firestore database, as seen in the screenshots.
DATABASE_NAME = "ad3gem-email"

# The list of collections to download from.
COLLECTION_NAMES = [
    "thread_summaries",
    "people_directory",
    "messages_full",
    "email_search_index",
]

# How many documents to fetch from each collection.
DOCUMENTS_PER_COLLECTION = 3

# The name of the output file.
OUTPUT_FILENAME = "firestore_export.json"


def download_firestore_data():
    """
    Connects to a specific Firestore database, fetches a limited number of
    documents from specified collections, and saves them to a single JSON file.
    """
    print("Starting Firestore data download...")

    # The library automatically uses credentials from your environment.
    # We specify the database ID when initializing the client.
    try:
        db = firestore.Client(database=DATABASE_NAME)
        print(f"Successfully connected to Firestore database: '{DATABASE_NAME}'")
    except Exception as e:
        print(f"Error: Could not connect to Firestore. {e}")
        print(
            "Please ensure you have authenticated via `gcloud auth application-default login`"
        )
        return

    # This dictionary will hold all the data from all collections.
    all_collections_data = {}

    # Loop through each collection name.
    for collection_name in COLLECTION_NAMES:
        print(
            f"\nFetching {DOCUMENTS_PER_COLLECTION} documents from '{collection_name}'..."
        )
        try:
            # Get a reference to the collection and query the first few documents.
            docs_stream = (
                db.collection(collection_name).limit(DOCUMENTS_PER_COLLECTION).stream()
            )

            documents_list = []
            for doc in docs_stream:
                # Convert each document to a dictionary and add its ID.
                doc_data = doc.to_dict()
                documents_list.append({"id": doc.id, "data": doc_data})

            if not documents_list:
                print(f"-> No documents found in '{collection_name}'.")
            else:
                print(f"-> Successfully fetched {len(documents_list)} documents.")

            # Add the list of documents to our main dictionary.
            all_collections_data[collection_name] = documents_list

        except NotFound:
            print(f"-> Collection '{collection_name}' not found. Skipping.")
        except Exception as e:
            print(f"-> An error occurred while fetching from '{collection_name}': {e}")

    # Write the combined data to a single JSON file.
    try:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            # Use default=str to handle non-serializable types like Timestamps.
            json.dump(
                all_collections_data, f, indent=4, ensure_ascii=False, default=str
            )
        print(f"\n✅ Success! All data has been saved to '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"\nError: Could not write data to file. {e}")


if __name__ == "__main__":
    download_firestore_data()

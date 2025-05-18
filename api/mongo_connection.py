from pymongo import MongoClient
from bson.binary import Binary
from bson import ObjectId
from PIL import Image
from io import BytesIO
import os
import streamlit as st

client = MongoClient(st.secrets["MONGO"]["MONGO_URI"], tls=True, tlsAllowInvalidCertificates=True)

def insert_image_data(filepath: str, database: str, collection: str) -> str:
    """
    Insert an image from a temp filepath into MongoDB.

    Args:
        filepath (str): The full temporary path to the image file.
        database (str): The name of the target MongoDB database.
        collection (str): The name of the target collection.

    Returns:
        str: Inserted document ID or error message.
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)

        with open(filepath, "rb") as image_file:
            image_data = image_file.read()

        data = {
            "filename": os.path.basename(filepath),
            "image_data": Binary(image_data)
        }

        result = col.insert_one(data)
        print(f"Inserted image data with ID: {result.inserted_id}")
        return str(result.inserted_id)

    except Exception as e:
        print("Error inserting image data:", e)
        return str(e)

def get_image_data(document_id: str, database: str, collection: str) -> bytes:
    """
    Retrieve image binary data from MongoDB using the document ID.

    Args:
        document_id (str): The ID of the document in MongoDB.
        database (str): The name of the target MongoDB database.
        collection (str): The name of the target collection.

    Returns:
        bytes: The binary image data if found, else None.
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)

        document = col.find_one({"_id": ObjectId(document_id)})

        if document and "image_data" in document:
            return bytes(document["image_data"])  # Ensure it's returned as raw bytes
        else:
            print("Document not found or no image_data field.")
            return None

    except Exception as e:
        print("Error retrieving image data:", e)
        return None
                 
def insert_yolo_data(document_id: str, yolo_data: dict, database: str, collection: str) -> str:
    """
    Inserts YOLO detection data (coordinates and optionally cropped images) into MongoDB.

    Args:
        document_id (str): The ID of the original document.
        yolo_data (dict): A dict containing 'coordinates' (list of [x1, y1, x2, y2]) and 'images' (list of PIL Images or image bytes).
        database (str): MongoDB database name.
        collection (str): MongoDB collection name to store detection data.

    Returns:
        str: Inserted document ID or error message.
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)

        coords = yolo_data['coordinates']
        cropped_images = yolo_data['images']

        # Convert each image to binary
        image_binaries = []
        for img in cropped_images:
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format="JPEG")  # or PNG depending on your needs
            image_binaries.append(Binary(buffer.getvalue()))

        data = {
            "image_id": document_id,
            "detections": [
                {"coords": coord, "cropped_image": img_bin}
                for coord, img_bin in zip(coords, image_binaries)
            ]
        }

        result = col.insert_one(data)
        print(f"Inserted YOLO data with ID: {result.inserted_id}")
        return str(result.inserted_id)

    except Exception as e:
        print("Error inserting YOLO data:", e)
        return str(e)

def extract_cropped_images(document_id: str, database: str, collection: str):
    """
    Extracts cropped YOLO images from a specific MongoDB document.

    Args:
        document_id (str): The MongoDB document ID (as a string).
        database (str): Name of the MongoDB database.
        collection (str): Name of the collection containing YOLO detection data.

    Returns:
        List[Image.Image]: A list of PIL images (cropped detections), or empty list on error.
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)

        doc = col.find_one({"_id": ObjectId(document_id)})

        if not doc:
            print(f"No document found with ID: {document_id}")
            return []

        cropped_images = []
        for detection in doc.get("detections", []):
            cropped_bin = detection.get("cropped_image")
            if cropped_bin:
                img = Image.open(BytesIO(cropped_bin))
                cropped_images.append(img)

        return cropped_images

    except Exception as e:
        print("Error extracting cropped images:", e)
        return None
      
def insert_category(document_id: str, categories: list, database: str, collection: str) -> bool:
    """
    Inserts predicted categories into the corresponding YOLO detection document.

    Args:
        document_id (str): MongoDB document ID as a string.
        categories (list): List of predicted categories (str), one per detection.
        database (str): MongoDB database name.
        collection (str): MongoDB collection name.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)

        # Fetch document to ensure it exists and to count detections
        doc = col.find_one({"_id": ObjectId(document_id)})
        if not doc:
            print(f"No document found with ID: {document_id}")
            return False

        detections = doc.get("detections", [])
        if len(categories) != len(detections):
            print("Mismatch between number of categories and detections.")
            return False

        # Add 'category' to each detection
        for i, category in enumerate(categories):
            detections[i]["category"] = category

        # Update the document
        result = col.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"detections": detections}}
        )

        print(f"Updated document {document_id} with categories.")
        return result.modified_count > 0

    except Exception as e:
        print("Error inserting categories:", e)
        return False
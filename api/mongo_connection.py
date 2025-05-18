from pymongo import MongoClient
from bson.binary import Binary
from bson import ObjectId
from PIL import Image
from io import BytesIO
import os
import streamlit as st
import cv2
import numpy as np
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
                 
def insert_yolo_data(document_id: str, yolo_data: list, database: str, collection: str) -> str:
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
        data = yolo_data[0]
        posicion_por_rect = yolo_data[2]
        espacios_vacios = yolo_data[4]

        db = client.get_database(database)
        col = db.get_collection(collection)
        object_id = ObjectId(document_id)

        yolo_output = []
        for box, cropped_img in zip(data['boxes'], data['images']):
            x1, y1, x2, y2 = map(int, box)
            box_tuple = (x1, y1, x2, y2)
            _, buffer = cv2.imencode('.jpg', cropped_img)
            image_binary = Binary(buffer.tobytes())
            position = posicion_por_rect.get(box_tuple, (None, None))
            yolo_output.append({
                "box_n": [x1, y1, x2, y2],
                "image_n": image_binary,
                "position": {(position[0], position[1])},
                "empty": False
            })
       
        for box_coords, box_id in espacios_vacios.items():
            x1, y1, x2, y2 = map(int, box_coords)
            yolo_output.append({
                "box_n": [x1, y1, x2, y2],
                "box_id": int(box_id),
                "empty": True
            })

        result = col.update_one(
            {"_id": object_id},
            {"$set": {"YOLO_output": yolo_output}}
        )
  
    except Exception as e:
        print("Error inserting YOLO data:", e)
        return str(e)


def get_yolo_data(document_id: str, database: str, collection: str):
    """
    Retrieve YOLO_output from MongoDB and return each crop as JPG bytes.

    Returns:
        List[dict]: Each dict has:
            - box_n (list[int])
            - empty (bool)
            - If empty is False:
                - image_jpg (bytes)  ← your JPEG data here
                - position (dict)
            - If empty is True:
                - box_id (int)
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)
        obj_id = ObjectId(document_id)

        doc = col.find_one({"_id": obj_id}, {"YOLO_output": 1})
        if not doc or "YOLO_output" not in doc:
            return []

        output = []
        for item in doc["YOLO_output"]:
            box = item["box_n"]
            empty = item.get('empty')
            if not empty:
                # Decode to BGR OpenCV image
                arr = np.frombuffer(item["image_n"], dtype=np.uint8)
                cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                # Re-encode as JPG
                _, buf = cv2.imencode('.jpg', cv_img)
                jpg_bytes = buf.tobytes()

                output.append({
                    "box_n": box,
                    "image_jpg": jpg_bytes,
                    "position": item.get("position"),
                    "empty": empty
                })
            else:
                output.append({
                    "box_n": box,
                    "box_id": item.get("box_id"),
                    "empty": empty
                })

        return output

    except Exception as e:
        print("Error retrieving YOLO data:", e)
        return []   

        
         

def insert_category(document_id: str,
                    categories_map: dict,
                    database: str,
                    collection: str) -> bool:
    """
    Inserts predicted categories into the YOLO_output array of a MongoDB document.

    Args:
        document_id (str): MongoDB document ID as a string.
        categories_map (dict): Mapping from box coords to category, e.g.
                               {
                                   (x1, y1, x2, y2): "cat_a",
                                   (x3, y3, x4, y4): "cat_b",
                                   ...
                               }
        CATEGORIES_MAP IS SUPPOSED TO RECEIVE THE RETURN STATEMENT
        FROM cloudflare_llavahf()...
        
        database (str): MongoDB database name.
        collection (str): MongoDB collection name.

    Returns:
        bool: True if all updates succeeded (at least one element updated per box), False otherwise.
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)
        obj_id = ObjectId(document_id)

        success = True
        for box_coords, category in categories_map.items():
            # ensure box_n matches the stored list
            coords_list = list(box_coords)

            res = col.update_one(
                {
                    "_id": obj_id,
                    "YOLO_output.box_n": coords_list
                },
                {
                    "$set": {
                        "YOLO_output.$.category": category
                    }
                }
            )
            if res.modified_count == 0:
                # no matching element found for these coords
                print(f"⚠️  No match for box {coords_list}")
                success = False

        return success

    except Exception as e:
        print("Error inserting categories:", e)
        return False

    
DATABASE = "files_hackathon"
COLLECTION = "anaquel_estante"
print(get_yolo_data("682a4a9fdd777bc1bbff97e6", DATABASE, COLLECTION))

def get_all_data(document_id: str, database: str, collection: str):
    """
    Retrieve YOLO_output from MongoDB and return each crop as JPG bytes.

    Returns:
        List[dict]: Each dict has:
            - box_n (list[int])
            - empty (bool)
            - If empty is False:
                - image_jpg (bytes)  ← your JPEG data here
                - position (dict)
            - If empty is True:
                - box_id (int)
    """
    try:
        db = client.get_database(database)
        col = db.get_collection(collection)
        obj_id = ObjectId(document_id)

        doc = col.find_one({"_id": obj_id}, {"YOLO_output": 1})
        if not doc or "YOLO_output" not in doc:
            return []

        output = []
        for item in doc["YOLO_output"]:
            box = item["box_n"]
            empty = item.get('empty')
            if not empty:
                # Decode to BGR OpenCV image
                arr = np.frombuffer(item["image_n"], dtype=np.uint8)
                cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                # Re-encode as JPG
                _, buf = cv2.imencode('.jpg', cv_img)
                jpg_bytes = buf.tobytes()

                output.append({
                    "box_n": box,
                    "image_jpg": jpg_bytes,
                    "position": item.get("position"),
                    "empty": empty

                })
            else:
                output.append({
                    "box_n": box,
                    "box_id": item.get("box_id"),
                    "empty": empty
                })

        return output

    except Exception as e:
        print("Error retrieving YOLO data:", e)
        return []   
from flask import Flask, request, jsonify, send_file
from firebase_admin import credentials, initialize_app, firestore, storage
from google.cloud import vision
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io
import mimetypes
from datetime import datetime
import json # Import json module
from flask_cors import CORS # Import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize CORS with your Flask app.
# For production, replace "*" with your deployed frontend's URL (e.g., "https://your-frontend-app.onrender.com")
CORS(app)

# --- Firebase Initialization ---
# Prioritize loading credentials from an environment variable (for Render deployment)
FIREBASE_SERVICE_ACCOUNT_KEY_JSON = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY_JSON') # Renamed for clarity

if FIREBASE_SERVICE_ACCOUNT_KEY_JSON:
    try:
        # Parse the JSON string from the environment variable
        cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_KEY_JSON)
        cred = credentials.Certificate(cred_dict)
        print("Firebase credentials loaded from environment variable.")
    except json.JSONDecodeError as e:
        print(f"Error decoding FIREBASE_SERVICE_ACCOUNT_KEY_JSON: {e}. Ensure it's valid JSON.")
        exit(1) # Exit if the JSON is malformed
else:
    # Fallback for local development using a file path
    FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
    if FIREBASE_CREDENTIALS_PATH and os.path.exists(FIREBASE_CREDENTIALS_PATH):
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        print("Firebase credentials loaded from local file path.")
    else:
        # Last resort: Attempt Application Default Credentials (for Google Cloud environments)
        print("FIREBASE_CREDENTIALS_PATH or FIREBASE_SERVICE_ACCOUNT_KEY_JSON not found. Attempting Application Default Credentials.")
        try:
            cred = credentials.ApplicationDefault()
            print("Firebase credentials loaded using Application Default Credentials.")
        except Exception as e:
            print(f"Failed to get ApplicationDefault credentials: {e}")
            print("Please ensure GOOGLE_APPLICATION_CREDENTIALS is set or Firebase credentials are provided via environment variable/file path.")
            exit(1) # Exit if no credentials can be found

firebase_app = initialize_app(cred, {
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
})

db = firestore.client()
bucket = storage.bucket()

# --- Google Cloud Vision Initialization ---
# The Vision client will automatically use GOOGLE_APPLICATION_CREDENTIALS
# or other standard authentication methods if deployed on Google Cloud.
# For local development, ensure GOOGLE_APPLICATION_CREDENTIALS is set
# to your service account key file.
vision_client = vision.ImageAnnotatorClient()

# --- Gemini API Configuration ---
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-pro') # Using gemini-pro for text tasks

# Define allowed document categories for Gemini
DOCUMENT_CATEGORIES = [
    'Lab Results', 'Prescriptions', 'Radiology', 'Discharge Summaries',
    'Vital Signs', 'Insurance', 'Consultation Notes', 'Other'
]

# Helper function to get user ID
# In a real application, this would involve verifying a Firebase ID token
# sent from the frontend (e.g., in an Authorization: Bearer header).
# For simplicity, this example assumes a 'X-User-Id' header is sent.
def get_user_id_from_request():
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        # Fallback for local testing or if header is missing
        print("Warning: X-User-Id header not found. Using 'anonymous_user'. For production, implement proper token verification.")
        user_id = "anonymous_user"
    return user_id

# Helper function to get app ID
def get_app_id():
    return os.getenv('APP_ID', 'default-app-id')

# --- Backend Endpoints ---

@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Handles document upload, stores original in Firebase Storage,
    performs OCR/text extraction, processes with Gemini for categorization and correction,
    and stores metadata/digital copy in Firestore.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    user_id = get_user_id_from_request()
    app_id = get_app_id()

    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    # Create a unique filename for storage
    unique_filename = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}{file_extension}"
    blob_path = f"artifacts/{app_id}/users/{user_id}/original_documents/{unique_filename}"

    try:
        # Ensure file pointer is at the beginning before reading for upload
        file.seek(0)
        # Upload original file to Firebase Storage
        blob = bucket.blob(blob_path)
        blob.upload_from_file(file, content_type=file.content_type)
        original_file_url = blob.public_url # Public URL for direct access (consider signed URLs for private files)

        extracted_text = ""
        digital_copy_content = ""
        category = "Other" # Default category

        # Determine file type and perform OCR/text extraction
        mime_type = file.content_type
        file.seek(0) # Reset file pointer again for text extraction if needed

        if mime_type.startswith('image/') or mime_type == 'application/pdf':
            # Use Google Cloud Vision for OCR
            image_content = file.read()
            vision_image = vision.Image(content=image_content)
            response = vision_client.document_text_detection(image=vision_image)
            extracted_text = response.full_text_annotation.text if response.full_text_annotation else ""
            print(f"OCR Extracted Text (first 200 chars): {extracted_text[:200]}...")

            # Process extracted text with Gemini for formatting and categorization
            if extracted_text:
                gemini_prompt = f"""
                Analyze the following medical document text.
                1. Correct any obvious typos or formatting issues to make it highly readable.
                2. Extract key information if applicable (e.g., patient name, date, test results).
                3. Categorize the document into one of these exact categories: {', '.join(DOCUMENT_CATEGORIES)}. If none fit, use 'Other'.
                
                Respond ONLY in a JSON format with two keys:
                "processed_text": "The cleaned and extracted text.",
                "category": "The determined category from the list."

                Example:
                {{
                  "processed_text": "Patient: John Doe\\nDate: 2023-01-15\\nBlood Test Results: Normal",
                  "category": "Lab Results"
                }}

                Document Text:
                {extracted_text}
                """
                try:
                    gemini_response = gemini_model.generate_content(gemini_prompt)
                    gemini_output = gemini_response.text.strip()
                    print(f"Gemini Raw Output: {gemini_output}")

                    # Attempt to parse JSON. Gemini might wrap it in markdown.
                    if gemini_output.startswith('```json') and gemini_output.endswith('```'):
                        gemini_output = gemini_output[7:-3].strip()

                    parsed_gemini = json.loads(gemini_output)
                    digital_copy_content = parsed_gemini.get('processed_text', extracted_text)
                    determined_category = parsed_gemini.get('category', 'Other')

                    # Validate category against predefined list
                    if determined_category in DOCUMENT_CATEGORIES:
                        category = determined_category
                    else:
                        category = "Other" # Fallback if Gemini suggests an invalid category

                except Exception as gemini_err:
                    print(f"Error processing with Gemini: {gemini_err}")
                    digital_copy_content = extracted_text # Fallback to raw OCR text
                    category = "Other" # Fallback category
            else:
                digital_copy_content = "" # No text extracted
                category = "Other"

        elif mime_type == 'text/plain':
            extracted_text = file.read().decode('utf-8')
            # Process text with Gemini for formatting and categorization
            if extracted_text:
                gemini_prompt = f"""
                Analyze the following medical document text.
                1. Correct any obvious typos or formatting issues to make it highly readable.
                2. Extract key information if applicable.
                3. Categorize the document into one of these exact categories: {', '.join(DOCUMENT_CATEGORIES)}. If none fit, use 'Other'.
                
                Respond ONLY in a JSON format with two keys:
                "processed_text": "The cleaned and extracted text.",
                "category": "The determined category from the list."

                Document Text:
                {extracted_text}
                """
                try:
                    gemini_response = gemini_model.generate_content(gemini_prompt)
                    gemini_output = gemini_response.text.strip()
                    if gemini_output.startswith('```json') and gemini_output.endswith('```'):
                        gemini_output = gemini_output[7:-3].strip()
                    parsed_gemini = json.loads(gemini_output)
                    digital_copy_content = parsed_gemini.get('processed_text', extracted_text)
                    determined_category = parsed_gemini.get('category', 'Other')

                    if determined_category in DOCUMENT_CATEGORIES:
                        category = determined_category
                    else:
                        category = "Other"
                except Exception as gemini_err:
                    print(f"Error processing with Gemini: {gemini_err}")
                    digital_copy_content = extracted_text
                    category = "Other"
            else:
                digital_copy_content = ""
                category = "Other"
        else:
            # For other file types, just store the original and categorize as 'Other'
            extracted_text = ""
            digital_copy_content = ""
            category = "Other"

        # Store document metadata in Firestore
        doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').add({
            'name': original_filename,
            'type': mime_type,
            'original_url': original_file_url,
            'digital_copy_content': digital_copy_content, # Storing as text
            'category': category,
            'size': file.content_length,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return jsonify({
            'message': 'Document uploaded and processed successfully',
            'documentId': doc_ref[1].id,
            'original_url': original_file_url,
            'digital_copy_content': digital_copy_content,
            'category': category
        }), 201

    except Exception as e:
        print(f"Error during document upload/processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """
    Retrieves all document metadata for the current user from Firestore.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    docs_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents')
    documents = []
    try:
        # Fetch documents and sort by timestamp in descending order
        # Note: Firestore's orderBy requires an index for multiple fields.
        # For simplicity, fetching all and sorting in memory.
        # For large datasets, consider proper Firestore indexing and pagination.
        for doc_snapshot in docs_ref.stream():
            doc_data = doc_snapshot.to_dict()
            documents.append({
                'id': doc_snapshot.id,
                'name': doc_data.get('name'),
                'type': doc_data.get('type'),
                'original_url': doc_data.get('original_url'),
                'digital_copy_content': doc_data.get('digital_copy_content', ''),
                'category': doc_data.get('category'),
                'size': doc_data.get('size'),
                'timestamp': doc_data.get('timestamp').isoformat() if doc_data.get('timestamp') else None
            })
        # Sort by timestamp (most recent first)
        documents.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '', reverse=True)
        return jsonify(documents), 200
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<document_id>/download/original', methods=['GET'])
def download_original_document(document_id):
    """
    Downloads the original document file from Firebase Storage.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').document(document_id)
    try:
        doc_snapshot = doc_ref.get()
        if not doc_snapshot.exists:
            return jsonify({'error': 'Document not found'}), 404

        doc_data = doc_snapshot.to_dict()
        original_url = doc_data.get('original_url')
        file_name = doc_data.get('name')
        mime_type = doc_data.get('type', 'application/octet-stream')

        if not original_url:
            return jsonify({'error': 'Original file URL not found for this document'}), 404

        # Extract blob path from the public URL
        # Example URL: https://storage.googleapis.com/your-bucket/path/to/file
        blob_name = original_url.split(f'https://storage.googleapis.com/{bucket.name}/')[1]
        blob = bucket.blob(blob_name)
        file_content = blob.download_as_bytes()

        return send_file(
            io.BytesIO(file_content),
            mimetype=mime_type,
            as_attachment=True,
            download_name=file_name
        )
    except Exception as e:
        print(f"Error downloading original document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<document_id>/download/digital_copy', methods=['GET'])
def download_digital_copy(document_id):
    """
    Downloads the processed digital copy content (text) as a .txt file.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').document(document_id)
    try:
        doc_snapshot = doc_ref.get()
        if not doc_snapshot.exists:
            return jsonify({'error': 'Document not found'}), 404

        doc_data = doc_snapshot.to_dict()
        digital_copy_content = doc_data.get('digital_copy_content', '')
        original_file_name = doc_data.get('name', 'digital_copy')
        # Create a new filename for the digital copy, e.g., "original_name_digital.txt"
        digital_copy_filename = f"{os.path.splitext(original_file_name)[0]}_digital.txt"

        if not digital_copy_content:
            return jsonify({'error': 'Digital copy content not available for this document'}), 404

        return send_file(
            io.BytesIO(digital_copy_content.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=True,
            download_name=digital_copy_filename
        )
    except Exception as e:
        print(f"Error downloading digital copy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    """
    Deletes a document from Firestore and its corresponding file from Firebase Storage.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').document(document_id)
    try:
        doc_snapshot = doc_ref.get()
        if not doc_snapshot.exists:
            return jsonify({'error': 'Document not found'}), 404

        doc_data = doc_snapshot.to_dict()
        original_url = doc_data.get('original_url')

        # Delete from Firebase Storage first
        if original_url:
            try:
                blob_name = original_url.split(f'https://storage.googleapis.com/{bucket.name}/')[1]
                blob = bucket.blob(blob_name)
                blob.delete()
                print(f"Deleted file from storage: {blob_name}")
            except Exception as e:
                print(f"Warning: Could not delete file from storage ({blob_name}): {e}")
                # Log the error but continue to delete Firestore record

        # Delete from Firestore
        doc_ref.delete()
        return jsonify({'message': 'Document deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """
    Generates an analytics report based on user's documents.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    docs_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents')
    category_counts = {category: 0 for category in DOCUMENT_CATEGORIES}
    total_documents = 0
    total_size_bytes = 0

    try:
        for doc_snapshot in docs_ref.stream():
            doc_data = doc_snapshot.to_dict()
            category = doc_data.get('category', 'Other')
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts['Other'] += 1 # Catch documents with unrecognized categories
            total_documents += 1
            total_size_bytes += doc_data.get('size', 0) # Size in bytes

        # Convert total size to KB for display
        total_size_kb = round(total_size_bytes / 1024, 2) if total_size_bytes > 0 else 0

        analytics_report = {
            'total_documents': total_documents,
            'total_size_kb': total_size_kb,
            'documents_by_category': category_counts,
            'last_updated': datetime.now().isoformat()
        }
        return jsonify(analytics_report), 200
    except Exception as e:
        print(f"Error generating analytics: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development, run on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)









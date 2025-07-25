from flask import Flask, request, jsonify, send_file
from flask_cors import CORS # Import CORS
from firebase_admin import credentials, initialize_app, firestore, storage
from google.cloud import vision
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io
import mimetypes
from datetime import datetime
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Initialize CORS with your Flask app. By default, this allows all origins.

# --- Firebase Initialization ---
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
if FIREBASE_CREDENTIALS_PATH and os.path.exists(FIREBASE_CREDENTIALS_PATH):
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
else:
    print("FIREBASE_CREDENTIALS_PATH not found or not set. Attempting default Firebase initialization (for Google Cloud environments).")
    try:
        cred = credentials.ApplicationDefault()
    except Exception as e:
        print(f"Failed to get ApplicationDefault credentials: {e}")
        print("Please ensure GOOGLE_APPLICATION_CREDENTIALS is set or FIREBASE_CREDENTIALS_PATH points to a valid service account key.")

firebase_app = initialize_app(cred, {
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
})

db = firestore.client()
bucket = storage.bucket()

# --- Google Cloud Vision Initialization ---
vision_client = vision.ImageAnnotatorClient()

# --- Gemini API Configuration ---
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
# Use gemini-2.0-flash for structured responses with schema
gemini_model = genai.GenerativeModel('gemini-2.0-flash') 

# Define allowed document categories for Gemini
DOCUMENT_CATEGORIES = [
    'Lab Results', 'Prescriptions', 'Radiology', 'Discharge Summaries',
    'Vital Signs', 'Insurance', 'Consultation Notes', 'Other'
]

# Define the JSON schema for Gemini's response
response_schema = {
    "type": "OBJECT",
    "properties": {
        "processed_text": {"type": "STRING", "description": "Cleaned, readable, and concise key information from the document."},
        "category": {
            "type": "STRING",
            "enum": DOCUMENT_CATEGORIES, # Force Gemini to choose from these specific categories
            "description": "The determined category from the predefined list."
        },
        "reasoning": {"type": "STRING", "description": "A brief explanation (1-2 sentences) why this category was chosen based on the document's content."}
    },
    "required": ["processed_text", "category", "reasoning"]
}

# Helper function to get user ID
def get_user_id_from_request():
    user_id = request.headers.get('X-User-Id')
    if not user_id:
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
    unique_filename = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}{file_extension}"
    blob_path = f"artifacts/{app_id}/users/{user_id}/original_documents/{unique_filename}"

    try:
        file.seek(0)
        blob = bucket.blob(blob_path)
        blob.upload_from_file(file, content_type=file.content_type)
        original_file_url = blob.public_url

        extracted_text = ""
        digital_copy_content = ""
        category = "Other"

        mime_type = file.content_type
        file.seek(0)

        if mime_type.startswith('image/') or mime_type == 'application/pdf':
            image_content = file.read()
            vision_image = vision.Image(content=image_content)
            response = vision_client.document_text_detection(image=vision_image)
            extracted_text = response.full_text_annotation.text if response.full_text_annotation else ""
            print(f"OCR Extracted Text (first 200 chars): {extracted_text[:200]}...")

            if extracted_text:
                # REVISED Gemini Prompt with stronger instructions
                gemini_prompt = f"""
                You are a highly accurate medical document categorization and summarization AI. Your primary goal is to classify documents into one of the provided medical categories based on their content.

                **Strict Categorization Rules:**
                1.  **Prioritize Medical Categories:** Carefully read the entire document. Identify key medical terms, formats, and information.
                    * If it contains blood test results, glucose levels, cholesterol, or lab names, categorize as 'Lab Results'.
                    * If it lists medications, dosages, instructions, or includes 'Rx', categorize as 'Prescriptions'.
                    * If it mentions MRI, X-ray, CT scan, ultrasound, findings, impression, or radiologist, categorize as 'Radiology'.
                    * If it summarizes a hospital stay, admission/discharge dates, or diagnoses, categorize as 'Discharge Summaries'.
                    * If it lists blood pressure, heart rate, temperature, or respiratory rate, categorize as 'Vital Signs'.
                    * If it pertains to billing, claims, policy numbers, or coverage, categorize as 'Insurance'.
                    * If it contains doctor's notes, patient complaints, or treatment plans from a consultation, categorize as 'Consultation Notes'.
                2.  **Last Resort 'Other':** ONLY if the document is clearly NOT medical, or if its content absolutely does not fit ANY of the above specific medical categories, then, and only then, categorize it as 'Other'. Do NOT invent new categories.

                **Digital Copy Content:**
                Provide a clean, readable, and concise summary of the document's key medical information. Correct any typos or formatting issues from the original text. This processed text should be a clear representation of what the document is about.

                ---
                Document Text to Process and Categorize:
                {extracted_text}
                """
                try:
                    # Pass the schema to generate_content
                    gemini_response = gemini_model.generate_content(
                        gemini_prompt,
                        generation_config={
                            "response_mime_type": "application/json",
                            "response_schema": response_schema
                        }
                    )
                    
                    # The response.text will now be a JSON string directly
                    gemini_output = gemini_response.text.strip()
                    print(f"Gemini Raw Output (JSON): {gemini_output}")

                    parsed_gemini = json.loads(gemini_output)
                    digital_copy_content = parsed_gemini.get('processed_text', extracted_text)
                    determined_category = parsed_gemini.get('category', 'Other')
                    reasoning = parsed_gemini.get('reasoning', 'No specific reasoning provided by LLM.')
                    print(f"Gemini Categorization Reasoning: {reasoning}")

                    # The schema should enforce valid categories, but a final check is good.
                    if determined_category in DOCUMENT_CATEGORIES:
                        category = determined_category
                    else:
                        print(f"Warning: Gemini suggested invalid category '{determined_category}' despite schema. Falling back to 'Other'.")
                        category = "Other"

                except Exception as gemini_err:
                    print(f"Error processing with Gemini: {gemini_err}")
                    # If Gemini fails to respond or parse, fall back to raw OCR and 'Other'
                    digital_copy_content = extracted_text
                    category = "Other"
            else:
                digital_copy_content = "" # No text extracted
                category = "Other"

        elif mime_type == 'text/plain':
            extracted_text = file.read().decode('utf-8')
            if extracted_text:
                # REVISED Gemini Prompt (same as above)
                gemini_prompt = f"""
                You are a highly accurate medical document categorization and summarization AI. Your primary goal is to classify documents into one of the provided medical categories based on their content.

                **Strict Categorization Rules:**
                1.  **Prioritize Medical Categories:** Carefully read the entire document. Identify key medical terms, formats, and information.
                    * If it contains blood test results, glucose levels, cholesterol, or lab names, categorize as 'Lab Results'.
                    * If it lists medications, dosages, instructions, or includes 'Rx', categorize as 'Prescriptions'.
                    * If it mentions MRI, X-ray, CT scan, ultrasound, findings, impression, or radiologist, categorize as 'Radiology'.
                    * If it summarizes a hospital stay, admission/discharge dates, or diagnoses, categorize as 'Discharge Summaries'.
                    * If it lists blood pressure, heart rate, temperature, or respiratory rate, categorize as 'Vital Signs'.
                    * If it pertains to billing, claims, policy numbers, or coverage, categorize as 'Insurance'.
                    * If it contains doctor's notes, patient complaints, or treatment plans from a consultation, categorize as 'Consultation Notes'.
                2.  **Last Resort 'Other':** ONLY if the document is clearly NOT medical, or if its content absolutely does not fit ANY of the above specific medical categories, then, and only then, categorize it as 'Other'. Do NOT invent new categories.

                **Digital Copy Content:**
                Provide a clean, readable, and concise summary of the document's key medical information. Correct any typos or formatting issues from the original text. This processed text should be a clear representation of what the document is about.

                ---
                Document Text to Process and Categorize:
                {extracted_text}
                """
                try:
                    gemini_response = gemini_model.generate_content(
                        gemini_prompt,
                        generation_config={
                            "response_mime_type": "application/json",
                            "response_schema": response_schema
                        }
                    )
                    gemini_output = gemini_response.text.strip()
                    print(f"Gemini Raw Output (JSON): {gemini_output}")

                    parsed_gemini = json.loads(gemini_output)
                    digital_copy_content = parsed_gemini.get('processed_text', extracted_text)
                    determined_category = parsed_gemini.get('category', 'Other')
                    reasoning = parsed_gemini.get('reasoning', 'No specific reasoning provided by LLM.')
                    print(f"Gemini Categorization Reasoning: {reasoning}")

                    if determined_category in DOCUMENT_CATEGORIES:
                        category = determined_category
                    else:
                        print(f"Warning: Gemini suggested invalid category '{determined_category}' despite schema. Falling back to 'Other'.")
                        category = "Other"
                except Exception as gemini_err:
                    print(f"Error processing with Gemini: {gemini_err}")
                    digital_copy_content = extracted_text
                    category = "Other"
            else:
                digital_copy_content = ""
                category = "Other"
        else:
            extracted_text = ""
            digital_copy_content = ""
            category = "Other"

        doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').add({
            'name': original_filename,
            'type': mime_type,
            'original_url': original_file_url,
            'digital_copy_content': digital_copy_content,
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

        blob_name = original_url.split(f'https://storage.googleapis.com/{bucket.name}/')[1]
        blob = bucket.blob(blob_name)
        file_content = blob.download_as_bytes()

        return send_file(
            io.BytesIO(file_content),
            mimetype=mime_type,
            as_attachment=False, # Changed to False for in-browser viewing
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
        digital_copy_filename = f"{os.path.splitext(original_file_name)[0]}_digital.txt"

        if not digital_copy_content:
            return jsonify({'error': 'Digital copy content not available for this document'}), 404

        return send_file(
            io.BytesIO(digital_copy_content.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=False, # Changed to False for in-browser viewing
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

        if original_url:
            try:
                blob_name = original_url.split(f'https://storage.googleapis.com/{bucket.name}/')[1]
                blob = bucket.blob(blob_name)
                blob.delete()
                print(f"Deleted file from storage: {blob_name}")
            except Exception as e:
                print(f"Warning: Could not delete file from storage ({blob_name}): {e}")

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
                category_counts['Other'] += 1 # Fallback for any unexpected category
            total_documents += 1
            total_size_bytes += doc_data.get('size', 0)

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
    app.run(debug=True, host='0.0.0.0', port=5000)








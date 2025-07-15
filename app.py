"""
Simplified FastAPI Receipt Analysis Application
Core operations: Upload, Read, Save, View, Delete documents
"""

import os
import uuid
import logging
from datetime import datetime, timezone, date
from typing import List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress uvicorn access logs for WebSocket attempts
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Load environment variables
load_dotenv()

# Lifespan context manager for MongoDB connection
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for MongoDB connection"""
    # Startup
    try:
        # Test the connection with longer timeout
        logger.info("Testing MongoDB connection...")
        await mongodb_client.admin.command('ping', serverSelectionTimeoutMS=30000)
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Please check your network connection and Cosmos DB settings")
        # Don't raise here - let the app start but log the error
        # The app can still work if the connection is established later
    
    yield
    
    # Shutdown
    mongodb_client.close()
    logger.info("MongoDB connection closed")

# FastAPI app initialization
app = FastAPI(
    title="Receipt Analysis API",
    description="Simple API for analyzing receipts using Azure Document Intelligence",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080", 
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Create necessary directories before mounting static files
if not os.path.exists("uploads"):
    print("Creating uploads folder")
    os.makedirs("uploads")
if not os.path.exists("receipt_images"):
    print("Creating receipt_images folder")
    os.makedirs("receipt_images")

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve receipt images
app.mount("/receipt_images", StaticFiles(directory="receipt_images"), name="receipt_images")

# MongoDB setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# MongoDB client
mongodb_client = AsyncIOMotorClient(DATABASE_URL)

# Extract database name from connection string or use default
from urllib.parse import urlparse, parse_qs
import re

# Debug: Log the connection string (without password)
parsed_url = urlparse(DATABASE_URL)
debug_url = DATABASE_URL.replace(parsed_url.password, '***') if parsed_url.password else DATABASE_URL
logger.info(f"Connecting to MongoDB with URL: {debug_url}")
logger.info(f"Parsed URL - path: '{parsed_url.path}', netloc: '{parsed_url.netloc}'")

# Try multiple methods to extract database name
database_name = None

# Method 1: Try to extract from path
if parsed_url.path:
    database_name = parsed_url.path.strip('/')
    logger.info(f"Database name from path: '{database_name}'")

# Method 2: Try to extract from query parameters
if not database_name:
    query_params = parse_qs(parsed_url.query)
    logger.info(f"Query parameters: {query_params}")
    for key, value in query_params.items():
        if key.lower() in ['database', 'db', 'database_name']:
            database_name = value[0]
            break

# Method 3: Try regex extraction for MongoDB+srv URLs
if not database_name:
    # Pattern for MongoDB+srv URLs: mongodb+srv://user:pass@host/database?params
    pattern = r'mongodb\+srv://[^/]+/([^?]+)'
    match = re.search(pattern, DATABASE_URL)
    if match:
        database_name = match.group(1)
        logger.info(f"Database name from regex: '{database_name}'")

# Method 4: Try regex extraction for regular MongoDB URLs
if not database_name:
    # Pattern for regular MongoDB URLs: mongodb://user:pass@host:port/database?params
    pattern = r'mongodb://[^/]+/([^?]+)'
    match = re.search(pattern, DATABASE_URL)
    if match:
        database_name = match.group(1)
        logger.info(f"Database name from regex (regular): '{database_name}'")

# If still no database name, use default
if not database_name:
    database_name = 'receipts_db'
    logger.info("No database name found, using default: receipts_db")

logger.info(f"Final database name: '{database_name}'")

# Validate database name
if not database_name or database_name.strip() == '':
    database_name = 'receipts_db'
    logger.warning("Database name was empty, using default: receipts_db")

# Get database
database = mongodb_client[database_name]

# Collections
receipts_collection = database.receipts
receipt_items_collection = database.receipt_items

# Pydantic models for API
class ReceiptItemResponse(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    confidence: Optional[float] = None

class ReceiptResponse(BaseModel):
    id: str
    filename: str
    image_url: Optional[str] = None
    merchant_name: Optional[str] = None
    transaction_date: Optional[datetime] = None
    total: Optional[float] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    receipt_type: Optional[str] = None
    country_region: Optional[str] = None
    confidence_score: Optional[float] = None
    items: List[ReceiptItemResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True

# Azure Document Intelligence setup
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

if not endpoint or not key:
    raise ValueError("Azure Document Intelligence credentials not found in environment variables")

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

# Utility functions
def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return file path"""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_extension = Path(upload_file.filename or "").suffix
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = upload_dir / file_name
    
    with open(file_path, "wb") as buffer:
        content = upload_file.file.read()
        buffer.write(content)
    
    return str(file_path)

def save_upload_file_with_content(file_content: bytes, filename: str) -> str:
    """Save uploaded file using pre-read content and return file path"""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_extension = Path(filename).suffix
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = upload_dir / file_name
    
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
    return str(file_path)

def save_receipt_image(upload_file: UploadFile, receipt_id: str) -> str:
    """Save receipt image permanently and return file path"""
    images_dir = Path("receipt_images")
    images_dir.mkdir(exist_ok=True)
    
    file_extension = Path(upload_file.filename or "").suffix
    file_name = f"{receipt_id}{file_extension}"
    file_path = images_dir / file_name
    
    # Reset file pointer to beginning
    upload_file.file.seek(0)
    
    with open(file_path, "wb") as buffer:
        content = upload_file.file.read()
        buffer.write(content)
    
    return str(file_path)

def save_receipt_image_with_content(file_content: bytes, filename: str, receipt_id: str) -> str:
    """Save receipt image permanently using pre-read content and return file path"""
    images_dir = Path("receipt_images")
    images_dir.mkdir(exist_ok=True)
    
    file_extension = Path(filename).suffix
    file_name = f"{receipt_id}{file_extension}"
    file_path = images_dir / file_name
    
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
    return str(file_path)

def analyze_receipt(file_path: str):
    """Analyze receipt from local file"""
    try:
        with open(file_path, "rb") as document:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-receipt", document
            )
        receipts = poller.result()
        return receipts
    except Exception as e:
        logger.error(f"Azure Document Intelligence error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze receipt. Please try again with a clearer image."
        )

def extract_receipt_data(receipt_doc):
    """Extract structured data from Azure response"""
    data = {
        "merchant_name": None,
        "transaction_date": None,
        "total": None,
        "subtotal": None,
        "tax_amount": None,
        "receipt_type": None,
        "country_region": None,
        "items": [],
        "confidence_scores": []
    }
    
    # Debug: Log available fields
    logger.info(f"Available fields: {list(receipt_doc.fields.keys())}")
    
    # Debug: Log all field names to check for payment-related fields
    all_field_names = list(receipt_doc.fields.keys())
    payment_related_fields = [field for field in all_field_names if 'payment' in field.lower() or 'pay' in field.lower()]
    logger.info(f"Payment-related fields found: {payment_related_fields}")
    
    # Extract basic fields
    if receipt_doc.fields.get("MerchantName"):
        data["merchant_name"] = receipt_doc.fields["MerchantName"].value_string
        data["confidence_scores"].append(receipt_doc.fields["MerchantName"].confidence)
    
    if receipt_doc.fields.get("TransactionDate"):
        data["transaction_date"] = receipt_doc.fields["TransactionDate"].value_date
        data["confidence_scores"].append(receipt_doc.fields["TransactionDate"].confidence)
    
    if receipt_doc.fields.get("Total"):
        total_field = receipt_doc.fields["Total"]
        if getattr(total_field, "value_currency", None) and getattr(total_field.value_currency, "amount", None) is not None:
            data["total"] = total_field.value_currency.amount
            data["confidence_scores"].append(total_field.confidence)
    
    # Extract new fields with fallback names
    # Subtotal - try multiple possible field names
    subtotal_field = None
    for field_name in ["Subtotal", "SubTotal", "subtotal"]:
        if receipt_doc.fields.get(field_name):
            subtotal_field = receipt_doc.fields[field_name]
            break
    
    if subtotal_field and getattr(subtotal_field, "value_currency", None) and getattr(subtotal_field.value_currency, "amount", None) is not None:
        data["subtotal"] = subtotal_field.value_currency.amount
        data["confidence_scores"].append(subtotal_field.confidence)
    
    # Tax - try multiple possible field names
    tax_field = None
    for field_name in ["TotalTax", "Tax", "SalesTax", "TaxAmount", "tax"]:
        if receipt_doc.fields.get(field_name):
            tax_field = receipt_doc.fields[field_name]
            break
    
    if tax_field and getattr(tax_field, "value_currency", None) and getattr(tax_field.value_currency, "amount", None) is not None:
        data["tax_amount"] = tax_field.value_currency.amount
        data["confidence_scores"].append(tax_field.confidence)
    
    # Extract receipt type and country region
    if receipt_doc.fields.get("ReceiptType"):
        data["receipt_type"] = receipt_doc.fields["ReceiptType"].value_string
        data["confidence_scores"].append(receipt_doc.fields["ReceiptType"].confidence)
    
    if receipt_doc.fields.get("CountryRegion"):
        data["country_region"] = receipt_doc.fields["CountryRegion"].value_country_region
        data["confidence_scores"].append(receipt_doc.fields["CountryRegion"].confidence)
    
    # Extract items
    if receipt_doc.fields.get("Items"):
        for item in receipt_doc.fields["Items"].value_array:
            item_data = {}
            if item.value_object.get("Description"):
                item_data["description"] = item.value_object["Description"].value_string
            if item.value_object.get("Quantity"):
                item_data["quantity"] = item.value_object["Quantity"].value_number
            if item.value_object.get("Price"):
                price_field = item.value_object["Price"]
                if getattr(price_field, "value_currency", None) and getattr(price_field.value_currency, "amount", None) is not None:
                    item_data["unit_price"] = price_field.value_currency.amount
            if item.value_object.get("TotalPrice"):
                total_price_field = item.value_object["TotalPrice"]
                if getattr(total_price_field, "value_currency", None) and getattr(total_price_field.value_currency, "amount", None) is not None:
                    item_data["total_price"] = total_price_field.value_currency.amount
            data["items"].append(item_data)
    
    # Calculate average confidence
    data["confidence_score"] = sum(data["confidence_scores"]) / len(data["confidence_scores"]) if data["confidence_scores"] else 0
    
    # Debug: Log extracted values
    logger.info(f"Extracted data: {data}")
    
    return data

# API Endpoints

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        await mongodb_client.admin.command('ping', serverSelectionTimeoutMS=5000)
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "azure_document_intelligence": "configured" if endpoint and key else "not_configured"
    }

# 1. UPLOAD DOCUMENT
@app.post("/upload", response_model=ReceiptResponse)
async def upload_document(
    file: UploadFile = File(...)
):
    """Upload and analyze receipt document"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG, BMP, TIFF, HEIF)"
        )
    
    # Validate file size (max 10MB for better performance)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size must be less than 10MB"
        )
    
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heif', '.heic'}
    file_extension = Path(file.filename or "").suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Read file content once and store it
    file_content = file.file.read()
    
    # Save uploaded file for analysis
    file_path = save_upload_file_with_content(file_content, file.filename or "unknown")
    
    try:
        # Analyze receipt
        receipts = analyze_receipt(file_path)
        
        if not receipts.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No receipt found in the image. Please ensure the image contains a clear, readable receipt."
            )
        
        # Extract data from first receipt
        receipt_data = extract_receipt_data(receipts.documents[0])

        # If all key fields are missing, likely not a receipt
        if (
            not receipt_data.get("transaction_date") and
            not receipt_data.get("total") and
            (not receipt_data.get("items") or len(receipt_data.get("items", [])) == 0)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No receipt found in the image. Please ensure the image contains a clear, readable receipt."
            )
        
        # Generate a unique ID for the receipt
        receipt_id = str(uuid.uuid4())
        
        # Save receipt image permanently using the receipt ID
        image_path = save_receipt_image_with_content(file_content, file.filename or "unknown", receipt_id)
        
        # Create receipt document for MongoDB
        # Convert date to datetime for MongoDB compatibility
        transaction_date = receipt_data["transaction_date"]
        if transaction_date and isinstance(transaction_date, date) and not isinstance(transaction_date, datetime):
            # If it's a date object (not datetime), convert to datetime
            transaction_date = datetime.combine(transaction_date, datetime.min.time())
        
        receipt_doc = {
            "_id": receipt_id,
            "filename": file.filename,
            "image_path": image_path,
            "merchant_name": receipt_data["merchant_name"],
            "transaction_date": transaction_date,
            "total": receipt_data["total"],
            "subtotal": receipt_data["subtotal"],
            "tax_amount": receipt_data["tax_amount"],
            "receipt_type": receipt_data["receipt_type"],
            "country_region": receipt_data["country_region"],
            "confidence_score": receipt_data["confidence_score"],
            "raw_data": str(receipts.documents[0].__dict__),
            "created_at": datetime.now(timezone.utc)
        }
        
        # Save receipt to MongoDB
        await receipts_collection.insert_one(receipt_doc)
        
        # Save items to MongoDB
        if receipt_data["items"]:
            items_docs = []
            for item_data in receipt_data["items"]:
                item_doc = {
                    "receipt_id": receipt_id,
                    "description": item_data.get("description"),
                    "quantity": item_data.get("quantity"),
                    "unit_price": item_data.get("unit_price"),
                    "total_price": item_data.get("total_price"),
                    "confidence": item_data.get("confidence")
                }
                items_docs.append(item_doc)
            
            if items_docs:
                await receipt_items_collection.insert_many(items_docs)
        
        # Get items for response
        items_cursor = receipt_items_collection.find({"receipt_id": receipt_id})
        items = await items_cursor.to_list(length=None)
        
        # Generate image URL
        image_url = f"/receipt_images/{receipt_id}{Path(file.filename or "").suffix}" if receipt_doc["image_path"] else None
        
        return ReceiptResponse(
            id=receipt_doc["_id"],
            filename=receipt_doc["filename"],
            image_url=image_url,
            merchant_name=receipt_doc["merchant_name"],
            transaction_date=receipt_doc["transaction_date"],
            total=receipt_doc["total"],
            subtotal=receipt_doc["subtotal"],
            tax_amount=receipt_doc["tax_amount"],
            receipt_type=receipt_doc["receipt_type"],
            country_region=receipt_doc["country_region"],
            confidence_score=receipt_doc["confidence_score"],
            items=[ReceiptItemResponse(
                description=item.get("description"),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total_price=item.get("total_price"),
                confidence=item.get("confidence")
            ) for item in items],
            created_at=receipt_doc["created_at"]
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Log the error for debugging
        logger.error(f"Receipt analysis error: {str(e)}")
        
        # Provide user-friendly error message
        if "InvalidContent" in str(e) or "corrupted" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The image file appears to be corrupted or in an unsupported format. Please try with a different image (JPEG or PNG recommended)."
            )
        elif "InvalidRequest" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The image could not be processed. Please ensure it's a clear, readable receipt image."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to analyze receipt. Please try again with a clearer image."
            )
    finally:
        # Always clean up the uploaded file after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up file {file_path}: {cleanup_error}")

@app.post("/upload_camera", response_model=ReceiptResponse)
async def upload_camera(
    request: Request,
    file: UploadFile = File(...)
):
    """Upload a camera-captured image and analyze as a receipt (calls upload_document)."""
    # Call the existing upload_document function
    return await upload_document(file=file)

# 2. READ DOCUMENT (Get specific receipt)
@app.get("/documents/{document_id}", response_model=ReceiptResponse)
async def read_document(document_id: str):
    """Read specific document by ID"""
    receipt = await receipts_collection.find_one({"_id": document_id})
    if not receipt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get items for this receipt
    items_cursor = receipt_items_collection.find({"receipt_id": document_id})
    items = await items_cursor.to_list(length=None)
    
    # Generate image URL
    image_url = f"/receipt_images/{receipt['_id']}{Path(receipt['filename']).suffix}" if receipt.get("image_path") else None
    
    return ReceiptResponse(
        id=receipt["_id"],
        filename=receipt["filename"],
        image_url=image_url,
        merchant_name=receipt.get("merchant_name"),
        transaction_date=receipt.get("transaction_date"),
        total=receipt.get("total"),
        subtotal=receipt.get("subtotal"),
        tax_amount=receipt.get("tax_amount"),
        receipt_type=receipt.get("receipt_type"),
        country_region=receipt.get("country_region"),
        confidence_score=receipt.get("confidence_score"),
        items=[ReceiptItemResponse(
            description=item.get("description"),
            quantity=item.get("quantity"),
            unit_price=item.get("unit_price"),
            total_price=item.get("total_price"),
            confidence=item.get("confidence")
        ) for item in items],
        created_at=receipt["created_at"]
    )

# 3. SAVE DOCUMENT (This is handled in upload, but keeping for clarity)
@app.post("/documents", response_model=ReceiptResponse)
async def save_document(
    file: UploadFile = File(...)
):
    """Save document to database (same as upload)"""
    return await upload_document(file)

# 4. VIEW DOCUMENTS (List all documents)
@app.get("/documents", response_model=List[ReceiptResponse])
async def view_documents(
    skip: int = 0,
    limit: int = 100
):
    """View all documents with pagination"""
    receipts_cursor = receipts_collection.find().skip(skip).limit(limit).sort("created_at", -1)
    receipts = await receipts_cursor.to_list(length=None)
    
    result = []
    for receipt in receipts:
        # Get items for this receipt
        items_cursor = receipt_items_collection.find({"receipt_id": receipt["_id"]})
        items = await items_cursor.to_list(length=None)
        
        # Generate image URL
        image_url = f"/receipt_images/{receipt['_id']}{Path(receipt['filename']).suffix}" if receipt.get("image_path") else None
        
        result.append(ReceiptResponse(
            id=receipt["_id"],
            filename=receipt["filename"],
            image_url=image_url,
            merchant_name=receipt.get("merchant_name"),
            transaction_date=receipt.get("transaction_date"),
            total=receipt.get("total"),
            subtotal=receipt.get("subtotal"),
            tax_amount=receipt.get("tax_amount"),
            receipt_type=receipt.get("receipt_type"),
            country_region=receipt.get("country_region"),
            confidence_score=receipt.get("confidence_score"),
            items=[ReceiptItemResponse(
                description=item.get("description"),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total_price=item.get("total_price"),
                confidence=item.get("confidence")
            ) for item in items],
            created_at=receipt["created_at"]
        ))
    
    return result

# 5. DELETE DOCUMENT
@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document by ID"""
    receipt = await receipts_collection.find_one({"_id": document_id})
    if not receipt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete associated items first
    await receipt_items_collection.delete_many({"receipt_id": document_id})
    
    # Delete receipt image if it exists
    if receipt.get("image_path") and os.path.exists(receipt["image_path"]):
        try:
            os.remove(receipt["image_path"])
        except Exception as e:
            logger.warning(f"Failed to delete receipt image {receipt['image_path']}: {e}")
    
    # Delete receipt from MongoDB
    await receipts_collection.delete_one({"_id": document_id})
    
    return {"message": "Document deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
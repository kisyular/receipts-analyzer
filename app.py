"""
FastAPI Receipt Analysis Application with MongoDB
Core operations: Upload, Read, View, Delete receipts
"""

import os
import uuid
import logging
from datetime import datetime, timezone, date
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urlparse

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

# Suppress uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Load environment variables
load_dotenv()

# Environment validation
DATABASE_URL = os.getenv("DATABASE_URL")
AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

if not all([DATABASE_URL, AZURE_ENDPOINT, AZURE_KEY]):
    raise ValueError("Missing required environment variables: DATABASE_URL, AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY")

# MongoDB setup
mongodb_client = AsyncIOMotorClient(DATABASE_URL)

# Extract database name from connection string
parsed_url = urlparse(DATABASE_URL)
print(parsed_url)
print(parsed_url.path)
database_name = parsed_url.path.strip('/') if parsed_url.path else 'receipts_db'
database = mongodb_client[database_name]
receipts_collection = database.receipts
receipt_items_collection = database.receipt_items

# Azure Document Intelligence setup
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=str(AZURE_ENDPOINT), credential=AzureKeyCredential(str(AZURE_KEY))
)

# Pydantic models
class ReceiptItemResponse(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None

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

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for MongoDB connection"""
    try:
        await mongodb_client.admin.command('ping', serverSelectionTimeoutMS=30000)
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
    
    yield
    
    mongodb_client.close()
    logger.info("MongoDB connection closed")

# FastAPI app initialization
app = FastAPI(
    title="Receipt Analysis API",
    description="API for analyzing receipts using Azure Document Intelligence and MongoDB",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Simplified for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("receipt_images").mkdir(exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")
app.mount("/receipt_images", StaticFiles(directory="receipt_images"), name="receipt_images")

# Utility functions
def save_receipt_image(file_content: bytes, filename: str, receipt_id: str) -> str:
    """Save receipt image permanently"""
    file_extension = Path(filename).suffix
    file_name = f"{receipt_id}{file_extension}"
    file_path = Path("receipt_images") / file_name
    
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
    return str(file_path)

from io import BytesIO

def analyze_receipt(file_content: bytes):
    """Analyze receipt using Azure Document Intelligence"""
    try:
        with BytesIO(file_content) as document:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-receipt", document
            )
        return poller.result()
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
    
    fields = receipt_doc.fields
    
    # Extract basic fields
    if fields.get("MerchantName"):
        data["merchant_name"] = fields["MerchantName"].value_string
        data["confidence_scores"].append(fields["MerchantName"].confidence)
    
    if fields.get("TransactionDate"):
        data["transaction_date"] = fields["TransactionDate"].value_date
        data["confidence_scores"].append(fields["TransactionDate"].confidence)
    
    if fields.get("Total"):
        total_field = fields["Total"]
        if getattr(total_field, "value_currency", None) and getattr(total_field.value_currency, "amount", None) is not None:
            data["total"] = total_field.value_currency.amount
            data["confidence_scores"].append(total_field.confidence)
    
    # Extract subtotal with fallback names
    subtotal_field = None
    for field_name in ["Subtotal", "SubTotal"]:
        if fields.get(field_name):
            subtotal_field = fields[field_name]
            break
    
    if subtotal_field and getattr(subtotal_field, "value_currency", None) and getattr(subtotal_field.value_currency, "amount", None) is not None:
        data["subtotal"] = subtotal_field.value_currency.amount
        data["confidence_scores"].append(subtotal_field.confidence)
    
    # Extract tax with fallback names
    tax_field = None
    for field_name in ["TotalTax", "Tax", "SalesTax", "TaxAmount"]:
        if fields.get(field_name):
            tax_field = fields[field_name]
            break
    
    if tax_field and getattr(tax_field, "value_currency", None) and getattr(tax_field.value_currency, "amount", None) is not None:
        data["tax_amount"] = tax_field.value_currency.amount
        data["confidence_scores"].append(tax_field.confidence)
    
    # Extract receipt type and country region
    if fields.get("ReceiptType"):
        data["receipt_type"] = fields["ReceiptType"].value_string
        data["confidence_scores"].append(fields["ReceiptType"].confidence)
    
    if fields.get("CountryRegion"):
        data["country_region"] = fields["CountryRegion"].value_country_region
        data["confidence_scores"].append(fields["CountryRegion"].confidence)
    
    # Extract items
    if fields.get("Items"):
        for item in fields["Items"].value_array:
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
        await mongodb_client.admin.command('ping', serverSelectionTimeoutMS=5000)
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "azure_document_intelligence": "configured"
    }

@app.post("/upload", response_model=ReceiptResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and analyze receipt document"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG, BMP, TIFF, HEIF)"
        )
    
    # Validate file size (max 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size must be less than 10MB"
        )
    
    # Read file content
    file_content = file.file.read()
    
    try:
        # Analyze receipt
        receipts = analyze_receipt(file_content)
        
        if not receipts.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No receipt found in the image. Please ensure the image contains a clear, readable receipt."
            )
        
        # Extract data from first receipt
        receipt_data = extract_receipt_data(receipts.documents[0])

        # Validate that we have essential data
        if not receipt_data.get("transaction_date") and not receipt_data.get("total"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No receipt found in the image. Please ensure the image contains a clear, readable receipt."
            )
        
        # Generate receipt ID and save image
        receipt_id = str(uuid.uuid4())
        image_path = save_receipt_image(file_content, file.filename or "unknown", receipt_id)
        
        # Convert date to datetime for MongoDB
        transaction_date = receipt_data["transaction_date"]
        if transaction_date and isinstance(transaction_date, date) and not isinstance(transaction_date, datetime):
            transaction_date = datetime.combine(transaction_date, datetime.min.time())
        
        # Create receipt document
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
            "created_at": datetime.now(timezone.utc)
        }
        
        # Save receipt to MongoDB
        await receipts_collection.insert_one(receipt_doc)
        
        # Save items to MongoDB
        if receipt_data["items"]:
            items_docs = [
                {
                    "receipt_id": receipt_id,
                    "description": item.get("description"),
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "total_price": item.get("total_price")
                }
                for item in receipt_data["items"]
            ]
            await receipt_items_collection.insert_many(items_docs)
        
        # Get items for response
        items_cursor = receipt_items_collection.find({"receipt_id": receipt_id})
        items = await items_cursor.to_list(length=None)
        
        # Generate image URL
        filename_suffix = Path(file.filename or "").suffix
        image_url = f"/receipt_images/{receipt_id}{filename_suffix}"
        
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
                total_price=item.get("total_price")
            ) for item in items],
            created_at=receipt_doc["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Receipt analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze receipt. Please try again with a clearer image."
        )

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
    filename_suffix = Path(receipt['filename']).suffix
    image_url = f"/receipt_images/{receipt['_id']}{filename_suffix}"
    
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
            total_price=item.get("total_price")
        ) for item in items],
        created_at=receipt["created_at"]
    )

@app.get("/documents", response_model=List[ReceiptResponse])
async def view_documents(skip: int = 0, limit: int = 100):
    """View all documents with pagination"""
    receipts_cursor = receipts_collection.find().skip(skip).limit(limit).sort("created_at", -1)
    receipts = await receipts_cursor.to_list(length=None)
    
    result = []
    for receipt in receipts:
        # Get items for this receipt
        items_cursor = receipt_items_collection.find({"receipt_id": receipt["_id"]})
        items = await items_cursor.to_list(length=None)
        
        # Generate image URL
        filename_suffix = Path(receipt['filename']).suffix
        image_url = f"/receipt_images/{receipt['_id']}{filename_suffix}"
        
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
                total_price=item.get("total_price")
            ) for item in items],
            created_at=receipt["created_at"]
        ))
    
    return result

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
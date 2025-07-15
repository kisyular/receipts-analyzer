"""
FastAPI Receipt Analysis Application with MongoDB
Core operations: Upload, Read, View, Delete receipts
"""

import os
import uuid
import logging
from datetime import datetime, timezone, date, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urlparse
import calendar
from collections import defaultdict, Counter

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
db_path = parsed_url.path
if isinstance(db_path, str):
    database_name = db_path.lstrip("/") if db_path else 'receipts_db'
else:
    database_name = 'receipts_db'
database = mongodb_client[str(database_name)]
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

# Analytics Models
class SpendingSummary(BaseModel):
    total_spent: float
    total_receipts: int
    average_per_receipt: float
    total_tax: float
    tax_percentage: float

class MonthlySpending(BaseModel):
    month: str
    year: int
    total_spent: float
    total_receipts: int
    total_tax: float

class ReceiptTypeAnalysis(BaseModel):
    receipt_type: str
    total_spent: float
    receipt_count: int
    average_spent: float
    percentage_of_total: float

# Analytics Functions
async def get_spending_summary() -> SpendingSummary:
    """Get overall spending summary"""
    pipeline = [
        {"$match": {"total": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": None,
            "total_spent": {"$sum": "$total"},
            "total_receipts": {"$sum": 1},
            "total_tax": {"$sum": {"$ifNull": ["$tax_amount", 0]}}
        }}
    ]
    
    result = await receipts_collection.aggregate(pipeline).to_list(1)
    if not result:
        return SpendingSummary(
            total_spent=0, total_receipts=0, average_per_receipt=0,
            total_tax=0, tax_percentage=0
        )
    
    data = result[0]
    total_spent = data["total_spent"]
    total_receipts = data["total_receipts"]
    total_tax = data["total_tax"]
    
    return SpendingSummary(
        total_spent=total_spent,
        total_receipts=total_receipts,
        average_per_receipt=total_spent / total_receipts if total_receipts > 0 else 0,
        total_tax=total_tax,
        tax_percentage=(total_tax / total_spent * 100) if total_spent > 0 else 0
    )

async def get_monthly_spending() -> List[MonthlySpending]:
    """Get spending by month"""
    pipeline = [
        {"$match": {"transaction_date": {"$exists": True}, "total": {"$exists": True}}},
        {"$addFields": {
            "year": {"$year": "$transaction_date"},
            "month": {"$month": "$transaction_date"}
        }},
        {"$group": {
            "_id": {"year": "$year", "month": "$month"},
            "total_spent": {"$sum": "$total"},
            "total_receipts": {"$sum": 1},
            "total_tax": {"$sum": {"$ifNull": ["$tax_amount", 0]}}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]
    
    results = await receipts_collection.aggregate(pipeline).to_list(None)
    
    monthly_data = []
    for result in results:
        month_val = result["_id"]["month"]
        if isinstance(month_val, (list, tuple)):
            # Defensive: flatten if somehow a sequence
            month_val = month_val[0] if month_val else 1
        month_name = calendar.month_name[month_val]
        monthly_data.append(MonthlySpending(
            month=str(month_name),
            year=result["_id"]["year"],
            total_spent=result["total_spent"],
            total_receipts=result["total_receipts"],
            total_tax=result["total_tax"]
        ))
    
    return monthly_data

async def get_receipt_type_analysis() -> List[ReceiptTypeAnalysis]:
    """Get spending analysis by receipt type"""
    pipeline = [
        {"$match": {"receipt_type": {"$exists": True}, "total": {"$exists": True}}},
        {"$group": {
            "_id": "$receipt_type",
            "total_spent": {"$sum": "$total"},
            "receipt_count": {"$sum": 1}
        }},
        {"$sort": {"total_spent": -1}}
    ]
    
    results = await receipts_collection.aggregate(pipeline).to_list(None)
    
    # Calculate total spending for percentage
    total_spending = sum(r["total_spent"] for r in results)
    
    type_analysis = []
    for result in results:
        total_spent = result["total_spent"]
        receipt_count = result["receipt_count"]
        
        type_analysis.append(ReceiptTypeAnalysis(
            receipt_type=result["_id"] or "Unknown",
            total_spent=total_spent,
            receipt_count=receipt_count,
            average_spent=total_spent / receipt_count if receipt_count > 0 else 0,
            percentage_of_total=(total_spent / total_spending * 100) if total_spending > 0 else 0
        ))
    
    return type_analysis

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

@app.get("/analytics")
async def analytics():
    """Serve the analytics dashboard page"""
    return FileResponse("analytics.html")

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

# Analytics Endpoints
@app.get("/analytics/summary", response_model=SpendingSummary)
async def get_analytics_summary():
    """Get overall spending summary"""
    return await get_spending_summary()

@app.get("/analytics/monthly", response_model=List[MonthlySpending])
async def get_analytics_monthly():
    """Get monthly spending breakdown"""
    return await get_monthly_spending()

@app.get("/analytics/receipt-types", response_model=List[ReceiptTypeAnalysis])
async def get_analytics_receipt_types():
    """Get spending analysis by receipt type"""
    return await get_receipt_type_analysis()

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get comprehensive analytics dashboard data"""
    try:
        summary = await get_spending_summary()
        monthly = await get_monthly_spending()
        receipt_types = await get_receipt_type_analysis()
        
        return {
            "summary": summary,
            "monthly_spending": monthly,
            "receipt_types": receipt_types,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Dashboard generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analytics dashboard"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
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
    has_discount: Optional[bool] = None
    discount_amount: Optional[float] = None

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
    try:
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
    except Exception as e:
        logger.error(f"Error in get_spending_summary: {str(e)}")
        return SpendingSummary(
            total_spent=0, total_receipts=0, average_per_receipt=0,
            total_tax=0, tax_percentage=0
        )

async def get_monthly_spending() -> List[MonthlySpending]:
    """Get spending by month - CosmosDB compatible version"""
    try:
        # Get all receipts with transaction dates and totals
        receipts = await receipts_collection.find({
            "transaction_date": {"$exists": True, "$ne": None},
            "total": {"$exists": True, "$ne": None}
        }).to_list(None)
        
        # Group by month and year manually
        monthly_groups = {}
        
        for receipt in receipts:
            transaction_date = receipt.get("transaction_date")
            total = receipt.get("total", 0) or 0
            tax_amount = receipt.get("tax_amount", 0) or 0
            
            if transaction_date:
                # Extract year and month from datetime
                if isinstance(transaction_date, datetime):
                    year = transaction_date.year
                    month = transaction_date.month
                elif isinstance(transaction_date, str):
                    # Parse string date
                    try:
                        dt = datetime.fromisoformat(transaction_date.replace('Z', '+00:00'))
                        year = dt.year
                        month = dt.month
                    except:
                        continue
                else:
                    continue
                
                key = f"{year}-{month:02d}"
                
                if key not in monthly_groups:
                    monthly_groups[key] = {
                        "year": year,
                        "month": month,
                        "total_spent": 0,
                        "total_receipts": 0,
                        "total_tax": 0
                    }
                
                monthly_groups[key]["total_spent"] += total
                monthly_groups[key]["total_receipts"] += 1
                monthly_groups[key]["total_tax"] += tax_amount
        
        # Convert to MonthlySpending objects
        monthly_data = []
        for key in sorted(monthly_groups.keys()):
            group = monthly_groups[key]
            
            # Validate year and month
            if group["year"] is None or group["month"] is None:
                continue
                
            month_name = calendar.month_name[group["month"]]
            
            monthly_data.append(MonthlySpending(
                month=str(month_name),
                year=int(group["year"]),
                total_spent=float(group["total_spent"]),
                total_receipts=int(group["total_receipts"]),
                total_tax=float(group["total_tax"])
            ))
        
        return monthly_data
    except Exception as e:
        logger.error(f"Error in get_monthly_spending: {str(e)}")
        return []

async def get_receipt_type_analysis() -> List[ReceiptTypeAnalysis]:
    """Get spending analysis by receipt type"""
    try:
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
    except Exception as e:
        logger.error(f"Error in get_receipt_type_analysis: {str(e)}")
        return []

async def get_top_merchants() -> List[dict]:
    """Get top merchants by spending"""
    try:
        receipts = await receipts_collection.find({
            "merchant_name": {"$exists": True, "$ne": None},
            "total": {"$exists": True, "$ne": None}
        }).to_list(None)
        
        merchant_groups = {}
        for receipt in receipts:
            merchant = receipt.get("merchant_name", "Unknown")
            total = receipt.get("total", 0) or 0
            
            if merchant not in merchant_groups:
                merchant_groups[merchant] = {
                    "total_spent": 0,
                    "receipt_count": 0,
                    "last_transaction": None
                }
            
            merchant_groups[merchant]["total_spent"] += total
            merchant_groups[merchant]["receipt_count"] += 1
            
            # Track latest transaction
            transaction_date = receipt.get("transaction_date")
            if transaction_date and (merchant_groups[merchant]["last_transaction"] is None or 
                                   transaction_date > merchant_groups[merchant]["last_transaction"]):
                merchant_groups[merchant]["last_transaction"] = transaction_date
        
        # Convert to list and sort by total spent
        top_merchants = []
        for merchant, data in merchant_groups.items():
            top_merchants.append({
                "merchant_name": merchant,
                "total_spent": data["total_spent"],
                "receipt_count": data["receipt_count"],
                "average_spent": data["total_spent"] / data["receipt_count"] if data["receipt_count"] > 0 else 0,
                "last_transaction": data["last_transaction"]
            })
        
        return sorted(top_merchants, key=lambda x: x["total_spent"], reverse=True)[:10]
    except Exception as e:
        logger.error(f"Error in get_top_merchants: {str(e)}")
        return []

async def get_discount_analysis() -> dict:
    """Get discount analysis"""
    try:
        # Get all items with discount information
        items = await receipt_items_collection.find({
            "has_discount": True
        }).to_list(None)
        
        total_discount = sum(item.get("discount_amount", 0) or 0 for item in items)
        total_items_with_discount = len(items)
        
        # Get total items for percentage calculation
        all_items = await receipt_items_collection.find({}).to_list(None)
        total_items = len(all_items)
        
        # Get receipts with discounts
        receipt_ids_with_discounts = list(set(item.get("receipt_id") for item in items))
        receipts_with_discounts = await receipts_collection.find({
            "_id": {"$in": receipt_ids_with_discounts}
        }).to_list(None)
        
        total_spent_with_discounts = sum(receipt.get("total", 0) or 0 for receipt in receipts_with_discounts)
        
        return {
            "total_discount_amount": total_discount,
            "total_items_with_discount": total_items_with_discount,
            "total_items": total_items,
            "discount_percentage": (total_items_with_discount / total_items * 100) if total_items > 0 else 0,
            "total_spent_with_discounts": total_spent_with_discounts,
            "average_discount_per_item": total_discount / total_items_with_discount if total_items_with_discount > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error in get_discount_analysis: {str(e)}")
        return {
            "total_discount_amount": 0,
            "total_items_with_discount": 0,
            "total_items": 0,
            "discount_percentage": 0,
            "total_spent_with_discounts": 0,
            "average_discount_per_item": 0
        }

async def get_recent_activity() -> List[dict]:
    """Get recent receipt activity"""
    try:
        receipts = await receipts_collection.find({}).sort("created_at", -1).limit(5).to_list(None)
        
        recent_activity = []
        for receipt in receipts:
            recent_activity.append({
                "id": receipt["_id"],
                "merchant_name": receipt.get("merchant_name", "Unknown"),
                "total": receipt.get("total", 0),
                "transaction_date": receipt.get("transaction_date"),
                "created_at": receipt["created_at"],
                "receipt_type": receipt.get("receipt_type", "Unknown")
            })
        
        return recent_activity
    except Exception as e:
        logger.error(f"Error in get_recent_activity: {str(e)}")
        return []

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

def process_item_pricing(item_data: dict) -> dict:
    """Process and validate item pricing, handle discounts and unit price calculations"""
    processed_item = item_data.copy()
    
    # Get values with defaults
    unit_price = processed_item.get("unit_price", 0)
    total_price = processed_item.get("total_price", 0)
    quantity = processed_item.get("quantity", 1)
    
    # Scenario 1: If unit price is 0.00 and quantity is 1, set unit price to total price
    if (unit_price == 0 or unit_price is None) and quantity == 1 and total_price > 0:
        processed_item["unit_price"] = total_price
        processed_item["has_discount"] = False
        processed_item["discount_amount"] = 0
    
    # Scenario 2: If unit price exists and is not 0, check for discount
    elif unit_price and unit_price > 0 and total_price > 0:
        expected_total = unit_price * quantity
        if abs(expected_total - total_price) > 0.01:  # Allow for small rounding differences
            discount_amount = expected_total - total_price
            processed_item["has_discount"] = True
            processed_item["discount_amount"] = discount_amount
        else:
            processed_item["has_discount"] = False
            processed_item["discount_amount"] = 0
    
    # Scenario 3: If we have total price but no unit price, calculate unit price
    elif total_price > 0 and quantity > 0 and (unit_price == 0 or unit_price is None):
        calculated_unit_price = total_price / quantity
        processed_item["unit_price"] = calculated_unit_price
        processed_item["has_discount"] = False
        processed_item["discount_amount"] = 0
    
    else:
        processed_item["has_discount"] = False
        processed_item["discount_amount"] = 0
    
    return processed_item

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
            
            # Extract unit price with multiple field name fallbacks
            unit_price_field = None
            for field_name in ["Price", "UnitPrice", "Unit_Price", "ItemPrice"]:
                if item.value_object.get(field_name):
                    unit_price_field = item.value_object[field_name]
                    break
            
            if unit_price_field:
                # Try different value types for unit price
                if hasattr(unit_price_field, "value_currency") and getattr(unit_price_field.value_currency, "amount", None) is not None:
                    item_data["unit_price"] = unit_price_field.value_currency.amount
                elif hasattr(unit_price_field, "value_number") and unit_price_field.value_number is not None:
                    item_data["unit_price"] = unit_price_field.value_number
                elif hasattr(unit_price_field, "value_string") and unit_price_field.value_string:
                    # Try to parse string as number
                    try:
                        item_data["unit_price"] = float(unit_price_field.value_string.replace("$", "").replace(",", ""))
                    except (ValueError, AttributeError):
                        pass
            
            # Extract total price with multiple field name fallbacks
            total_price_field = None
            for field_name in ["TotalPrice", "Total_Price", "ItemTotal", "LineTotal"]:
                if item.value_object.get(field_name):
                    total_price_field = item.value_object[field_name]
                    break
            
            if total_price_field:
                # Try different value types for total price
                if hasattr(total_price_field, "value_currency") and getattr(total_price_field.value_currency, "amount", None) is not None:
                    item_data["total_price"] = total_price_field.value_currency.amount
                elif hasattr(total_price_field, "value_number") and total_price_field.value_number is not None:
                    item_data["total_price"] = total_price_field.value_number
                elif hasattr(total_price_field, "value_string") and total_price_field.value_string:
                    # Try to parse string as number
                    try:
                        item_data["total_price"] = float(total_price_field.value_string.replace("$", "").replace(",", ""))
                    except (ValueError, AttributeError):
                        pass
            
            # Look for discount fields from Azure Document Intelligence
            discount_field = None
            for field_name in ["Discount", "DiscountAmount", "ItemDiscount", "Discount_Amount"]:
                if item.value_object.get(field_name):
                    discount_field = item.value_object[field_name]
                    break
            
            if discount_field:
                # Try different value types for discount
                if hasattr(discount_field, "value_currency") and getattr(discount_field.value_currency, "amount", None) is not None:
                    item_data["discount_amount"] = discount_field.value_currency.amount
                    item_data["has_discount"] = True
                elif hasattr(discount_field, "value_number") and discount_field.value_number is not None:
                    item_data["discount_amount"] = discount_field.value_number
                    item_data["has_discount"] = True
                elif hasattr(discount_field, "value_string") and discount_field.value_string:
                    # Try to parse string as number
                    try:
                        item_data["discount_amount"] = float(discount_field.value_string.replace("$", "").replace(",", ""))
                        item_data["has_discount"] = True
                    except (ValueError, AttributeError):
                        pass
            
            # Process the item pricing (handle unit price calculations and discount detection)
            processed_item = process_item_pricing(item_data)
            data["items"].append(processed_item)
    
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
                    "total_price": item.get("total_price"),
                    "has_discount": item.get("has_discount", False),
                    "discount_amount": item.get("discount_amount", 0)
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
                total_price=item.get("total_price"),
                has_discount=item.get("has_discount", False),
                discount_amount=item.get("discount_amount", 0)
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
            total_price=item.get("total_price"),
            has_discount=item.get("has_discount", False),
            discount_amount=item.get("discount_amount", 0)
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
                total_price=item.get("total_price"),
                has_discount=item.get("has_discount", False),
                discount_amount=item.get("discount_amount", 0)
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
        top_merchants = await get_top_merchants()
        discount_analysis = await get_discount_analysis()
        recent_activity = await get_recent_activity()
        
        return {
            "summary": summary,
            "monthly_spending": monthly,
            "receipt_types": receipt_types,
            "top_merchants": top_merchants,
            "discount_analysis": discount_analysis,
            "recent_activity": recent_activity,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Dashboard generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analytics dashboard"
        )


@app.get("/analytics/test_db")
async def test_database():
    '''Test database connection and collections'''
    try:
        # Test database connection
        await mongodb_client.admin.command('ping')
        
        # Test collections
        receipts_count = await receipts_collection.count_documents({})
        items_count = await receipt_items_collection.count_documents({})
        
        # Test new analytics functions
        top_merchants = await get_top_merchants()
        discount_analysis = await get_discount_analysis()
        recent_activity = await get_recent_activity()
        
        return {
            "status": "connected",
            "database": database_name,
            "receipts_count": receipts_count,
            "items_count": items_count,
            "collections": ["receipts", "receipt_items"],
            "new_analytics": {
                "top_merchants_count": len(top_merchants),
                "discount_analysis": discount_analysis,
                "recent_activity_count": len(recent_activity)
            }
        }
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database test failed: {str(e)}"
        )

@app.get("/analytics/drop_db")
async def drop_database():
    '''Drop the entire database'''
    try:
        await mongodb_client.drop_database(database_name)
        return {"message": f"Database '{database_name}' dropped successfully"}
    except Exception as e:
        logger.error(f"Failed to drop database: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to drop database: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
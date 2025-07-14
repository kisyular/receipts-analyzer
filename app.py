"""
Simplified FastAPI Receipt Analysis Application
Core operations: Upload, Read, Save, View, Delete documents
"""

import os
import uuid
import logging
from datetime import datetime
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
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI(
    title="Receipt Analysis API",
    description="Simple API for analyzing receipts using Azure Document Intelligence",
    version="1.0.0"
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

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./receipts.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Simplified Database Models
class Receipt(Base):
    __tablename__ = "receipts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    image_path = Column(String)  # Store path to saved image
    merchant_name = Column(String)
    transaction_date = Column(DateTime)
    total = Column(Float)
    confidence_score = Column(Float)
    raw_data = Column(Text)  # Store full Azure response as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

class ReceiptItem(Base):
    __tablename__ = "receipt_items"
    
    id = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(String, nullable=False)
    description = Column(String)
    quantity = Column(Float)
    unit_price = Column(Float)
    total_price = Column(Float)
    confidence = Column(Float)

# Create database tables
Base.metadata.create_all(bind=engine)

# Check if image_path column exists, if not add it (for existing databases)
def migrate_database():
    """Add image_path column to existing receipts table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            # Check if image_path column exists (PostgreSQL)
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'receipts' AND column_name = 'image_path'
            """))
            
            if not result.fetchone():
                conn.execute(text("ALTER TABLE receipts ADD COLUMN image_path TEXT"))
                conn.commit()
                logger.info("Added image_path column to receipts table")
            else:
                logger.info("image_path column already exists")
    except Exception as e:
        logger.warning(f"Database migration check failed: {e}")

# Run migration
migrate_database()

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

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
        "items": [],
        "confidence_scores": []
    }
    
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
    
    return data

# API Endpoints

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# 1. UPLOAD DOCUMENT
@app.post("/upload", response_model=ReceiptResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and analyze receipt document"""
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG, BMP, TIFF, HEIF)"
        )
    
    # Validate file size (max 10MB for better performance)
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size must be less than 10MB"
        )
    
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heif', '.heic'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Read file content once and store it
    file_content = file.file.read()
    
    # Save uploaded file for analysis
    file_path = save_upload_file_with_content(file_content, file.filename)
    
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
        
        # Generate a unique ID for the receipt
        receipt_id = str(uuid.uuid4())
        
        # Save receipt image permanently using the receipt ID
        image_path = save_receipt_image_with_content(file_content, file.filename, receipt_id)
        
        # Save to database
        receipt = Receipt(
            id=receipt_id,
            filename=file.filename,
            image_path=image_path,
            merchant_name=receipt_data["merchant_name"],
            transaction_date=receipt_data["transaction_date"],
            total=receipt_data["total"],
            confidence_score=receipt_data["confidence_score"],
            raw_data=str(receipts.documents[0].__dict__)
        )
        
        db.add(receipt)
        db.commit()
        db.refresh(receipt)
        
        # Save items
        for item_data in receipt_data["items"]:
            item = ReceiptItem(
                receipt_id=receipt.id,
                description=item_data.get("description"),
                quantity=item_data.get("quantity"),
                unit_price=item_data.get("unit_price"),
                total_price=item_data.get("total_price")
            )
            db.add(item)
        
        db.commit()
        
        # Get items for response
        items = db.query(ReceiptItem).filter(ReceiptItem.receipt_id == receipt.id).all()
        
        # Generate image URL
        image_url = f"/receipt_images/{receipt_id}{Path(file.filename).suffix}" if receipt.image_path else None
        
        return ReceiptResponse(
            id=receipt.id,
            filename=receipt.filename,
            image_url=image_url,
            merchant_name=receipt.merchant_name,
            transaction_date=receipt.transaction_date,
            total=receipt.total,
            confidence_score=receipt.confidence_score,
            items=[ReceiptItemResponse(
                description=item.description,
                quantity=item.quantity,
                unit_price=item.unit_price,
                total_price=item.total_price,
                confidence=item.confidence
            ) for item in items],
            created_at=receipt.created_at
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a camera-captured image and analyze as a receipt (calls upload_document)."""
    # Call the existing upload_document function
    return await upload_document(file=file, db=db)

# 2. READ DOCUMENT (Get specific receipt)
@app.get("/documents/{document_id}", response_model=ReceiptResponse)
async def read_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Read specific document by ID"""
    receipt = db.query(Receipt).filter(Receipt.id == document_id).first()
    if not receipt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get items for this receipt
    items = db.query(ReceiptItem).filter(ReceiptItem.receipt_id == receipt.id).all()
    
    # Generate image URL
    image_url = f"/receipt_images/{receipt.id}{Path(receipt.filename).suffix}" if receipt.image_path else None
    
    return ReceiptResponse(
        id=receipt.id,
        filename=receipt.filename,
        image_url=image_url,
        merchant_name=receipt.merchant_name,
        transaction_date=receipt.transaction_date,
        total=receipt.total,
        confidence_score=receipt.confidence_score,
        items=[ReceiptItemResponse(
            description=item.description,
            quantity=item.quantity,
            unit_price=item.unit_price,
            total_price=item.total_price,
            confidence=item.confidence
        ) for item in items],
        created_at=receipt.created_at
    )

# 3. SAVE DOCUMENT (This is handled in upload, but keeping for clarity)
@app.post("/documents", response_model=ReceiptResponse)
async def save_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Save document to database (same as upload)"""
    return await upload_document(file, db)

# 4. VIEW DOCUMENTS (List all documents)
@app.get("/documents", response_model=List[ReceiptResponse])
async def view_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """View all documents with pagination"""
    receipts = db.query(Receipt).offset(skip).limit(limit).all()
    
    result = []
    for receipt in receipts:
        # Get items for this receipt
        items = db.query(ReceiptItem).filter(ReceiptItem.receipt_id == receipt.id).all()
        
        # Generate image URL
        image_url = f"/receipt_images/{receipt.id}{Path(receipt.filename).suffix}" if receipt.image_path else None
        
        result.append(ReceiptResponse(
            id=receipt.id,
            filename=receipt.filename,
            image_url=image_url,
            merchant_name=receipt.merchant_name,
            transaction_date=receipt.transaction_date,
            total=receipt.total,
            confidence_score=receipt.confidence_score,
            items=[ReceiptItemResponse(
                description=item.description,
                quantity=item.quantity,
                unit_price=item.unit_price,
                total_price=item.total_price,
                confidence=item.confidence
            ) for item in items],
            created_at=receipt.created_at
        ))
    
    return result

# 5. DELETE DOCUMENT
@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Delete document by ID"""
    receipt = db.query(Receipt).filter(Receipt.id == document_id).first()
    if not receipt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete associated items first
    db.query(ReceiptItem).filter(ReceiptItem.receipt_id == document_id).delete()
    
    # Delete receipt image if it exists
    if receipt.image_path and os.path.exists(receipt.image_path):
        try:
            os.remove(receipt.image_path)
        except Exception as e:
            logger.warning(f"Failed to delete receipt image {receipt.image_path}: {e}")
    
    db.delete(receipt)
    db.commit()
    
    return {"message": "Document deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
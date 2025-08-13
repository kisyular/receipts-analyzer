# Receipts Analyzer - Complete Guide

analyzes receipts using AI. Simply upload a photo of any receipt and get detailed spending insights, analytics, and item breakdowns.

## 🎯 What This App Does

- **📸 Upload Receipts**: Take a photo or upload any receipt image
- **🤖 AI Analysis**: Automatically extracts merchant name, date, amounts, and individual items
- **📊 Smart Analytics**: Shows spending trends, top merchants, and discount analysis
- **💾 Save Everything**: Stores all your receipts for future reference
- **📱 Works Everywhere**: Use on phone, tablet, or computer

## 🚀 Quick Start (5 Minutes)

### Step 1: Extract the Files

1. Download the zip file you received via email
2. Extract it to a folder on your computer (e.g., `C:\receipts-analyzer` or `/home/user/receipts-analyzer`)
3. Open a command prompt/terminal in that folder

**After extraction, your folder should look like this:**

```
receipts-analyzer/
├── app.py               # Main application file
├── startup.py           # Startup script
├── requirements.txt     # Python dependencies
├── README.md            # This documentation file
├── index.html           # Main web interface
├── analytics.html       # Analytics dashboard
├── test_endpoints.py    # Test suite
├── TEST_README.md       # Test documentation
├── azure.yaml           # Azure configuration
├── receipt_images/      # Folder for storing receipt images
└── uploads/             # Temporary upload folder
```

**Important:** Make sure you see all these files after extraction. If any are missing, the zip file may be corrupted.

### Step 2: Install Dependencies

In your command prompt/terminal, run:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Your Accounts

#### MongoDB Database (Free)

1. Go to https://cloud.mongodb.com
2. Click "Try Free" and create an account
3. Create a new cluster (choose the free option)
4. Click "Connect" → "Connect your application"
5. Copy the connection string (looks like: `mongodb+srv://username:password@cluster.mongodb.net/`)

#### Azure Document Intelligence (Free Trial)

1. Go to https://portal.azure.com
2. Create a free account if you don't have one
3. Search for "Document Intelligence" in the search bar
4. Click "Create" → "Document Intelligence"
5. Fill in the details and create the resource
6. Go to "Keys and Endpoint" and copy:
   - Endpoint URL (looks like: `https://your-resource.cognitiveservices.azure.com/`)
   - Key 1 (long string of letters and numbers)

### Step 4: Create Environment File

In your project folder, create a file named `.env` with this content:

```env
# Replace with your MongoDB connection string
DATABASE_URL=mongodb+srv://username:password@cluster.mongodb.net/receipts_db

# Replace with your Azure endpoint
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Replace with your Azure key
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-actual-key-here
```

### Step 5: Run the Application

```bash
python startup.py
```

### Step 6: Open in Browser

Go to: `http://localhost:8000`

🎉 **You're ready to analyze receipts!**

## 📱 How to Use the App

### Upload Your First Receipt

1. **Open the app** in your browser: `http://localhost:8000`
2. **Choose how to upload**:
   - **Drag & Drop**: Drag a receipt image onto the upload area
   - **Choose File**: Click "Choose File" and select an image
   - **Take Photo**: Click "Take Photo" to use your camera
3. **Wait for analysis** (2-5 seconds)
4. **View results**: See merchant name, date, amounts, and item breakdown

### View Your Analytics

1. Click **"Analytics"** in the top navigation
2. See your spending insights:
   - Total spent and number of receipts
   - Monthly spending trends
   - Top merchants you shop at
   - Discount analysis
   - Recent activity

### Manage Your Receipts

1. Click **"Saved Receipts"** to see all uploaded receipts
2. Click **"Details"** on any receipt to see full information
3. Click **"Delete"** to remove receipts you don't need

## 🔌 API Reference (For Developers)

The app provides a REST API for programmatic access. All endpoints return JSON data.

### Base URL

```
http://localhost:8000
```

### Health Check

Check if the app is running properly:

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": "connected",
  "azure_document_intelligence": "configured"
}
```

## 💻 Programming Examples

### Python Example

```python
import requests

# Upload a receipt
with open('receipt.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/upload', files=files)
    receipt = response.json()
    print(f"Uploaded receipt from {receipt['merchant_name']} for ${receipt['total']}")

# Get analytics
analytics = requests.get('http://localhost:8000/analytics/dashboard').json()
print(f"Total spent: ${analytics['summary']['total_spent']}")
```

## 🧪 Testing the App

### Run the Test Suite

```bash
python test_endpoints.py
```

This will test all features and show you if everything is working correctly.

### Manual Testing

1. **Health Check**: Visit `http://localhost:8000/health`
2. **Upload Test**: Try uploading any receipt image
3. **Analytics Test**: Check the analytics dashboard
4. **Error Handling**: Try uploading non-image files

## 🔧 Configuration

### Environment Variables

Create a `.env` file in your project folder:

```env
# Required: MongoDB connection string
DATABASE_URL=mongodb+srv://username:password@cluster.mongodb.net/receipts_db

# Required: Azure Document Intelligence endpoint
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Required: Azure Document Intelligence key
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-api-key-here
```

### App Settings

- **File size limit**: 10MB maximum
- **Supported formats**: JPG, PNG, BMP, TIFF, HEIF
- **Image storage**: Local `receipt_images/` folder
- **Database**: MongoDB collections created automatically

## 🚨 Troubleshooting

### Common Problems and Solutions

#### 1. "Failed to connect to MongoDB"

**Problem**: Can't connect to database
**Solutions**:

- Check your `DATABASE_URL` in the `.env` file
- Make sure your MongoDB Atlas cluster is running
- Verify your IP address is whitelisted in MongoDB Atlas

#### 2. "Azure Document Intelligence error"

**Problem**: AI analysis not working
**Solutions**:

- Check your Azure credentials in the `.env` file
- Make sure your Azure Document Intelligence resource is active
- Verify you have enough credits in your Azure account

#### 3. "File must be an image"

**Problem**: Upload rejected
**Solutions**:

- Use only image files (JPG, PNG, BMP, TIFF, HEIF)
- Make sure the file is actually an image, not a PDF or document
- Check file size is under 10MB

#### 4. "Address already in use"

**Problem**: Port 8000 is busy
**Solutions**:

- Close other applications using port 8000
- Or change the port in `startup.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8001)  # Change 8000 to 8001
  ```

#### 5. "Module not found" errors

**Problem**: Missing Python packages
**Solutions**:

- Run: `pip install -r requirements.txt`
- Make sure you're in the correct folder
- Try: `pip install --upgrade pip` first

### Debug Mode

To see detailed error messages, edit `app.py` and change:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Check Logs

Look for error messages in your command prompt/terminal where you ran the app.

## 📈 Performance Tips

### For Better Receipt Analysis

- **Clear images**: Well-lit, focused photos work best
- **Good angles**: Take photos straight-on, not at angles
- **High resolution**: Use the highest quality your camera supports
- **Avoid shadows**: Make sure text is clearly visible

### For Faster Processing

- **Smaller files**: Keep images under 5MB for faster uploads
- **Good internet**: Faster upload speeds help
- **Close other apps**: Free up memory and CPU

## 🔒 Security Notes

- **Local only**: The app runs on your computer, data stays local
- **No internet required**: Once set up, works offline (except for AI analysis)
- **Secure storage**: Receipt images stored locally in `receipt_images/` folder
- **No data sharing**: Your data never leaves your computer

## 📱 Mobile Usage

### On Your Phone

1. **Find your computer's IP address**:

   - Windows: Run `ipconfig` in Command Prompt
   - Mac/Linux: Run `ifconfig` in Terminal
   - Look for something like `192.168.1.100`

2. **Access from phone**:
   - Make sure phone and computer are on same WiFi
   - Open browser on phone
   - Go to: `http://192.168.1.100:8000` (replace with your computer's IP)

### Camera Upload

- Use the "Take Photo" button for best results
- Hold phone steady and well-lit
- Make sure receipt text is clear and readable

## 🔄 Maintenance

### Regular Tasks

- **Backup data**: Copy the `receipt_images/` folder to a safe location
- **Clean up**: Delete old receipts you don't need
- **Update Python**: Keep Python updated for security

### Database Backup

Your data is stored in MongoDB Atlas, which has automatic backups. For extra safety:

1. Go to your MongoDB Atlas dashboard
2. Click "Backup" in the left menu
3. Create manual backups when needed

## 🆘 Getting Help

### Before Asking for Help

1. **Check this guide**: Re-read the troubleshooting section
2. **Test the app**: Run `python test_endpoints.py`
3. **Check logs**: Look for error messages in your terminal
4. **Verify setup**: Make sure all accounts and credentials are correct

### Common Issues Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] MongoDB Atlas account created and cluster running
- [ ] Azure Document Intelligence resource created
- [ ] `.env` file created with correct credentials
- [ ] No other apps using port 8000
- [ ] Receipt images are clear and readable

### Where to Get Help

- **Check the logs**: Error messages in your terminal
- **Test endpoints**: Use the test script to verify functionality
- **Health check**: Visit `http://localhost:8000/health`
- **Documentation**: Re-read this guide

## 🎉 You're All Set!

Your receipt analyzer is now ready to use! Here's what you can do:

1. **Upload receipts** using drag & drop, file picker, or camera
2. **View detailed analysis** of each receipt
3. **See spending trends** in the analytics dashboard
4. **Manage your collection** of saved receipts
5. **Use the API** for custom integrations

The app will automatically:

- Extract merchant names, dates, and amounts
- Identify individual items and prices
- Detect discounts and savings
- Calculate spending trends
- Store everything securely

Happy receipt analyzing! 📊✨

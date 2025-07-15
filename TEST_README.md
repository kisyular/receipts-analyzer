# Receipt Analyzer API Test Suite

This comprehensive test script validates all endpoints of your Receipt Analyzer API, including upload, read, list, delete, and analytics functionality.

## Features

- ✅ **Complete API Coverage**: Tests all endpoints including CRUD operations and analytics
- 🎨 **Colored Output**: Beautiful terminal output with timestamps and status indicators
- 📊 **Performance Metrics**: Response time tracking and performance analysis
- 🧹 **Auto Cleanup**: Automatically deletes test data after testing
- ⚠️ **Error Handling**: Tests both success and error scenarios
- 📈 **Detailed Reports**: Comprehensive test results with success rates

## Prerequisites

1. **Running API Server**: Make sure your FastAPI server is running on `http://localhost:8000`
2. **Test Image**: Provide a receipt image for upload testing (optional but recommended)

## Installation

1. Install test dependencies:

```bash
pip install -r test_requirements.txt
```

2. (Optional) Prepare a test receipt image:
   - Find any receipt image (JPEG, PNG, etc.)
   - Rename it to `test_receipt.jpg` and place it in the project root
   - Or modify `TEST_IMAGE_PATH` in the script to point to your image

## Usage

### Basic Test Run

```bash
python test_endpoints.py
```

### Test with Custom Server URL

Edit the `BASE_URL` variable in `test_endpoints.py`:

```python
BASE_URL = "http://your-server:8000"
```

## What Gets Tested

### 🔍 Health Check

- Server connectivity
- Database status
- Azure Document Intelligence status

### 📋 Core Operations

- **Upload Receipt**: Tests file upload and analysis
- **Get Receipt**: Retrieves specific receipt by ID
- **List Receipts**: Gets all receipts with pagination
- **Delete Receipt**: Removes receipt and associated data

### 📊 Analytics Endpoints

- **Spending Summary**: Overall spending statistics
- **Monthly Analytics**: Spending trends by month
- **Receipt Type Analysis**: Spending by receipt type
- **Spending Prediction**: Next month spending forecast
- **Item Analysis**: Most frequently purchased items
- **Analytics Dashboard**: Comprehensive dashboard data

### ⚠️ Error Handling

- Non-existent receipt retrieval
- Invalid file uploads
- Connection errors

## Test Output Example

```
🚀 Starting Receipt Analyzer API Tests
Base URL: http://localhost:8000
============================================================

[14:30:15] Testing Health Check...
[14:30:15]   Health Check: 200 OK
[14:30:15]   Database: connected
[14:30:15]   Azure: configured

[14:30:16] Testing Upload Receipt...
[14:30:18]   Upload Receipt: 200 OK
[14:30:18]   Uploaded receipt ID: 12345678-1234-1234-1234-123456789abc
[14:30:18]   Merchant: Walmart
[14:30:18]   Total: $45.67
[14:30:18]   Confidence: 95.2%

[14:30:19] Testing Analytics Summary...
[14:30:19]   Analytics Summary: 200 OK
[14:30:19]   Total spent: $1,234.56
[14:30:19]   Total receipts: 25
[14:30:19]   Average per receipt: $49.38

📈 Test Results Summary
============================================================
Total Tests: 15
Passed: 15
Failed: 0
Success Rate: 100.0%
Average Response Time: 0.847s

⚡ Performance Summary:
  ✅ Health Check: 0.023s
  ✅ Upload Receipt: 2.156s
  ✅ Analytics Summary: 0.045s
  ✅ Monthly Analytics: 0.034s

🎉 Test Suite Complete!
```

## Configuration Options

### Environment Variables

The test script uses the same environment variables as your main application:

- `DATABASE_URL`: MongoDB connection string
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`: Azure endpoint
- `AZURE_DOCUMENT_INTELLIGENCE_KEY`: Azure API key

### Customization

You can modify these variables in `test_endpoints.py`:

- `BASE_URL`: API server URL (default: `http://localhost:8000`)
- `TEST_IMAGE_PATH`: Path to test receipt image (default: `test_receipt.jpg`)

## Troubleshooting

### Common Issues

1. **Connection Refused**

   - Ensure your FastAPI server is running
   - Check if the port is correct (default: 8000)
   - Verify firewall settings

2. **Upload Test Skipped**

   - Provide a test receipt image named `test_receipt.jpg`
   - Or modify `TEST_IMAGE_PATH` in the script

3. **MongoDB Connection Errors**

   - Verify your `DATABASE_URL` environment variable
   - Check MongoDB server status
   - Ensure network connectivity

4. **Azure API Errors**
   - Verify Azure credentials in environment variables
   - Check Azure service status
   - Ensure proper endpoint URL

### Debug Mode

For detailed debugging, you can add logging to the script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with CI/CD

You can integrate this test script into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run API Tests
  run: |
    pip install -r test_requirements.txt
    python test_endpoints.py
```

## Performance Benchmarks

The test script tracks performance metrics:

- **Fast**: < 1.0s response time
- **Acceptable**: 1.0s - 3.0s response time
- **Slow**: > 3.0s response time

## Contributing

To add new tests:

1. Add a new test method to the `APITester` class
2. Call it in the `run_all_tests()` method
3. Update this README with new test descriptions

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Review the test output for specific error messages
4. Ensure your API server is running and accessible

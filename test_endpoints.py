#!/usr/bin/env python3
"""
Comprehensive Test Script for Receipt Analyzer API
Tests all endpoints including upload, read, list, delete, and analytics
"""

import requests
import json
import time
import os
from datetime import datetime
from pathlib import Path
import sys

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "images/test_receipt.png"  # You'll need to provide a test receipt image

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.uploaded_receipt_ids = []
        
    def log(self, message: str, color: str = Colors.END):
        """Print colored log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] {message}{Colors.END}")
    
    def test_endpoint(self, name: str, method: str, endpoint: str, expected_status: int = 200, **kwargs):
        """Test a single endpoint and record results"""
        self.log(f"Testing {name}...", Colors.BLUE)
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            
            success = response.status_code == expected_status
            status_color = Colors.GREEN if success else Colors.RED
            
            self.log(f"  {name}: {response.status_code} {response.reason}", status_color)
            
            if not success:
                self.log(f"  Expected: {expected_status}, Got: {response.status_code}", Colors.YELLOW)
                if response.text:
                    try:
                        error_detail = response.json()
                        self.log(f"  Error: {error_detail.get('detail', 'Unknown error')}", Colors.YELLOW)
                    except:
                        self.log(f"  Error: {response.text[:200]}", Colors.YELLOW)
            
            # Store result
            self.test_results.append({
                "name": name,
                "method": method,
                "endpoint": endpoint,
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "success": success,
                "response_time": response.elapsed.total_seconds(),
                "response_size": len(response.content) if response.content else 0
            })
            
            return response if success else None
            
        except requests.exceptions.RequestException as e:
            self.log(f"  {name}: Connection error - {str(e)}", Colors.RED)
            self.test_results.append({
                "name": name,
                "method": method,
                "endpoint": endpoint,
                "expected_status": expected_status,
                "actual_status": None,
                "success": False,
                "error": str(e)
            })
            return None
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.test_endpoint(
            "Health Check",
            "GET",
            "/health",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Database: {data.get('database', 'Unknown')}", Colors.CYAN)
                self.log(f"  Azure: {data.get('azure_document_intelligence', 'Unknown')}", Colors.CYAN)
            except:
                pass
    
    def test_upload_receipt(self):
        """Test receipt upload endpoint"""
        if not os.path.exists(TEST_IMAGE_PATH):
            self.log(f"Test image not found: {TEST_IMAGE_PATH}", Colors.YELLOW)
            self.log("Skipping upload test - please provide a test receipt image", Colors.YELLOW)
            return None
        
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': (TEST_IMAGE_PATH, f, 'image/jpeg')}
            response = self.test_endpoint(
                "Upload Receipt",
                "POST",
                "/upload",
                200,
                files=files
            )
        
        if response:
            try:
                data = response.json()
                receipt_id = data.get('id')
                if receipt_id:
                    self.uploaded_receipt_ids.append(receipt_id)
                    self.log(f"  Uploaded receipt ID: {receipt_id}", Colors.GREEN)
                    self.log(f"  Merchant: {data.get('merchant_name', 'Unknown')}", Colors.CYAN)
                    self.log(f"  Total: ${data.get('total', 0):.2f}", Colors.CYAN)
                    self.log(f"  Confidence: {data.get('confidence_score', 0):.2%}", Colors.CYAN)
                return receipt_id
            except:
                pass
        
        return None
    
    def test_get_receipt(self, receipt_id: str):
        """Test getting a specific receipt"""
        response = self.test_endpoint(
            f"Get Receipt {receipt_id[:8]}...",
            "GET",
            f"/documents/{receipt_id}",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Retrieved receipt: {data.get('merchant_name', 'Unknown')}", Colors.CYAN)
                self.log(f"  Items count: {len(data.get('items', []))}", Colors.CYAN)
            except:
                pass
    
    def test_list_receipts(self):
        """Test listing all receipts"""
        response = self.test_endpoint(
            "List All Receipts",
            "GET",
            "/documents",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Total receipts: {len(data)}", Colors.CYAN)
                if data:
                    latest = data[0]
                    self.log(f"  Latest: {latest.get('merchant_name', 'Unknown')} - ${latest.get('total', 0):.2f}", Colors.CYAN)
            except:
                pass
    
    def test_analytics_summary(self):
        """Test analytics summary endpoint"""
        response = self.test_endpoint(
            "Analytics Summary",
            "GET",
            "/analytics/summary",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Total spent: ${data.get('total_spent', 0):.2f}", Colors.CYAN)
                self.log(f"  Total receipts: {data.get('total_receipts', 0)}", Colors.CYAN)
                self.log(f"  Average per receipt: ${data.get('average_per_receipt', 0):.2f}", Colors.CYAN)
            except:
                pass
    
    def test_analytics_monthly(self):
        """Test monthly analytics endpoint"""
        response = self.test_endpoint(
            "Monthly Analytics",
            "GET",
            "/analytics/monthly",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Months with data: {len(data)}", Colors.CYAN)
                if data:
                    latest_month = data[-1]
                    self.log(f"  Latest month: {latest_month.get('month')} {latest_month.get('year')} - ${latest_month.get('total_spent', 0):.2f}", Colors.CYAN)
            except:
                pass
    
    def test_analytics_receipt_types(self):
        """Test receipt type analytics endpoint"""
        response = self.test_endpoint(
            "Receipt Type Analytics",
            "GET",
            "/analytics/receipt-types",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Receipt types: {len(data)}", Colors.CYAN)
                if data:
                    top_type = data[0]
                    self.log(f"  Top type: {top_type.get('receipt_type')} - ${top_type.get('total_spent', 0):.2f}", Colors.CYAN)
            except:
                pass
    

    
    def test_analytics_dashboard(self):
        """Test comprehensive analytics dashboard endpoint"""
        response = self.test_endpoint(
            "Analytics Dashboard",
            "GET",
            "/analytics/dashboard",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Dashboard generated successfully", Colors.GREEN)
                self.log(f"  Generated at: {data.get('generated_at', 'Unknown')}", Colors.CYAN)
                self.log(f"  Summary data: ${data.get('summary', {}).get('total_spent', 0):.2f} total spent", Colors.CYAN)
            except:
                pass
    
    def test_delete_receipt(self, receipt_id: str):
        """Test deleting a receipt"""
        response = self.test_endpoint(
            f"Delete Receipt {receipt_id[:8]}...",
            "DELETE",
            f"/documents/{receipt_id}",
            200
        )
        
        if response:
            try:
                data = response.json()
                self.log(f"  Deleted successfully: {data.get('message', 'Unknown')}", Colors.GREEN)
            except:
                pass
    
    def test_invalid_endpoints(self):
        """Test invalid endpoints for error handling"""
        # Test non-existent receipt
        self.test_endpoint(
            "Get Non-existent Receipt",
            "GET",
            "/documents/non-existent-id",
            404
        )
        
        # Test delete non-existent receipt
        self.test_endpoint(
            "Delete Non-existent Receipt",
            "DELETE",
            "/documents/non-existent-id",
            404
        )
        
        # Test invalid file upload
        invalid_file = b"not an image"
        files = {'file': ('test.txt', invalid_file, 'text/plain')}
        self.test_endpoint(
            "Upload Invalid File",
            "POST",
            "/upload",
            400,
            files=files
        )
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        self.log("🚀 Starting Receipt Analyzer API Tests", Colors.BOLD + Colors.PURPLE)
        self.log(f"Base URL: {self.base_url}", Colors.CYAN)
        self.log("=" * 60, Colors.CYAN)
        
        # Test health check first
        self.test_health_check()
        self.log("", Colors.END)
        
        # Test core CRUD operations
        self.log("📋 Testing Core Operations", Colors.BOLD + Colors.BLUE)
        receipt_id = self.test_upload_receipt()
        self.log("", Colors.END)
        
        if receipt_id:
            self.test_get_receipt(receipt_id)
            self.log("", Colors.END)
        
        self.test_list_receipts()
        self.log("", Colors.END)
        
        # Test analytics endpoints
        self.log("📊 Testing Analytics Endpoints", Colors.BOLD + Colors.BLUE)
        self.test_analytics_summary()
        self.log("", Colors.END)
        
        self.test_analytics_monthly()
        self.log("", Colors.END)
        
        self.test_analytics_receipt_types()
        self.log("", Colors.END)
        
        self.test_analytics_dashboard()
        self.log("", Colors.END)
        
        # Test error handling
        self.log("⚠️  Testing Error Handling", Colors.BOLD + Colors.BLUE)
        self.test_invalid_endpoints()
        self.log("", Colors.END)
        
        # Clean up - delete uploaded receipts
        if self.uploaded_receipt_ids:
            self.log("🧹 Cleaning Up Test Data", Colors.BOLD + Colors.BLUE)
            for receipt_id in self.uploaded_receipt_ids:
                self.test_delete_receipt(receipt_id)
            self.log("", Colors.END)
        
        # Generate test report
        self.generate_report()
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        self.log("📈 Test Results Summary", Colors.BOLD + Colors.PURPLE)
        self.log("=" * 60, Colors.CYAN)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        # Calculate average response time
        response_times = [r['response_time'] for r in self.test_results if 'response_time' in r and r['response_time'] is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Print summary
        self.log(f"Total Tests: {total_tests}", Colors.CYAN)
        self.log(f"Passed: {passed_tests}", Colors.GREEN)
        self.log(f"Failed: {failed_tests}", Colors.RED)
        self.log(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%", Colors.CYAN)
        self.log(f"Average Response Time: {avg_response_time:.3f}s", Colors.CYAN)
        
        # Print failed tests
        if failed_tests > 0:
            self.log("\n❌ Failed Tests:", Colors.BOLD + Colors.RED)
            for result in self.test_results:
                if not result['success']:
                    self.log(f"  - {result['name']}: {result.get('error', f'Status {result.get('actual_status')}')}", Colors.RED)
        
        # Print performance summary
        self.log("\n⚡ Performance Summary:", Colors.BOLD + Colors.CYAN)
        for result in self.test_results:
            if result['success'] and 'response_time' in result:
                status_icon = "✅" if result['response_time'] < 1.0 else "⚠️"
                self.log(f"  {status_icon} {result['name']}: {result['response_time']:.3f}s", Colors.CYAN)
        
        self.log("\n🎉 Test Suite Complete!", Colors.BOLD + Colors.GREEN)

def main():
    """Main function to run the test suite"""
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"{Colors.YELLOW}Warning: Test image '{TEST_IMAGE_PATH}' not found.{Colors.END}")
        print(f"{Colors.YELLOW}Upload tests will be skipped. Please provide a test receipt image.{Colors.END}")
        print(f"{Colors.CYAN}You can use any receipt image and rename it to '{TEST_IMAGE_PATH}'{Colors.END}")
        print()
    
    # Create tester and run tests
    tester = APITester(BASE_URL)
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {str(e)}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
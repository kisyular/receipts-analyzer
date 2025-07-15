#!/bin/bash
# Startup script for Azure App Service with MongoDB/Cosmos DB

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI application
uvicorn app:app --host 0.0.0.0 --port 8000 
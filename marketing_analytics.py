# marketing_analytics.py - FIXED VERSION
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import dspy
import os
from datetime import datetime, timedelta
import logging
from google.oauth2 import service_account

# Separate database setup instead of importing from main
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Only import these if available (so app doesn't crash if missing)
try:
    from google.cloud import bigquery
    import pandas as pd
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    bigquery = None
    pd = None

# Database setup - reuse the same DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
else:
    engine = None
    SessionLocal = None
    Base = None

# Database Models for Marketing Analytics (only if database available)
if Base is not None:
    class TrendAnalysis(Base):
        __tablename__ = "trend_analysis"
        
        id = Column(Integer, primary_key=True, index=True)
        analysis_date = Column(DateTime, default=datetime.utcnow)
        utm_medium_group = Column(String)
        trend_direction = Column(String)
        key_insights = Column(JSON)
        performance_metrics = Column(JSON)
        recommendations = Column(Text)
        confidence_score = Column(Float)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    # Create marketing tables if database is available
    try:
        Base.metadata.create_all(bind=engine)
    except:
        pass  # Ignore if tables already exist or DB not available

# BigQuery setup (only if available)
PROJECT_ID = "gtm-p3gj3zzk-nthlo"
DATASET_ID = "last_14_days_analysis"

if BIGQUERY_AVAILABLE:
    try:
        import json
        from google.oauth2 import service_account
        
        # Try to get credentials from environment
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            bigquery_client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        else:
            # Fallback to default credentials
            bigquery_client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"BigQuery client initialization failed: {e}")
        bigquery_client = None
else:
    bigquery_client = None

# Database dependency
def get_db():
    if SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None

# Pydantic Models
class TrendAnalysisRequest(BaseModel):
    date_range: Dict[str, str]
    include_historical: bool = True
    analysis_depth: str = "standard"

class TrendAnalysisResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Create router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API",
        "version": "1.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE,
        "database_available": SessionLocal is not None,
        "endpoints": [
            "/analyze-trends - Main trend analysis",
            "/trend-history - Historical trend data",
            "/test-bigquery - Test BigQuery connection",
            "/status - System status check"
        ]
    }

@marketing_router.get("/status")
async def system_status():
    """Check system status and dependencies"""
    return {
        "bigquery_available": BIGQUERY_AVAILABLE,
        "bigquery_client_ready": bigquery_client is not None,
        "database_available": SessionLocal is not None,
        "pandas_available": pd is not None,
        "project_id": PROJECT_ID,
        "dataset_id": DATASET_ID,
        "dependencies": {
            "google-cloud-bigquery": BIGQUERY_AVAILABLE,
            "pandas": pd is not None,
            "sqlalchemy": SessionLocal is not None
        }
    }

@marketing_router.get("/test-bigquery")
async def test_bigquery_connection():
    """Test BigQuery connection"""
    if not BIGQUERY_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="BigQuery dependencies not installed. Add 'google-cloud-bigquery' and 'pandas' to requirements.txt"
        )
    
    if not bigquery_client:
        raise HTTPException(
            status_code=503,
            detail="BigQuery client not initialized. Check GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable"
        )
    
    try:
        # Simple test query
        test_query = f"""
        SELECT COUNT(*) as total_records
        FROM `{PROJECT_ID}.{DATASET_ID}.hex_data`
        LIMIT 1
        """
        result = bigquery_client.query(test_query).to_dataframe()
        
        return {
            "status": "success",
            "bigquery_connection": "working",
            "test_query_result": int(result.iloc[0]['total_records']),
            "project_id": PROJECT_ID,
            "dataset_id": DATASET_ID
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery test failed: {str(e)}")

@marketing_router.post("/analyze-trends", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(request: TrendAnalysisRequest):
    """Analyze marketing trends (basic version until BigQuery is set up)"""
    try:
        since_date = request.date_range['since']
        until_date = request.date_range['until']
        
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            # Return mock analysis
            mock_data = {
                'paid-social': {
                    'total_spend': 1500.0,
                    'total_leads': 25,
                    'avg_cpa': 60.0
                },
                'paid-search-video': {
                    'total_spend': 800.0,
                    'total_leads': 12,
                    'avg_cpa': 66.7
                }
            }
            
            return TrendAnalysisResponse(
                status="success_mock",
                message="Analysis completed using mock data (BigQuery not configured)",
                data={
                    "medium_insights": [
                        {
                            "medium_group": medium,
                            "metrics": metrics,
                            "trend_direction": "stable",
                            "confidence_score": 0.7
                        }
                        for medium, metrics in mock_data.items()
                    ],
                    "analysis_metadata": {
                        "analysis_timestamp": datetime.utcnow().isoformat(),
                        "date_range": f"{since_date} to {until_date}",
                        "data_source": "mock",
                        "note": "Set up BigQuery credentials for real data analysis"
                    }
                }
            )
        
        # Real BigQuery analysis would go here
        return TrendAnalysisResponse(
            status="success",
            message="BigQuery analysis not yet implemented",
            data={"note": "Real analysis coming soon"}
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="Analysis failed",
            error=str(e)
        )

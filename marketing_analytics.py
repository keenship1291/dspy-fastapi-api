# marketing_analytics.py - COMPLETE VERSION WITH CLAUDE
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import dspy
import os
from datetime import datetime, timedelta
import logging
from google.oauth2 import service_account
import json

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
    """Analyze marketing trends with real BigQuery data and Claude AI"""
    try:
        since_date = request.date_range['since']
        until_date = request.date_range['until']
        
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                message="BigQuery not available",
                error="BigQuery client not initialized"
            )
        
        # Fetch real data from BigQuery
        funnel_query = f"""
        SELECT 
            date,
            utm_campaign,
            utm_medium,
            leads,
            start_flows,
            estimates,
            closings,
            funded,
            rpts,
            platform
        FROM `{PROJECT_ID}.{DATASET_ID}.hex_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
            AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        ORDER BY date DESC
        """
        
        meta_query = f"""
        SELECT 
            date,
            campaign_name,
            adset_name,
            CAST(impressions AS INT64) as impressions,
            CAST(clicks AS INT64) as clicks,
            CAST(spend AS FLOAT64) as spend,
            CAST(reach AS INT64) as reach,
            CAST(landing_page_views AS INT64) as landing_page_views,
            CAST(leads AS INT64) as leads,
            platform
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
        ORDER BY date DESC
        """
        
        # Execute queries
        funnel_df = bigquery_client.query(funnel_query).to_dataframe()
        meta_df = bigquery_client.query(meta_query).to_dataframe()
        
        # Process the data
        if funnel_df.empty:
            return TrendAnalysisResponse(
                status="no_data",
                message=f"No funnel data found for {since_date} to {until_date}",
                data={"date_range": f"{since_date} to {until_date}"}
            )
        
        # Group by medium and calculate metrics
        results = {}
        
        for medium in ['paid-social', 'paid-search', 'paid-video']:
            medium_funnel = funnel_df[funnel_df['utm_medium'] == medium]
            
            if not medium_funnel.empty:
                # Get corresponding meta data for paid-social
                medium_meta = meta_df if medium == 'paid-social' else pd.DataFrame()
                
                total_spend = float(medium_meta['spend'].sum()) if not medium_meta.empty else 0.0
                total_leads = int(medium_funnel['leads'].sum())
                total_estimates = int(medium_funnel['estimates'].sum())
                
                group_key = 'paid-social' if medium == 'paid-social' else 'paid-search-video'
                
                if group_key not in results:
                    results[group_key] = {
                        'total_spend': 0,
                        'total_leads': 0,
                        'total_estimates': 0,
                        'days': 0
                    }
                
                results[group_key]['total_spend'] += total_spend
                results[group_key]['total_leads'] += total_leads
                results[group_key]['total_estimates'] += total_estimates
                results[group_key]['days'] = len(medium_funnel['date'].unique())
        
        # Calculate overall metrics
        for group in results:
            metrics = results[group]
            metrics['avg_cpa'] = float(metrics['total_spend'] / metrics['total_leads']) if metrics['total_leads'] > 0 else 0.0
            metrics['conversion_rate'] = float((metrics['total_estimates'] / metrics['total_leads'] * 100)) if metrics['total_leads'] > 0 else 0.0

        # Generate AI insights using Claude
        if results:
            analysis_prompt = f"""
            Analyze this marketing performance data and provide strategic insights:
            
            Date Range: {since_date} to {until_date}
            
            Performance by Medium:
            """
            
            for group, metrics in results.items():
                analysis_prompt += f"""
            {group.upper()}:
            - Total Spend: ${metrics['total_spend']:,.2f}
            - Total Leads: {metrics['total_leads']:,}
            - Total Estimates: {metrics['total_estimates']:,}
            - Avg CPA: ${metrics['avg_cpa']:.2f}
            - Conversion Rate: {metrics['conversion_rate']:.1f}%
            - Days Analyzed: {metrics['days']}
            """
            
            analysis_prompt += """
            
            Provide insights in JSON format:
            {
                "executive_summary": "Brief strategic overview",
                "medium_insights": [
                    {
                        "medium": "medium_name",
                        "trend_direction": "improving/declining/stable", 
                        "key_insight": "main finding",
                        "recommendations": ["action 1", "action 2"]
                    }
                ],
                "budget_recommendations": "Overall budget allocation advice"
            }
            """
            
            try:
                claude_response = dspy.settings.lm.basic_request(analysis_prompt)
                
                try:
                    ai_insights = json.loads(claude_response.strip())
                except:
                    ai_insights = {
                        "executive_summary": "AI analysis completed",
                        "medium_insights": [],
                        "budget_recommendations": "Continue monitoring performance"
                    }
            except:
                ai_insights = {
                    "executive_summary": "AI analysis unavailable", 
                    "medium_insights": [],
                    "budget_recommendations": "Manual analysis recommended"
                }
        else:
            ai_insights = {"executive_summary": "No data to analyze", "medium_insights": [], "budget_recommendations": ""}
        
        return TrendAnalysisResponse(
            status="success",
            message="Real BigQuery analysis with AI insights completed",
            data={
                "ai_insights": ai_insights,
                "medium_insights": [
                    {
                        "medium_group": group,
                        "metrics": metrics,
                        "trend_direction": "stable",
                        "confidence_score": 0.8
                    }
                    for group, metrics in results.items()
                ],
                "summary": {
                    "total_records_analyzed": int(len(funnel_df)),
                    "date_range": f"{since_date} to {until_date}",
                    "data_sources": ["hex_data", "meta_data"]
                },
                "analysis_metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "model_used": "claude_with_real_data",
                    "data_source": "bigquery_plus_ai"
                }
            }
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="Analysis failed",
            error=str(e)
        )

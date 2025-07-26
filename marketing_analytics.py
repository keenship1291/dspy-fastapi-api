# marketing_analytics.py - COMPREHENSIVE VERSION
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
        "version": "2.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE,
        "database_available": SessionLocal is not None,
        "endpoints": [
            "/analyze-trends - Comprehensive trend analysis with week-over-week comparisons",
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

def get_comprehensive_marketing_data(since_date: str, until_date: str):
    """Get comprehensive marketing data from all tables with proper joins"""
    
    # Main query that joins all tables and gets comprehensive metrics
    comprehensive_query = f"""
    WITH 
    -- Get funnel data
    funnel_data AS (
        SELECT 
            date,
            utm_campaign,
            utm_medium,
            SUM(leads) as total_leads,
            SUM(start_flows) as total_start_flows,
            SUM(estimates) as total_estimates,
            SUM(closings) as total_closings,
            SUM(funded) as total_funded,
            SUM(rpts) as total_rpts,
            platform as funnel_platform
        FROM `{PROJECT_ID}.{DATASET_ID}.hex_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
            AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        GROUP BY date, utm_campaign, utm_medium, platform
    ),
    
    -- Get Meta data  
    meta_data AS (
        SELECT 
            date,
            campaign_name,
            CAST(impressions AS INT64) as impressions,
            CAST(clicks AS INT64) as clicks,
            CAST(spend AS FLOAT64) as spend,
            CAST(reach AS INT64) as reach,
            CAST(landing_page_views AS INT64) as landing_page_views,
            CAST(leads AS INT64) as meta_leads,
            CAST(ctr AS FLOAT64) as ctr,
            CAST(cpc AS FLOAT64) as cpc,
            CAST(cpm AS FLOAT64) as cpm,
            platform as meta_platform
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
    ),
    
    -- Get Google data
    google_data AS (
        SELECT 
            date,
            campaign_name,
            spend_usd,
            clicks,
            impressions,
            conversions,
            cpa_usd,
            ctr_percent,
            cpc_usd,
            roas_percent,
            platform as google_platform
        FROM `{PROJECT_ID}.{DATASET_ID}.google_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
    ),
    
    -- Get mapping data (only for Meta)
    mapping_data AS (
        SELECT 
            utm_campaign,
            campaign_name_mapped,
            adset_name_mapped
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_mapping_table`
    )
    
    -- Main join query
    SELECT 
        f.date,
        f.utm_campaign,
        f.utm_medium,
        
        -- Funnel metrics
        f.total_leads,
        f.total_start_flows,
        f.total_estimates,
        f.total_closings,
        f.total_funded,
        f.total_rpts,
        
        -- Meta metrics (for paid-social only, using mapping table)
        COALESCE(m.impressions, 0) as meta_impressions,
        COALESCE(m.clicks, 0) as meta_clicks,
        COALESCE(m.spend, 0) as meta_spend,
        COALESCE(m.reach, 0) as meta_reach,
        COALESCE(m.landing_page_views, 0) as meta_landing_page_views,
        COALESCE(m.ctr, 0) as meta_ctr,
        COALESCE(m.cpc, 0) as meta_cpc,
        COALESCE(m.cpm, 0) as meta_cpm,
        
        -- Google metrics (for paid-search/paid-video, direct join on utm_campaign = campaign_name)
        COALESCE(g.spend_usd, 0) as google_spend,
        COALESCE(g.clicks, 0) as google_clicks,
        COALESCE(g.impressions, 0) as google_impressions,
        COALESCE(g.conversions, 0) as google_conversions,
        COALESCE(g.cpa_usd, 0) as google_cpa,
        COALESCE(g.ctr_percent, 0) as google_ctr,
        COALESCE(g.cpc_usd, 0) as google_cpc,
        COALESCE(g.roas_percent, 0) as google_roas,
        
        -- Mapping info (only for Meta)
        map.campaign_name_mapped,
        map.adset_name_mapped
        
    FROM funnel_data f
    -- Left join mapping only for paid-social
    LEFT JOIN mapping_data map ON f.utm_campaign = map.utm_campaign 
        AND f.utm_medium = 'paid-social'
    -- Meta data join using mapping table for paid-social
    LEFT JOIN meta_data m ON f.date = m.date 
        AND map.campaign_name_mapped = m.campaign_name
        AND f.utm_medium = 'paid-social'
    -- Google data direct join on utm_campaign for paid-search/paid-video
    LEFT JOIN google_data g ON f.date = g.date 
        AND f.utm_campaign = g.campaign_name
        AND f.utm_medium IN ('paid-search', 'paid-video')
    
    ORDER BY f.date DESC, f.utm_campaign
    """
    
    return bigquery_client.query(comprehensive_query).to_dataframe()

def calculate_week_comparisons(df, target_date):
    """Calculate yesterday vs same day last week and last week vs week before comparisons"""
    
    target_date = pd.to_datetime(target_date).date()
    yesterday = target_date - timedelta(days=1)
    same_day_last_week = yesterday - timedelta(days=7)
    
    # Week ranges
    last_week_start = yesterday - timedelta(days=6)  # 7 days including yesterday
    last_week_end = yesterday
    
    week_before_start = last_week_start - timedelta(days=7)
    week_before_end = last_week_end - timedelta(days=7)
    
    comparisons = {}
    
    # Yesterday vs Same Day Last Week
    yesterday_data = df[df['date'] == yesterday]
    same_day_last_week_data = df[df['date'] == same_day_last_week]
    
    # Last Week vs Week Before
    last_week_data = df[(df['date'] >= last_week_start) & (df['date'] <= last_week_end)]
    week_before_data = df[(df['date'] >= week_before_start) & (df['date'] <= week_before_end)]
    
    def aggregate_metrics(data):
        """Aggregate metrics for a time period"""
        if data.empty:
            return {}
        
        return {
            'total_spend': float(data['meta_spend'].sum() + data['google_spend'].sum()),
            'total_leads': int(data['total_leads'].sum()),
            'total_estimates': int(data['total_estimates'].sum()),
            'total_clicks': int(data['meta_clicks'].sum() + data['google_clicks'].sum()),
            'total_impressions': int(data['meta_impressions'].sum() + data['google_impressions'].sum()),
            'total_conversions': float(data['google_conversions'].sum()),
            'days_count': int(data['date'].nunique())
        }
    
    def calculate_change(current, previous, metric):
        """Calculate percentage change between periods"""
        if previous.get(metric, 0) == 0:
            return 0.0 if current.get(metric, 0) == 0 else 100.0
        return float((current.get(metric, 0) - previous.get(metric, 0)) / previous.get(metric, 0) * 100)
    
    # Yesterday comparisons
    yesterday_metrics = aggregate_metrics(yesterday_data)
    same_day_last_week_metrics = aggregate_metrics(same_day_last_week_data)
    
    comparisons['yesterday_vs_same_day_last_week'] = {
        'yesterday': yesterday_metrics,
        'same_day_last_week': same_day_last_week_metrics,
        'changes': {
            'spend_change': calculate_change(yesterday_metrics, same_day_last_week_metrics, 'total_spend'),
            'leads_change': calculate_change(yesterday_metrics, same_day_last_week_metrics, 'total_leads'),
            'estimates_change': calculate_change(yesterday_metrics, same_day_last_week_metrics, 'total_estimates'),
            'clicks_change': calculate_change(yesterday_metrics, same_day_last_week_metrics, 'total_clicks'),
        }
    }
    
    # Weekly comparisons
    last_week_metrics = aggregate_metrics(last_week_data)
    week_before_metrics = aggregate_metrics(week_before_data)
    
    comparisons['last_week_vs_week_before'] = {
        'last_week': last_week_metrics,
        'week_before': week_before_metrics,
        'changes': {
            'spend_change': calculate_change(last_week_metrics, week_before_metrics, 'total_spend'),
            'leads_change': calculate_change(last_week_metrics, week_before_metrics, 'total_leads'),
            'estimates_change': calculate_change(last_week_metrics, week_before_metrics, 'total_estimates'),
            'clicks_change': calculate_change(last_week_metrics, week_before_metrics, 'total_clicks'),
        }
    }
    
    return comparisons

@marketing_router.post("/analyze-trends", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(request: TrendAnalysisRequest):
    """Comprehensive marketing trends analysis with week-over-week comparisons"""
    try:
        since_date = request.date_range['since']
        until_date = request.date_range['until']
        
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                message="BigQuery not available",
                error="BigQuery client not initialized"
            )
        
        # Get comprehensive data
        df = get_comprehensive_marketing_data(since_date, until_date)
        
        if df.empty:
            return TrendAnalysisResponse(
                status="no_data",
                message=f"No data found for {since_date} to {until_date}",
                data={"date_range": f"{since_date} to {until_date}"}
            )
        
        # Calculate week-over-week comparisons
        comparisons = calculate_week_comparisons(df, until_date)
        
        # Group by medium and calculate comprehensive metrics
        medium_analysis = {}
        
        for medium in ['paid-social', 'paid-search', 'paid-video']:
            medium_data = df[df['utm_medium'] == medium]
            
            if not medium_data.empty:
                group_key = 'paid-social' if medium == 'paid-social' else 'paid-search-video'
                
                if group_key not in medium_analysis:
                    medium_analysis[group_key] = {
                        'total_spend': 0.0,
                        'total_leads': 0,
                        'total_estimates': 0,
                        'total_clicks': 0,
                        'total_impressions': 0,
                        'campaigns': []
                    }
                
                # Use appropriate spend source
                spend = float(medium_data['meta_spend'].sum()) if medium == 'paid-social' else float(medium_data['google_spend'].sum())
                clicks = int(medium_data['meta_clicks'].sum()) if medium == 'paid-social' else int(medium_data['google_clicks'].sum())
                impressions = int(medium_data['meta_impressions'].sum()) if medium == 'paid-social' else int(medium_data['google_impressions'].sum())
                
                medium_analysis[group_key]['total_spend'] += spend
                medium_analysis[group_key]['total_leads'] += int(medium_data['total_leads'].sum())
                medium_analysis[group_key]['total_estimates'] += int(medium_data['total_estimates'].sum())
                medium_analysis[group_key]['total_clicks'] += clicks
                medium_analysis[group_key]['total_impressions'] += impressions
                medium_analysis[group_key]['campaigns'].extend(medium_data['utm_campaign'].unique().tolist())
        
        # Calculate derived metrics
        for group in medium_analysis:
            metrics = medium_analysis[group]
            metrics['avg_cpa'] = float(metrics['total_spend'] / metrics['total_leads']) if metrics['total_leads'] > 0 else 0.0
            metrics['conversion_rate'] = float((metrics['total_estimates'] / metrics['total_leads'] * 100)) if metrics['total_leads'] > 0 else 0.0
            metrics['avg_ctr'] = float((metrics['total_clicks'] / metrics['total_impressions'] * 100)) if metrics['total_impressions'] > 0 else 0.0
            metrics['avg_cpc'] = float(metrics['total_spend'] / metrics['total_clicks']) if metrics['total_clicks'] > 0 else 0.0
            metrics['campaign_count'] = len(set(metrics['campaigns']))
        
        # Generate AI insights using Claude
        if medium_analysis:
            analysis_prompt = f"""
            Analyze this comprehensive marketing performance data with week-over-week comparisons:
            
            DATE RANGE: {since_date} to {until_date}
            
            PERFORMANCE BY MEDIUM:
            """
            
            for group, metrics in medium_analysis.items():
                analysis_prompt += f"""
            {group.upper()}:
            - Total Spend: ${metrics['total_spend']:,.2f}
            - Total Leads: {metrics['total_leads']:,}
            - Total Estimates: {metrics['total_estimates']:,}
            - Avg CPA: ${metrics['avg_cpa']:.2f}
            - Conversion Rate: {metrics['conversion_rate']:.1f}%
            - Avg CTR: {metrics['avg_ctr']:.2f}%
            - Avg CPC: ${metrics['avg_cpc']:.2f}
            - Active Campaigns: {metrics['campaign_count']}
            """
            
            # Add comparison data
            analysis_prompt += f"""
            
            WEEK-OVER-WEEK COMPARISONS:
            
            Yesterday vs Same Day Last Week:
            - Spend Change: {comparisons['yesterday_vs_same_day_last_week']['changes']['spend_change']:.1f}%
            - Leads Change: {comparisons['yesterday_vs_same_day_last_week']['changes']['leads_change']:.1f}%
            - Estimates Change: {comparisons['yesterday_vs_same_day_last_week']['changes']['estimates_change']:.1f}%
            
            Last Week vs Week Before:
            - Spend Change: {comparisons['last_week_vs_week_before']['changes']['spend_change']:.1f}%
            - Leads Change: {comparisons['last_week_vs_week_before']['changes']['leads_change']:.1f}%
            - Estimates Change: {comparisons['last_week_vs_week_before']['changes']['estimates_change']:.1f}%
            
            Provide strategic insights in JSON format:
            {{
                "executive_summary": "Brief strategic overview with key trends",
                "medium_insights": [
                    {{
                        "medium": "medium_name",
                        "trend_direction": "improving/declining/stable", 
                        "key_insight": "main finding including week-over-week changes",
                        "recommendations": ["action 1", "action 2"]
                    }}
                ],
                "weekly_trends": "Analysis of week-over-week performance changes",
                "budget_recommendations": "Strategic budget allocation advice based on trends"
            }}
            """
            
            try:
                claude_response = dspy.settings.lm.basic_request(analysis_prompt)
                
                # Clean up Claude's response to extract JSON
                response_text = claude_response.strip()
                if response_text.startswith('```'):
                    lines = response_text.split('\n')
                    response_text = '\n'.join([line for line in lines if not line.startswith('```')])
                
                # Find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    response_text = response_text[json_start:json_end]
                
                try:
                    ai_insights = json.loads(response_text)
                except:
                    ai_insights = {
                        "executive_summary": f"Analysis shows mixed performance across channels with total spend of ${sum(m['total_spend'] for m in medium_analysis.values()):,.2f} generating {sum(m['total_leads'] for m in medium_analysis.values()):,} leads.",
                        "medium_insights": [
                            {
                                "medium": group,
                                "trend_direction": "stable",
                                "key_insight": f"Generated {metrics['total_leads']:,} leads at ${metrics['avg_cpa']:.2f} CPA",
                                "recommendations": ["Monitor performance", "Optimize based on CPA efficiency"]
                            }
                            for group, metrics in medium_analysis.items()
                        ],
                        "weekly_trends": "Week-over-week analysis available in detailed metrics",
                        "budget_recommendations": "Focus budget on channels with lowest CPA and highest conversion rates"
                    }
            except:
                ai_insights = {
                    "executive_summary": "AI analysis unavailable", 
                    "medium_insights": [],
                    "weekly_trends": "",
                    "budget_recommendations": "Manual analysis recommended"
                }
        else:
            ai_insights = {"executive_summary": "No data to analyze", "medium_insights": [], "weekly_trends": "", "budget_recommendations": ""}
        
        return TrendAnalysisResponse(
            status="success",
            message="Comprehensive marketing analysis with week-over-week comparisons completed",
            data={
                "ai_insights": ai_insights,
                "medium_analysis": medium_analysis,
                "week_over_week_comparisons": comparisons,
                "summary": {
                    "total_records_analyzed": int(len(df)),
                    "date_range": f"{since_date} to {until_date}",
                    "data_sources": ["hex_data", "meta_data", "google_data", "meta_mapping_table"],
                    "mediums_analyzed": list(medium_analysis.keys())
                },
                "analysis_metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "model_used": "claude_comprehensive_analysis",
                    "data_source": "bigquery_all_tables",
                    "includes_week_comparisons": True
                }
            }
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="Comprehensive analysis failed",
            error=str(e)
        )

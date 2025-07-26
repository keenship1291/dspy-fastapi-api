# marketing_analytics.py - FIXED VERSION
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
    
    -- Get Meta data (handle STRING conversions safely)
    meta_data AS (
        SELECT 
            date,
            campaign_name,
            SAFE_CAST(impressions AS INT64) as impressions,
            SAFE_CAST(clicks AS INT64) as clicks,
            SAFE_CAST(spend AS FLOAT64) as spend,
            SAFE_CAST(reach AS INT64) as reach,
            SAFE_CAST(landing_page_views AS INT64) as landing_page_views,
            SAFE_CAST(leads AS INT64) as meta_leads,
            SAFE_CAST(ctr AS FLOAT64) as ctr,
            SAFE_CAST(cpc AS FLOAT64) as cpc,
            SAFE_CAST(cpm AS FLOAT64) as cpm,
            platform as meta_platform
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
            AND impressions IS NOT NULL 
            AND impressions != ''
            AND spend IS NOT NULL 
            AND spend != ''
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
    
    -- Get mapping data (correct table name)
    mapping_data AS (
        SELECT 
            utm_campaign,
            campaign_name_mapped,
            adset_name_mapped
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data_mapping`
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
    """Calculate yesterday vs same day last week and last week vs week before comparisons by channel"""
    
    target_date = pd.to_datetime(target_date).date()
    yesterday = target_date - timedelta(days=1)
    same_day_last_week = yesterday - timedelta(days=7)
    
    # Debug: Print the dates being used and data availability
    print(f"DEBUG: Target date: {target_date}")
    print(f"DEBUG: Yesterday: {yesterday}")
    print(f"DEBUG: Same day last week: {same_day_last_week}")
    print(f"DEBUG: Data date range: {df['date'].min()} to {df['date'].max()}")
    print(f"DEBUG: Available dates: {sorted(df['date'].unique())}")
    print(f"DEBUG: Yesterday data count: {len(df[df['date'] == yesterday])}")
    print(f"DEBUG: Same day last week data count: {len(df[df['date'] == same_day_last_week])}")
    
    # Week ranges
    last_week_start = yesterday - timedelta(days=6)  # 7 days including yesterday
    last_week_end = yesterday
    
    week_before_start = last_week_start - timedelta(days=7)
    week_before_end = last_week_end - timedelta(days=7)
    
    def aggregate_metrics_by_channel(data, channel_filter):
        """Aggregate metrics for a specific channel"""
        if data.empty:
            return {}
        
        # Filter data by channel (exclude Bing completely)
        if channel_filter == 'paid-social':
            filtered_data = data[data['utm_medium'] == 'paid-social']
        elif channel_filter == 'paid-search-video':
            filtered_data = data[data['utm_medium'].isin(['paid-search', 'paid-video'])]
        else:
            # For combined calculations, exclude any Bing data
            filtered_data = data[data['utm_medium'].isin(['paid-social', 'paid-search', 'paid-video'])]
        
        if filtered_data.empty:
            return {}
        
        # Use appropriate spend/click sources based on channel
        if channel_filter == 'paid-social':
            total_spend = float(filtered_data['meta_spend'].sum())
            total_clicks = int(filtered_data['meta_clicks'].sum())
            total_impressions = int(filtered_data['meta_impressions'].sum())
            total_cpm = float(filtered_data['meta_cpm'].mean()) if len(filtered_data) > 0 else 0.0
            total_cpc = float(filtered_data['meta_cpc'].mean()) if len(filtered_data) > 0 else 0.0
            total_ctr = float(filtered_data['meta_ctr'].mean()) if len(filtered_data) > 0 else 0.0
        else:
            total_spend = float(filtered_data['google_spend'].sum())
            total_clicks = int(filtered_data['google_clicks'].sum())
            total_impressions = int(filtered_data['google_impressions'].sum())
            # Calculate CPM, CPC, CTR for Google
            total_cpm = (total_spend / total_impressions * 1000) if total_impressions > 0 else 0.0
            total_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0.0
            total_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0.0
        
        total_estimates = int(filtered_data['total_estimates'].sum())
        total_closings = int(filtered_data['total_closings'].sum())
        total_leads = int(filtered_data['total_leads'].sum())
        
        # Calculate conversion rates
        estimate_cvr = (total_estimates / total_leads * 100) if total_leads > 0 else 0.0
        closings_cvr = (total_closings / total_estimates * 100) if total_estimates > 0 else 0.0
        
        return {
            'total_spend': total_spend,
            'total_clicks': total_clicks,
            'total_impressions': total_impressions,
            'total_estimates': total_estimates,
            'total_closings': total_closings,
            'total_leads': total_leads,
            'estimate_cvr': float(estimate_cvr),
            'closings_cvr': float(closings_cvr),
            'avg_cpm': float(total_cpm),
            'avg_cpc': float(total_cpc),
            'avg_ctr': float(total_ctr),
            'days_count': int(filtered_data['date'].nunique())
        }
    
    def calculate_change(current, previous, metric):
        """Calculate percentage change between periods"""
        current_val = current.get(metric, 0)
        previous_val = previous.get(metric, 0)
        
        # Debug output
        print(f"DEBUG: Calculating change for {metric}: current={current_val}, previous={previous_val}")
        
        if previous_val == 0:
            if current_val == 0:
                return 0.0
            else:
                return 100.0  # If previous was 0 but current has value, it's 100% increase
        return float((current_val - previous_val) / previous_val * 100)
    
    def format_comparison_by_channel(current_metrics, previous_metrics, period_name, channel_name, comparison_possible=True):
        """Format comparison in the requested bullet point format for specific channel"""
        if not current_metrics:
            return f"\n{channel_name} - {period_name}\n• No data available for this period"
        
        if not comparison_possible or not previous_metrics:
            # Show current metrics without comparison
            return f"""
{channel_name} - {period_name} (Baseline - No Comparison Data Available)
• Total Spend: ${current_metrics.get('total_spend', 0):,.0f}
• Total Impressions: {current_metrics.get('total_impressions', 0):,}
• Total Clicks: {current_metrics.get('total_clicks', 0):,}
• CPM: ${current_metrics.get('avg_cpm', 0):.2f}
• CPC: ${current_metrics.get('avg_cpc', 0):.2f}
• CTR: {current_metrics.get('avg_ctr', 0):.2f}%
• Estimate CVR: {current_metrics.get('estimate_cvr', 0):.1f}%
• Total Estimates: {current_metrics.get('total_estimates', 0):,}
• Closings CVR: {current_metrics.get('closings_cvr', 0):.1f}%
• Total Closings: {current_metrics.get('total_closings', 0):,}
"""
        
        # Debug output
        print(f"DEBUG: {channel_name} - {period_name}")
        print(f"DEBUG: Current metrics: {current_metrics}")
        print(f"DEBUG: Previous metrics: {previous_metrics}")
        
        spend_change = calculate_change(current_metrics, previous_metrics, 'total_spend')
        clicks_change = calculate_change(current_metrics, previous_metrics, 'total_clicks')
        impressions_change = calculate_change(current_metrics, previous_metrics, 'total_impressions')
        estimates_change = calculate_change(current_metrics, previous_metrics, 'total_estimates')
        closings_change = calculate_change(current_metrics, previous_metrics, 'total_closings')
        cpm_change = calculate_change(current_metrics, previous_metrics, 'avg_cpm')
        cpc_change = calculate_change(current_metrics, previous_metrics, 'avg_cpc')
        ctr_change = calculate_change(current_metrics, previous_metrics, 'avg_ctr')
        
        # For CVR, calculate absolute percentage point change
        estimate_cvr_change = current_metrics.get('estimate_cvr', 0) - previous_metrics.get('estimate_cvr', 0)
        closings_cvr_change = current_metrics.get('closings_cvr', 0) - previous_metrics.get('closings_cvr', 0)
        
        # Format with + or - signs
        def format_change(value, is_cvr=False):
            if is_cvr:
                return f"({value:+.1f}pp)" if abs(value) > 0.1 else "(0.0pp)"  # pp = percentage points
            else:
                return f"({value:+.1f}%)" if abs(value) > 0.1 else "(0.0%)"
        
        return f"""
{channel_name} - {period_name}
• Total Spend: ${current_metrics.get('total_spend', 0):,.0f} {format_change(spend_change)}
• Total Impressions: {current_metrics.get('total_impressions', 0):,} {format_change(impressions_change)}
• Total Clicks: {current_metrics.get('total_clicks', 0):,} {format_change(clicks_change)}
• CPM: ${current_metrics.get('avg_cpm', 0):.2f} {format_change(cpm_change)}
• CPC: ${current_metrics.get('avg_cpc', 0):.2f} {format_change(cpc_change)}
• CTR: {current_metrics.get('avg_ctr', 0):.2f}% {format_change(ctr_change)}
• Estimate CVR: {current_metrics.get('estimate_cvr', 0):.1f}% {format_change(estimate_cvr_change, True)}
• Total Estimates: {current_metrics.get('total_estimates', 0):,} {format_change(estimates_change)}
• Closings CVR: {current_metrics.get('closings_cvr', 0):.1f}% {format_change(closings_cvr_change, True)}
• Total Closings: {current_metrics.get('total_closings', 0):,} {format_change(closings_change)}
"""
    
    # Get data for time periods
    yesterday_data = df[df['date'] == yesterday]
    same_day_last_week_data = df[df['date'] == same_day_last_week]
    last_week_data = df[(df['date'] >= last_week_start) & (df['date'] <= last_week_end)]
    week_before_data = df[(df['date'] >= week_before_start) & (df['date'] <= week_before_end)]
    
    # Calculate for each channel (Meta and Google only)
    channels = ['paid-social', 'paid-search-video']
    channel_names = ['Paid Social (Meta)', 'Paid Search & Video (Google)']
    
    formatted_comparisons = {}
    
    # Yesterday vs Same Day Last Week (separate for each channel)
    for period_type, period_name in [('yesterday_vs_same_day_last_week', 'Yesterday vs Same Day Last Week'), 
                                     ('last_7_days_vs_previous_7_days', 'Last 7 Days vs Previous 7 Days')]:
        
        if period_type == 'yesterday_vs_same_day_last_week':
            current_data = yesterday_data
            previous_data = same_day_last_week_data
            comparison_possible = can_do_daily_comparison
        else:
            current_data = last_week_data
            previous_data = week_before_data
            comparison_possible = can_do_weekly_comparison
        
        period_comparisons = []
        for channel, channel_name in zip(channels, channel_names):
            current_metrics = aggregate_metrics_by_channel(current_data, channel)
            previous_metrics = aggregate_metrics_by_channel(previous_data, channel)
            
            if current_metrics:  # Only include channels that have data
                comparison = format_comparison_by_channel(
                    current_metrics, previous_metrics, period_name, channel_name, comparison_possible
                )
                period_comparisons.append(comparison)
        
        formatted_comparisons[period_type] = period_comparisons
    
    # Yesterday vs Same Day Last Week (separate for each channel)
    for period_type, period_name in [('yesterday_vs_same_day_last_week', 'Yesterday vs Same Day Last Week'), 
                                     ('last_7_days_vs_previous_7_days', 'Last 7 Days vs Previous 7 Days')]:
        
        if period_type == 'yesterday_vs_same_day_last_week':
            current_data = yesterday_data
            previous_data = same_day_last_week_data
            comparison_possible = can_do_daily_comparison
        else:
            current_data = last_week_data
            previous_data = week_before_data
            comparison_possible = can_do_weekly_comparison
        
        period_comparisons = []
        for channel, channel_name in zip(channels, channel_names):
            current_metrics = aggregate_metrics_by_channel(current_data, channel)
            previous_metrics = aggregate_metrics_by_channel(previous_data, channel)
            
            if current_metrics:  # Only include channels that have data
                comparison = format_comparison_by_channel(
                    current_metrics, previous_metrics, period_name, channel_name, comparison_possible
                )
                period_comparisons.append(comparison)
        
        formatted_comparisons[period_type] = period_comparisons
    
    return {
        'formatted_comparisons': formatted_comparisons,
        'comparison_info': {
            'can_do_daily_comparison': can_do_daily_comparison,
            'can_do_weekly_comparison': can_do_weekly_comparison,
            'days_of_data_available': days_of_data,
            'data_start_date': str(data_start_date),
            'dates_needed': {
                'yesterday': str(yesterday),
                'same_day_last_week': str(same_day_last_week),
                'last_week_range': f"{last_week_start} to {last_week_end}",
                'week_before_range': f"{week_before_start} to {week_before_end}"
            }
        },
        'raw_metrics': {
            'yesterday_by_channel': {
                channel: aggregate_metrics_by_channel(yesterday_data, channel) 
                for channel in channels
            },
            'same_day_last_week_by_channel': {
                channel: aggregate_metrics_by_channel(same_day_last_week_data, channel) 
                for channel in channels
            },
            'last_week_by_channel': {
                channel: aggregate_metrics_by_channel(last_week_data, channel) 
                for channel in channels
            },
            'week_before_by_channel': {
                channel: aggregate_metrics_by_channel(week_before_data, channel) 
                for channel in channels
            }
        }
    }
    
    return {
        'formatted_comparisons': formatted_comparisons,
        'comparison_info': {
            'can_do_daily_comparison': can_do_daily_comparison,
            'can_do_weekly_comparison': can_do_weekly_comparison,
            'days_of_data_available': days_of_data,
            'data_start_date': str(data_start_date),
            'dates_needed': {
                'yesterday': str(yesterday),
                'same_day_last_week': str(same_day_last_week),
                'last_week_range': f"{last_week_start} to {last_week_end}",
                'week_before_range': f"{week_before_start} to {week_before_end}"
            }
        },
        'raw_metrics': {
            'yesterday_by_channel': {
                channel: aggregate_metrics_by_channel(yesterday_data, channel) 
                for channel in channels
            },
            'same_day_last_week_by_channel': {
                channel: aggregate_metrics_by_channel(same_day_last_week_data, channel) 
                for channel in channels
            },
            'last_week_by_channel': {
                channel: aggregate_metrics_by_channel(last_week_data, channel) 
                for channel in channels
            },
            'week_before_by_channel': {
                channel: aggregate_metrics_by_channel(week_before_data, channel) 
                for channel in channels
            }
        }
    }

@marketing_router.get("/debug-data")
async def debug_data_sources():
    """Debug endpoint to check data in each table"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return {"error": "BigQuery not available"}
        
        # Check hex_data
        hex_query = f"""
        SELECT utm_medium, COUNT(*) as records, SUM(leads) as total_leads
        FROM `{PROJECT_ID}.{DATASET_ID}.hex_data`
        WHERE date >= '2025-07-20'
        GROUP BY utm_medium
        ORDER BY utm_medium
        """
        
        # Check google_data  
        google_query = f"""
        SELECT COUNT(*) as records, SUM(spend_usd) as total_spend
        FROM `{PROJECT_ID}.{DATASET_ID}.google_data`
        WHERE date >= '2025-07-20'
        """
        
        # Check meta_data
        meta_query = f"""
        SELECT COUNT(*) as records, SUM(SAFE_CAST(spend AS FLOAT64)) as total_spend
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data`
        WHERE date >= '2025-07-20'
            AND spend IS NOT NULL 
            AND spend != ''
        """
        
        hex_result = bigquery_client.query(hex_query).to_dataframe()
        google_result = bigquery_client.query(google_query).to_dataframe()
        meta_result = bigquery_client.query(meta_query).to_dataframe()
        
        return {
            "hex_data": hex_result.to_dict('records'),
            "google_data": google_result.to_dict('records'),
            "meta_data": meta_result.to_dict('records')
        }
        
    except Exception as e:
        return {"error": str(e)}

@marketing_router.get("/debug-comprehensive/{since_date}/{until_date}")
async def debug_comprehensive_data(since_date: str, until_date: str):
    """Debug the comprehensive data query to see what we're actually getting"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return {"error": "BigQuery not available"}
        
        # Test the exact same query that analyze-trends uses
        df = get_comprehensive_marketing_data(since_date, until_date)
        
        if df.empty:
            return {
                "error": "No data returned from comprehensive query",
                "date_range": f"{since_date} to {until_date}"
            }
        
        # Show what we got
        result = {
            "date_range_requested": f"{since_date} to {until_date}",
            "total_records": len(df),
            "date_range_actual": f"{df['date'].min()} to {df['date'].max()}",
            "unique_dates": sorted(df['date'].unique().tolist()),
            "utm_mediums": df['utm_medium'].unique().tolist(),
            "sample_records": df.head(10).to_dict('records'),
            "records_by_date_medium": df.groupby(['date', 'utm_medium']).size().reset_index(name='count').to_dict('records'),
            "meta_spend_summary": {
                "total": float(df['meta_spend'].sum()),
                "records_with_spend": len(df[df['meta_spend'] > 0]),
                "max_spend": float(df['meta_spend'].max()),
                "by_date": df.groupby('date')['meta_spend'].sum().to_dict()
            },
            "google_spend_summary": {
                "total": float(df['google_spend'].sum()),
                "records_with_spend": len(df[df['google_spend'] > 0]),
                "max_spend": float(df['google_spend'].max()),
                "by_date": df.groupby('date')['google_spend'].sum().to_dict()
            }
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}

@marketing_router.get("/debug-raw-tables/{since_date}/{until_date}")
async def debug_raw_tables(since_date: str, until_date: str):
    """Check each table individually to see where data might be missing"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return {"error": "BigQuery not available"}
        
        results = {}
        
        # Check hex_data
        hex_query = f"""
        SELECT date, utm_medium, COUNT(*) as records, SUM(leads) as leads, SUM(estimates) as estimates
        FROM `{PROJECT_ID}.{DATASET_ID}.hex_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
        GROUP BY date, utm_medium
        ORDER BY date DESC, utm_medium
        """
        results['hex_data'] = bigquery_client.query(hex_query).to_dataframe().to_dict('records')
        
        # Check meta_data  
        meta_query = f"""
        SELECT date, COUNT(*) as records, 
               SUM(SAFE_CAST(spend AS FLOAT64)) as total_spend,
               COUNT(CASE WHEN spend IS NOT NULL AND spend != '' THEN 1 END) as valid_spend_records
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
        GROUP BY date
        ORDER BY date DESC
        """
        results['meta_data'] = bigquery_client.query(meta_query).to_dataframe().to_dict('records')
        
        # Check google_data
        google_query = f"""
        SELECT date, COUNT(*) as records, SUM(spend_usd) as total_spend
        FROM `{PROJECT_ID}.{DATASET_ID}.google_data`
        WHERE date BETWEEN '{since_date}' AND '{until_date}'
        GROUP BY date
        ORDER BY date DESC
        """
        results['google_data'] = bigquery_client.query(google_query).to_dataframe().to_dict('records')
        
        # Check mapping table
        mapping_query = f"""
        SELECT COUNT(*) as total_mappings, COUNT(DISTINCT utm_campaign) as unique_utm_campaigns
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data_mapping`
        """
        results['mapping_data'] = bigquery_client.query(mapping_query).to_dataframe().to_dict('records')
        
        return {
            "date_range": f"{since_date} to {until_date}",
            "individual_tables": results
        }
        
    except Exception as e:
        return {"error": str(e)}

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
        
        # Expand date range to include historical data for comparisons
        target_date = pd.to_datetime(until_date).date()
        # Go back 14 more days to ensure we have comparison data
        expanded_since_date = (pd.to_datetime(since_date).date() - timedelta(days=14)).strftime('%Y-%m-%d')
        
        print(f"DEBUG: Original date range: {since_date} to {until_date}")
        print(f"DEBUG: Expanded date range for historical data: {expanded_since_date} to {until_date}")
        
        # Get comprehensive data with expanded range
        df = get_comprehensive_marketing_data(expanded_since_date, until_date)
        
        if df.empty:
            return TrendAnalysisResponse(
                status="no_data",
                message=f"No data found for expanded range {expanded_since_date} to {until_date}",
                data={"date_range": f"{since_date} to {until_date}", "expanded_range": f"{expanded_since_date} to {until_date}"}
            )
        
        print(f"DEBUG: Retrieved {len(df)} records from {df['date'].min()} to {df['date'].max()}")
        print(f"DEBUG: Unique dates in data: {sorted(df['date'].unique())}")
        
        # Calculate week-over-week comparisons using the target date
        comparisons = calculate_week_comparisons(df, until_date)
        
        # Filter df back to original date range for medium analysis (but keep expanded for comparisons)
        analysis_df = df[(df['date'] >= since_date) & (df['date'] <= until_date)]
        
        print(f"DEBUG: Analysis DF has {len(analysis_df)} records from {analysis_df['date'].min() if not analysis_df.empty else 'N/A'} to {analysis_df['date'].max() if not analysis_df.empty else 'N/A'}")
        
        # Group by medium and calculate comprehensive metrics (Meta and Google only)
        medium_analysis = {}
        
        for medium in ['paid-social', 'paid-search', 'paid-video']:
            medium_data = analysis_df[analysis_df['utm_medium'] == medium]
            
            if not medium_data.empty:
                group_key = 'paid-social' if medium == 'paid-social' else 'paid-search-video'
                
                if group_key not in medium_analysis:
                    medium_analysis[group_key] = {
                        'total_spend': 0.0, 'total_leads': 0, 'total_estimates': 0,
                        'total_start_flows': 0, 'total_closings': 0, 'total_funded': 0,
                        'total_rpts': 0, 'total_clicks': 0, 'total_impressions': 0, 'campaigns': []
                    }
                
                # Use appropriate spend source (no Bing filtering needed)
                spend = float(medium_data['meta_spend'].sum()) if medium == 'paid-social' else float(medium_data['google_spend'].sum())
                clicks = int(medium_data['meta_clicks'].sum()) if medium == 'paid-social' else int(medium_data['google_clicks'].sum())
                impressions = int(medium_data['meta_impressions'].sum()) if medium == 'paid-social' else int(medium_data['google_impressions'].sum())
                
                # Add FUNNEL METRICS from hex_data
                medium_analysis[group_key]['total_spend'] += spend
                medium_analysis[group_key]['total_leads'] += int(medium_data['total_leads'].sum())
                medium_analysis[group_key]['total_estimates'] += int(medium_data['total_estimates'].sum())
                medium_analysis[group_key]['total_start_flows'] += int(medium_data['total_start_flows'].sum())
                medium_analysis[group_key]['total_closings'] += int(medium_data['total_closings'].sum())
                medium_analysis[group_key]['total_funded'] += int(medium_data['total_funded'].sum())
                medium_analysis[group_key]['total_rpts'] += int(medium_data['total_rpts'].sum())
                medium_analysis[group_key]['total_clicks'] += clicks
                medium_analysis[group_key]['total_impressions'] += impressions
                medium_analysis[group_key]['campaigns'].extend(medium_data['utm_campaign'].unique().tolist())
        
        # Calculate derived metrics for both groups
        for group in medium_analysis:
            metrics = medium_analysis[group]
            metrics['avg_cpa'] = float(metrics['total_spend'] / metrics['total_leads']) if metrics['total_leads'] > 0 else 0.0
            metrics['lead_to_estimate_rate'] = float((metrics['total_estimates'] / metrics['total_leads'] * 100)) if metrics['total_leads'] > 0 else 0.0
            metrics['estimate_to_closing_rate'] = float((metrics['total_closings'] / metrics['total_estimates'] * 100)) if metrics['total_estimates'] > 0 else 0.0
            metrics['closing_to_funding_rate'] = float((metrics['total_funded'] / metrics['total_closings'] * 100)) if metrics['total_closings'] > 0 else 0.0
            metrics['avg_ctr'] = float((metrics['total_clicks'] / metrics['total_impressions'] * 100)) if metrics['total_impressions'] > 0 else 0.0
            metrics['avg_cpc'] = float(metrics['total_spend'] / metrics['total_clicks']) if metrics['total_clicks'] > 0 else 0.0
            metrics['avg_cpm'] = float((metrics['total_spend'] / metrics['total_impressions'] * 1000)) if metrics['total_impressions'] > 0 else 0.0
            metrics['campaign_count'] = len(set(metrics['campaigns']))
        
        # Generate AI insights using Claude
        if medium_analysis and comparisons:
            analysis_prompt = f"""
            Analyze this marketing performance data and provide insights in the EXACT format below:
            
            PERFORMANCE OVERVIEW:
            {comparisons['formatted_comparisons']['yesterday_vs_same_day_last_week']}
            {comparisons['formatted_comparisons']['last_7_days_vs_previous_7_days']}
            
            MEDIUM BREAKDOWN:
            """
            
            for group, metrics in medium_analysis.items():
                analysis_prompt += f"""
            {group.upper()}:
            - Total Spend: ${metrics['total_spend']:,.2f}
            - Total Leads: {metrics['total_leads']:,}
            - Total Start Flows: {metrics['total_start_flows']:,}
            - Total Estimates: {metrics['total_estimates']:,}
            - Total Closings: {metrics['total_closings']:,}
            - Total Funded: {metrics['total_funded']:,}
            - Lead to Estimate Rate: {metrics['lead_to_estimate_rate']:.1f}%
            - Estimate to Closing Rate: {metrics['estimate_to_closing_rate']:.1f}%
            - Closing to Funding Rate: {metrics['closing_to_funding_rate']:.1f}%
            - Active Campaigns: {metrics['campaign_count']}
            """
            
            analysis_prompt += f"""
            
            Please provide analysis understanding that:
            - PAID-SOCIAL (Meta): Designed for prospecting/lead generation, not expected to drive high conversions
            - PAID-SEARCH-VIDEO (Google): Focused on conversions from higher-intent traffic
            
            Respond in JSON format:
            {{
                "performance_summary": "The formatted comparison data above showing separate metrics for each channel",
                "cross_channel_analysis": "One paragraph comparing Meta (prospecting focus) vs Google (conversion focus) performance, highlighting their different roles in the funnel",
                "paid_social_analysis": "One paragraph analyzing Meta's prospecting performance, lead generation trends, and campaign efficiency",
                "paid_search_video_analysis": "One paragraph analyzing Google's conversion performance, higher-intent traffic patterns, and closing efficiency"
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
                    # Add the formatted comparisons to the AI insights
                    ai_insights['performance_summary'] = comparisons['formatted_comparisons']
                except:
                    ai_insights = {
                        "performance_summary": comparisons['formatted_comparisons'],
                        "cross_channel_analysis": f"Paid-social demonstrates volume efficiency with {medium_analysis.get('paid-social', {}).get('total_leads', 0):,} leads at ${medium_analysis.get('paid-social', {}).get('avg_cpa', 0):.2f} CPA, while paid-search-video shows quality focus with {medium_analysis.get('paid-search-video', {}).get('lead_to_estimate_rate', 0):.1f}% conversion rates despite higher costs.",
                        "paid_social_analysis": f"Paid-social generated {medium_analysis.get('paid-social', {}).get('total_estimates', 0):,} estimates from {medium_analysis.get('paid-social', {}).get('campaign_count', 0)} active campaigns, showing consistent performance across the meta platform with strong volume generation capabilities.",
                        "paid_search_video_analysis": f"Paid-search-video delivered {medium_analysis.get('paid-search-video', {}).get('total_estimates', 0):,} estimates with higher intent quality, operating across {medium_analysis.get('paid-search-video', {}).get('campaign_count', 0)} campaigns with focus on conversion efficiency over volume."
                    }
            except Exception as e:
                ai_insights = {
                    "performance_summary": comparisons['formatted_comparisons'],
                    "cross_channel_analysis": "Analysis unavailable due to processing error",
                    "paid_social_analysis": "Medium-specific analysis unavailable",
                    "paid_search_video_analysis": "Medium-specific analysis unavailable"
                }
        else:
            ai_insights = {
                "performance_summary": {},
                "cross_channel_analysis": "No data available for analysis",
                "paid_social_analysis": "No paid-social data available",
                "paid_search_video_analysis": "No paid-search-video data available"
            }
        
        return TrendAnalysisResponse(
            status="success",
            message="Comprehensive marketing analysis with week-over-week comparisons completed",
            data={
                "ai_insights": ai_insights,
                "medium_analysis": medium_analysis,
                "week_over_week_comparisons": comparisons,
                "summary": {
                    "total_records_analyzed": int(len(analysis_df)),
                    "total_records_for_comparisons": int(len(df)),
                    "original_date_range": f"{since_date} to {until_date}",
                    "expanded_date_range": f"{expanded_since_date} to {until_date}",
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

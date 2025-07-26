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
        
        total_spend = float(data['meta_spend'].sum() + data['google_spend'].sum())
        total_clicks = int(data['meta_clicks'].sum() + data['google_clicks'].sum())
        total_estimates = int(data['total_estimates'].sum())
        total_closings = int(data['total_closings'].sum())
        total_leads = int(data['total_leads'].sum())
        
        # Calculate conversion rates
        estimate_cvr = (total_estimates / total_leads * 100) if total_leads > 0 else 0.0
        closings_cvr = (total_closings / total_estimates * 100) if total_estimates > 0 else 0.0
        
        return {
            'total_spend': total_spend,
            'total_clicks': total_clicks,
            'total_estimates': total_estimates,
            'total_closings': total_closings,
            'total_leads': total_leads,
            'estimate_cvr': float(estimate_cvr),
            'closings_cvr': float(closings_cvr),
            'days_count': int(data['date'].nunique())
        }
    
    def calculate_change(current, previous, metric):
        """Calculate percentage change between periods"""
        if previous.get(metric, 0) == 0:
            return 0.0 if current.get(metric, 0) == 0 else 100.0
        return float((current.get(metric, 0) - previous.get(metric, 0)) / previous.get(metric, 0) * 100)
    
    def format_comparison(current_metrics, previous_metrics, period_name):
        """Format comparison in the requested bullet point format"""
        spend_change = calculate_change(current_metrics, previous_metrics, 'total_spend')
        clicks_change = calculate_change(current_metrics, previous_metrics, 'total_clicks')
        estimates_change = calculate_change(current_metrics, previous_metrics, 'total_estimates')
        closings_change = calculate_change(current_metrics, previous_metrics, 'total_closings')
        
        # For CVR, calculate absolute percentage point change
        estimate_cvr_change = current_metrics.get('estimate_cvr', 0) - previous_metrics.get('estimate_cvr', 0)
        closings_cvr_change = current_metrics.get('closings_cvr', 0) - previous_metrics.get('closings_cvr', 0)
        
        # Format with + or - signs
        def format_change(value, is_cvr=False):
            if is_cvr:
                return f"({value:+.1f}%)" if abs(value) > 0.1 else "(0.0%)"
            else:
                return f"({value:+.1f}%)" if abs(value) > 0.1 else "(0.0%)"
        
        return f"""
{period_name}
• Total Spend: ${current_metrics.get('total_spend', 0):,.0f} {format_change(spend_change)}
• Total Clicks: {current_metrics.get('total_clicks', 0):,} {format_change(clicks_change)}
• Estimate CVR: {current_metrics.get('estimate_cvr', 0):.1f}% {format_change(estimate_cvr_change, True)}
• Total Estimates: {current_metrics.get('total_estimates', 0):,} {format_change(estimates_change)}
• Closings CVR: {current_metrics.get('closings_cvr', 0):.1f}% {format_change(closings_cvr_change, True)}
• Total Closings: {current_metrics.get('total_closings', 0):,} {format_change(closings_change)}
"""
    
    # Yesterday comparisons
    yesterday_metrics = aggregate_metrics(yesterday_data)
    same_day_last_week_metrics = aggregate_metrics(same_day_last_week_data)
    
    # Weekly comparisons
    last_week_metrics = aggregate_metrics(last_week_data)
    week_before_metrics = aggregate_metrics(week_before_data)
    
    comparisons['formatted_comparisons'] = {
        'yesterday_vs_same_day_last_week': format_comparison(
            yesterday_metrics, same_day_last_week_metrics, "Yesterday vs Same Day Last Week"
        ),
        'last_7_days_vs_previous_7_days': format_comparison(
            last_week_metrics, week_before_metrics, "Last 7 Days vs Previous 7 Days"
        )
    }
    
    # Add raw metrics for AI analysis
    comparisons['raw_metrics'] = {
        'yesterday': yesterday_metrics,
        'same_day_last_week': same_day_last_week_metrics,
        'last_week': last_week_metrics,
        'week_before': week_before_metrics
    }
    
    # Calculate medium-specific changes for narrative
    comparisons['medium_changes'] = {}
    for medium in ['paid-social', 'paid-search', 'paid-video']:
        medium_yesterday = yesterday_data[yesterday_data['utm_medium'] == medium]
        medium_same_day_last_week = same_day_last_week_data[same_day_last_week_data['utm_medium'] == medium]
        medium_last_week = last_week_data[last_week_data['utm_medium'] == medium]
        medium_week_before = week_before_data[week_before_data['utm_medium'] == medium]
        
        group_key = 'paid-social' if medium == 'paid-social' else 'paid-search-video'
        if group_key not in comparisons['medium_changes']:
            comparisons['medium_changes'][group_key] = {
                'yesterday_metrics': aggregate_metrics(pd.DataFrame()),
                'last_week_metrics': aggregate_metrics(pd.DataFrame())
            }
        
        # Aggregate for grouped mediums
        if medium == 'paid-social':
            comparisons['medium_changes'][group_key]['yesterday_metrics'] = aggregate_metrics(medium_yesterday)
            comparisons['medium_changes'][group_key]['same_day_last_week_metrics'] = aggregate_metrics(medium_same_day_last_week)
            comparisons['medium_changes'][group_key]['last_week_metrics'] = aggregate_metrics(medium_last_week)
            comparisons['medium_changes'][group_key]['week_before_metrics'] = aggregate_metrics(medium_week_before)
        else:
            # Combine paid-search and paid-video
            if 'yesterday_metrics' not in comparisons['medium_changes'][group_key] or not comparisons['medium_changes'][group_key]['yesterday_metrics']:
                comparisons['medium_changes'][group_key]['yesterday_metrics'] = aggregate_metrics(medium_yesterday)
                comparisons['medium_changes'][group_key]['same_day_last_week_metrics'] = aggregate_metrics(medium_same_day_last_week)
                comparisons['medium_changes'][group_key]['last_week_metrics'] = aggregate_metrics(medium_last_week)
                comparisons['medium_changes'][group_key]['week_before_metrics'] = aggregate_metrics(medium_week_before)
            else:
                # Add to existing metrics
                current = comparisons['medium_changes'][group_key]['yesterday_metrics']
                new = aggregate_metrics(medium_yesterday)
                for key in ['total_spend', 'total_clicks', 'total_estimates', 'total_closings', 'total_leads']:
                    current[key] = current.get(key, 0) + new.get(key, 0)
                
                # Recalculate CVRs
                current['estimate_cvr'] = (current['total_estimates'] / current['total_leads'] * 100) if current['total_leads'] > 0 else 0.0
                current['closings_cvr'] = (current['total_closings'] / current['total_estimates'] * 100) if current['total_estimates'] > 0 else 0.0
    
    return comparisons

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
        SELECT COUNT(*) as records, SUM(CAST(spend AS FLOAT64)) as total_spend
        FROM `{PROJECT_ID}.{DATASET_ID}.meta_data`
        WHERE date >= '2025-07-20'
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
                        'total_start_flows': 0,
                        'total_closings': 0,
                        'total_funded': 0,
                        'total_rpts': 0,
                        'total_clicks': 0,
                        'total_impressions': 0,
                        'campaigns': []
                    }
                
                # Use appropriate spend source
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
        
        # Calculate derived metrics
        for group in medium_analysis:
            metrics = medium_analysis[group]
            metrics['avg_cpa'] = float(metrics['total_spend'] / metrics['total_leads']) if metrics['total_leads'] > 0 else 0.0
            metrics['conversion_rate'] = float((metrics['total_estimates'] / metrics['total_leads'] * 100)) if metrics['total_leads'] > 0 else 0.0
            metrics['closing_rate'] = float((metrics['total_closings'] / metrics['total_estimates'] * 100)) if metrics['total_estimates'] > 0 else 0.0
            metrics['funding_rate'] = float((metrics['total_funded'] / metrics['total_closings'] * 100)) if metrics['total_closings'] > 0 else 0.0
            metrics['avg_ctr'] = float((metrics['total_clicks'] / metrics['total_impressions'] * 100)) if metrics['total_impressions'] > 0 else 0.0
            metrics['avg_cpc'] = float(metrics['total_spend'] / metrics['total_clicks']) if metrics['total_clicks'] > 0 else 0.0
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
            - Lead to Estimate Rate: {metrics['conversion_rate']:.1f}%
            - Estimate to Closing Rate: {metrics['closing_rate']:.1f}%
            - Closing to Funding Rate: {metrics['funding_rate']:.1f}%
            - Active Campaigns: {metrics['campaign_count']}
            """
            
            analysis_prompt += """
            
            Please provide analysis in this JSON format:
            {
                "performance_summary": "The formatted comparison data above",
                "cross_channel_analysis": "One paragraph comparing paid-social vs paid-search-video performance, highlighting key differences in efficiency, volume, and conversion patterns",
                "paid_social_analysis": "One paragraph analyzing paid-social changes, campaign performance trends, and strategic insights",
                "paid_search_video_analysis": "One paragraph analyzing paid-search-video changes, campaign performance trends, and strategic insights"
            }
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
                        "cross_channel_analysis": f"Paid-social demonstrates volume efficiency with {medium_analysis.get('paid-social', {}).get('total_leads', 0):,} leads at ${medium_analysis.get('paid-social', {}).get('avg_cpa', 0):.2f} CPA, while paid-search-video shows quality focus with {medium_analysis.get('paid-search-video', {}).get('conversion_rate', 0):.1f}% conversion rates despite higher costs.",
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
                    "total_records_analyzed": int(len(df)),
                    "date_range": f"{since_date} to {until_date}",
                    "data_sources": ["hex_data", "meta_data", "google_data", "meta_data_mapping"],
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

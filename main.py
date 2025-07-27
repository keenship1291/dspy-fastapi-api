# marketing_analytics.py - CLAUDE ANALYSIS WITH CHANNEL SEPARATION
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import dspy
import os
from datetime import datetime, timedelta, date
import logging
from google.oauth2 import service_account
import json

# Only import these if available (so app doesn't crash if missing)
try:
    from google.cloud import bigquery
    import pandas as pd
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    bigquery = None
    pd = None

# BigQuery setup
PROJECT_ID = "gtm-p3gj3zzk-nthlo"
DATASET_ID = "last_14_days_analysis"

if BIGQUERY_AVAILABLE:
    try:
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
        "message": "Marketing Analytics API - Claude Analysis with Channel Separation",
        "version": "6.1.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE
    }

def get_comprehensive_query(date_filter, period_name, analysis_depth):
    """Generate comprehensive query with proper CTEs"""
    base_query = f"""
    WITH meta_aggregated_by_campaign AS (
      -- Pre-aggregate Meta data by utm_campaign and date 
      -- This gives us the TRUE platform metrics regardless of hex mapping
      SELECT 
        mm.utm_campaign,
        m.date,
        SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend,
        SUM(SAFE_CAST(m.impressions AS INT64)) as meta_impressions,
        SUM(SAFE_CAST(m.clicks AS INT64)) as meta_clicks,
        SUM(SAFE_CAST(m.leads AS INT64)) as meta_leads,
        SUM(SAFE_CAST(m.purchases AS INT64)) as meta_purchases,
        SUM(SAFE_CAST(m.landing_page_views AS INT64)) as meta_landing_page_views
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
      JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
        ON mm.adset_name_mapped = m.adset_name
      GROUP BY mm.utm_campaign, m.date
    ),
    google_aggregated_by_campaign AS (
      -- Pre-aggregate Google data by campaign and date
      SELECT 
        campaign_name as utm_campaign,
        date,
        SUM(spend_usd) as google_spend,
        SUM(impressions) as google_impressions,
        SUM(clicks) as google_clicks,
        SUM(conversions) as google_conversions
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
      GROUP BY campaign_name, date
    )
    
    SELECT 
      '{period_name}' as period,
      
      -- Hex Funnel Metrics (prioritize hex_data for conversions)
      SUM(h.leads) as hex_leads,
      SUM(h.start_flows) as hex_start_flows,
      SUM(h.estimates) as hex_estimates,
      SUM(h.closings) as hex_closings,
      SUM(h.funded) as hex_funded,
      SUM(h.rpts) as hex_rpts,
      
      -- Meta Advertising Metrics (prioritize meta platform data)
      SUM(COALESCE(ma.meta_spend, 0)) as meta_spend,
      SUM(COALESCE(ma.meta_impressions, 0)) as meta_impressions,
      SUM(COALESCE(ma.meta_clicks, 0)) as meta_clicks,
      SUM(COALESCE(ma.meta_leads, 0)) as meta_leads,
      SUM(COALESCE(ma.meta_purchases, 0)) as meta_purchases,
      SUM(COALESCE(ma.meta_landing_page_views, 0)) as meta_landing_page_views,
      
      -- Google Advertising Metrics (prioritize google platform data)
      SUM(COALESCE(ga.google_spend, 0)) as google_spend,
      SUM(COALESCE(ga.google_impressions, 0)) as google_impressions,
      SUM(COALESCE(ga.google_clicks, 0)) as google_clicks,
      SUM(COALESCE(ga.google_conversions, 0)) as google_conversions,
      
      -- Combined Spend
      SUM(COALESCE(ma.meta_spend, 0)) + SUM(COALESCE(ga.google_spend, 0)) as total_ad_spend,
      
      -- Conversion Rates (using hex_data as source of truth for conversions)
      SAFE_DIVIDE(SUM(h.start_flows), SUM(h.leads)) * 100 as lead_to_start_flow_rate,
      SAFE_DIVIDE(SUM(h.estimates), SUM(h.start_flows)) * 100 as start_flow_to_estimate_rate,
      SAFE_DIVIDE(SUM(h.closings), SUM(h.estimates)) * 100 as estimate_to_closing_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.closings)) * 100 as closing_to_funded_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as overall_lead_to_funded_rate,
      
      -- Cost Metrics (platform spend / hex conversions)
      SAFE_DIVIDE(SUM(COALESCE(ma.meta_spend, 0)) + SUM(COALESCE(ga.google_spend, 0)), SUM(h.leads)) as cost_per_lead,
      SAFE_DIVIDE(SUM(COALESCE(ma.meta_spend, 0)) + SUM(COALESCE(ga.google_spend, 0)), SUM(h.funded)) as cost_per_funded,
      
      -- CTR (platform metrics)
      SAFE_DIVIDE(SUM(COALESCE(ma.meta_clicks, 0)) + SUM(COALESCE(ga.google_clicks, 0)), 
                  SUM(COALESCE(ma.meta_impressions, 0)) + SUM(COALESCE(ga.google_impressions, 0))) * 100 as overall_ctr"""
    
    # Add channel-specific metrics only if analysis_depth is not "basic"
    if analysis_depth != "basic":
        base_query += f"""
      
      -- Channel-specific metrics (based on utm_medium from hex_data)
      -- Meta platform spend for paid-social campaigns
      ,SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_spend, 0) ELSE 0 END) as paid_social_spend,
      -- Google platform spend for paid-search/paid-video campaigns  
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_spend, 0) ELSE 0 END) as paid_search_video_spend,
      
      -- Meta platform impressions for paid-social campaigns
      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_impressions, 0) ELSE 0 END) as paid_social_impressions,
      -- Google platform impressions for paid-search/paid-video campaigns
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_impressions, 0) ELSE 0 END) as paid_search_video_impressions,
      
      -- Meta platform clicks for paid-social campaigns
      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_clicks, 0) ELSE 0 END) as paid_social_clicks,
      -- Google platform clicks for paid-search/paid-video campaigns
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_clicks, 0) ELSE 0 END) as paid_search_video_clicks,
      
      -- Hex conversion data by channel (source of truth for conversions)
      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.leads ELSE 0 END) as paid_social_leads,
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.leads ELSE 0 END) as paid_search_video_leads,
      
      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.estimates ELSE 0 END) as paid_social_estimates,
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.estimates ELSE 0 END) as paid_search_video_estimates,
      
      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.closings ELSE 0 END) as paid_social_closings,
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.closings ELSE 0 END) as paid_search_video_closings,
      
      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.funded ELSE 0 END) as paid_social_funded,
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.funded ELSE 0 END) as paid_search_video_funded,
      
      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.rpts ELSE 0 END) as paid_social_rpts,
      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.rpts ELSE 0 END) as paid_search_video_rpts,
      
      -- Channel-specific conversion rates (platform clicks / hex conversions)
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_clicks, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_impressions, 0) ELSE 0 END)) * 100 as paid_social_ctr,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_clicks, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_impressions, 0) ELSE 0 END)) * 100 as paid_search_video_ctr,
      
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.estimates ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.leads ELSE 0 END)) * 100 as paid_social_estimate_cvr,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.estimates ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.leads ELSE 0 END)) * 100 as paid_search_video_estimate_cvr,
      
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.closings ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.estimates ELSE 0 END)) * 100 as paid_social_closing_cvr,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.closings ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.estimates ELSE 0 END)) * 100 as paid_search_video_closing_cvr,
      
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.funded ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.closings ELSE 0 END)) * 100 as paid_social_funded_cvr,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.funded ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.closings ELSE 0 END)) * 100 as paid_search_video_funded_cvr,
      
      -- Channel-specific cost metrics (platform spend / platform impressions|clicks or hex conversions)
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_impressions, 0) ELSE 0 END) / 1000) as paid_social_cpm,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_impressions, 0) ELSE 0 END) / 1000) as paid_search_video_cpm,
      
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_clicks, 0) ELSE 0 END)) as paid_social_cpc,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_clicks, 0) ELSE 0 END)) as paid_search_video_cpc,
      
      -- Platform spend / hex conversions (best of both worlds)
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.leads ELSE 0 END)) as paid_social_cost_per_lead,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.estimates ELSE 0 END)) as paid_social_cost_per_estimate,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.closings ELSE 0 END)) as paid_social_cost_per_closing,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.meta_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.funded ELSE 0 END)) as paid_social_cost_per_funded,
      
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.leads ELSE 0 END)) as paid_search_video_cost_per_lead,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.estimates ELSE 0 END)) as paid_search_video_cost_per_estimate,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.closings ELSE 0 END)) as paid_search_video_cost_per_closing,
      SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.google_spend, 0) ELSE 0 END), 
                  SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.funded ELSE 0 END)) as paid_search_video_cost_per_funded"""

    base_query += f"""

    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    LEFT JOIN meta_aggregated_by_campaign ma ON h.utm_campaign = ma.utm_campaign AND h.date = ma.date
    LEFT JOIN google_aggregated_by_campaign ga ON h.utm_campaign = ga.utm_campaign AND h.date = ga.date
    WHERE {date_filter}
    """
    
    return base_query

def get_campaign_performance_query(last_7_days_start, last_7_days_end):
    """Generate campaign performance query"""
    return f"""
    WITH meta_by_campaign AS (
      SELECT 
        mm.utm_campaign,
        m.date,
        SUM(SAFE_CAST(m.spend AS FLOAT64)) as spend
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
      JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
        ON mm.adset_name_mapped = m.adset_name
      GROUP BY mm.utm_campaign, m.date
    )
    
    SELECT 
      h.utm_campaign,
      mm.campaign_name_mapped AS campaign_name,
      mm.adset_name_mapped,
      
      -- Total Metrics
      SUM(h.leads) as total_leads,
      SUM(h.funded) as total_funded,
      SUM(COALESCE(mc.spend, 0)) as total_meta_spend,
      SUM(COALESCE(g.spend_usd, 0)) as total_google_spend,
      SUM(COALESCE(mc.spend, 0)) + SUM(COALESCE(g.spend_usd, 0)) as total_combined_spend,
      
      -- Performance Metrics
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as lead_to_funded_rate,
      SAFE_DIVIDE(SUM(COALESCE(mc.spend, 0)) + SUM(COALESCE(g.spend_usd, 0)), SUM(h.leads)) as cost_per_lead,
      SAFE_DIVIDE(SUM(COALESCE(mc.spend, 0)) + SUM(COALESCE(g.spend_usd, 0)), SUM(h.funded)) as cost_per_funded,
      
      -- Channel Performance
      SAFE_DIVIDE(SUM(COALESCE(mc.spend, 0)), SUM(COALESCE(mc.spend, 0)) + SUM(COALESCE(g.spend_usd, 0))) * 100 as meta_spend_percentage,
      SAFE_DIVIDE(SUM(COALESCE(g.spend_usd, 0)), SUM(COALESCE(mc.spend, 0)) + SUM(COALESCE(g.spend_usd, 0))) * 100 as google_spend_percentage

    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm ON h.utm_campaign = mm.utm_campaign
    LEFT JOIN meta_by_campaign mc ON h.utm_campaign = mc.utm_campaign AND h.date = mc.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.utm_campaign = g.campaign_name AND h.date = g.date
    WHERE h.date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"
    GROUP BY h.utm_campaign, mm.campaign_name_mapped, mm.adset_name_mapped
    HAVING SUM(h.leads) > 0
    ORDER BY total_combined_spend DESC
    """

def execute_comprehensive_sql(request: TrendAnalysisRequest):
    """Execute comprehensive SQL queries with better error handling"""
    
    try:
        # Parse date ranges from request
        since_date = datetime.strptime(request.date_range["since"], "%Y-%m-%d").date()
        until_date = datetime.strptime(request.date_range["until"], "%Y-%m-%d").date()
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid date format in request: {e}")
    
    # Calculate date ranges
    yesterday = until_date - timedelta(days=1)
    same_day_last_week = yesterday - timedelta(days=7)
    last_7_days_end = yesterday
    last_7_days_start = yesterday - timedelta(days=6)
    previous_7_days_end = same_day_last_week
    previous_7_days_start = previous_7_days_end - timedelta(days=6)
    
    print(f"Date calculations:")
    print(f"  Yesterday: {yesterday}")
    print(f"  Same day last week: {same_day_last_week}")
    print(f"  Last 7 days: {last_7_days_start} to {last_7_days_end}")
    print(f"  Previous 7 days: {previous_7_days_start} to {previous_7_days_end}")
    
    # Define queries for each period
    queries = {}
    
    try:
        queries['yesterday'] = get_comprehensive_query(f'h.date = "{yesterday}"', 'Yesterday', request.analysis_depth)
        queries['same_day_last_week'] = get_comprehensive_query(f'h.date = "{same_day_last_week}"', 'Same Day Last Week', request.analysis_depth)
        queries['last_7_days'] = get_comprehensive_query(f'h.date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"', 'Last 7 Days', request.analysis_depth)
        
        if request.include_historical:
            queries['previous_7_days'] = get_comprehensive_query(f'h.date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}"', 'Previous 7 Days', request.analysis_depth)
        
        if request.analysis_depth != "basic":
            queries['campaign_performance'] = get_campaign_performance_query(last_7_days_start, last_7_days_end)
            
    except Exception as e:
        print(f"Error building queries: {e}")
        raise e
    
    results = {}
    
    for period, sql in queries.items():
        print(f"\n=== EXECUTING {period.upper()} ===")
        
        try:
            if not bigquery_client:
                raise Exception("BigQuery client not available")
                
            df = bigquery_client.query(sql).to_dataframe()
            print(f"Query returned {len(df)} rows")
            
            if not df.empty:
                if period == 'campaign_performance':
                    # For campaign performance, return all rows
                    campaigns = []
                    for _, row in df.iterrows():
                        campaign_data = {}
                        for col in df.columns:
                            value = row[col]
                            if pd.isna(value):
                                campaign_data[col] = 0
                            elif hasattr(value, 'item'):
                                campaign_data[col] = value.item()
                            else:
                                campaign_data[col] = value
                        campaigns.append(campaign_data)
                    results[period] = campaigns
                else:
                    # For aggregate queries, return single row
                    if len(df) > 0:
                        result = {}
                        for col in df.columns:
                            value = df.iloc[0][col]
                            if pd.isna(value):
                                result[col] = 0
                            elif hasattr(value, 'item'):
                                result[col] = value.item()
                            else:
                                result[col] = value
                        results[period] = result
                        
                        # Debug key metrics
                        print(f"RESULTS {period}:")
                        print(f"  Total leads: {result.get('hex_leads', 0)}")
                        print(f"  Meta clicks: {result.get('meta_clicks', 0)}")
                        print(f"  Paid social clicks: {result.get('paid_social_clicks', 0)}")
                    else:
                        print(f"Empty dataframe for {period}")
                        results[period] = {}
            else:
                print(f"No data returned for {period}")
                results[period] = {} if period != 'campaign_performance' else []
                
        except Exception as e:
            print(f"ERROR executing {period}: {e}")
            print(f"SQL that failed: {sql[:500]}...")
            # Return empty structure instead of failing completely
            results[period] = {} if period != 'campaign_performance' else []
    
    print(f"\n=== FINAL RESULTS STRUCTURE ===")
    print(f"Results keys: {list(results.keys())}")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        elif isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
        else:
            print(f"  {key}: {type(value)}")
    
    return results

def generate_enhanced_claude_analysis(data):
    """Generate enhanced cross-platform analysis with specific numbers and outlier detection"""
    
    def safe_get(data_dict, key, default=0):
        if not isinstance(data_dict, dict):
            return default
        val = data_dict.get(key, default)
        return float(val) if val is not None else default
    
    def calc_change(current, previous):
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return ((current - previous) / previous) * 100
    
    def format_currency(value):
        return f"${value:,.0f}" if value >= 1000 else f"${value:.0f}"
    
    def format_number(value):
        return f"{value:,.0f}" if value >= 1000 else f"{value:.0f}"
    
    def format_percentage(value):
        return f"{value:.1f}%" if value else "0.0%"
    
    # Validate input data
    if not isinstance(data, dict):
        return "Analysis temporarily unavailable due to data processing issue", "yellow"
    
    # Extract data with safety checks
    yesterday = data.get('yesterday', {}) or {}
    same_day_last_week = data.get('same_day_last_week', {}) or {}
    last_7_days = data.get('last_7_days', {}) or {}
    previous_7_days = data.get('previous_7_days', {}) or {}
    campaign_performance = data.get('campaign_performance', []) or []
    
    # Ensure campaign_performance is a list
    if not isinstance(campaign_performance, list):
        campaign_performance = []
    
    insights = []
    
    try:
        # Cross-platform correlation analysis
        ps_spend_curr = safe_get(yesterday, 'paid_social_spend')
        ps_spend_prev = safe_get(same_day_last_week, 'paid_social_spend')
        psv_spend_curr = safe_get(yesterday, 'paid_search_video_spend')
        psv_spend_prev = safe_get(same_day_last_week, 'paid_search_video_spend')
        
        ps_ctr_curr = safe_get(yesterday, 'paid_social_ctr')
        ps_ctr_prev = safe_get(same_day_last_week, 'paid_social_ctr')
        psv_leads_curr = safe_get(yesterday, 'paid_search_video_leads')
        psv_leads_prev = safe_get(same_day_last_week, 'paid_search_video_leads')
        
        ps_leads_curr = safe_get(yesterday, 'paid_social_leads')
        ps_leads_prev = safe_get(same_day_last_week, 'paid_social_leads')
        
        # Calculate changes
        ps_spend_change = calc_change(ps_spend_curr, ps_spend_prev)
        psv_spend_change = calc_change(psv_spend_curr, psv_spend_prev)
        ps_ctr_change = calc_change(ps_ctr_curr, ps_ctr_prev)
        psv_leads_change = calc_change(psv_leads_curr, psv_leads_prev)
        ps_leads_change = calc_change(ps_leads_curr, ps_leads_prev)
        
        # Brand awareness effect detection
        if ps_ctr_change > 15 and psv_leads_change > 10:
            insights.append(f"Social CTR surge (+{ps_ctr_change:.1f}% to {format_percentage(ps_ctr_curr)}) driving Search+Video lead growth (+{psv_leads_change:.1f}% to {format_number(psv_leads_curr)} leads) - strong brand awareness effect")
        
        # Spend reallocation insights
        if abs(ps_spend_change) > 20 or abs(psv_spend_change) > 20:
            total_spend = ps_spend_curr + psv_spend_curr
            if total_spend > 0:
                ps_share = ps_spend_curr / total_spend * 100
                psv_share = psv_spend_curr / total_spend * 100
                
                if ps_spend_change > 20 and psv_spend_change < -10:
                    insights.append(f"Major spend shift: Social increased to {format_currency(ps_spend_curr)} (+{ps_spend_change:.1f}%) while Search+Video dropped to {format_currency(psv_spend_curr)} ({psv_spend_change:.1f}%) - now {format_percentage(ps_share)} social vs {format_percentage(psv_share)} search allocation")
                elif psv_spend_change > 20 and ps_spend_change < -10:
                    insights.append(f"Search+Video investment surge: {format_currency(psv_spend_curr)} (+{psv_spend_change:.1f}%) vs Social {format_currency(ps_spend_curr)} ({ps_spend_change:.1f}%) - {format_percentage(psv_share)} search vs {format_percentage(ps_share)} social allocation")
        
        # Efficiency comparison
        ps_cost_per_lead = safe_get(yesterday, 'paid_social_cost_per_lead')
        psv_cost_per_lead = safe_get(yesterday, 'paid_search_video_cost_per_lead')
        
        if ps_cost_per_lead > 0 and psv_cost_per_lead > 0:
            if ps_cost_per_lead < psv_cost_per_lead * 0.7:
                efficiency_diff = ((psv_cost_per_lead - ps_cost_per_lead) / psv_cost_per_lead) * 100
                insights.append(f"Social significantly outperforming: {format_currency(ps_cost_per_lead)} cost/lead vs Search+Video {format_currency(psv_cost_per_lead)} ({efficiency_diff:.0f}% more efficient)")
            elif psv_cost_per_lead < ps_cost_per_lead * 0.7:
                efficiency_diff = ((ps_cost_per_lead - psv_cost_per_lead) / ps_cost_per_lead) * 100
                insights.append(f"Search+Video dominating efficiency: {format_currency(psv_cost_per_lead)} cost/lead vs Social {format_currency(ps_cost_per_lead)} ({efficiency_diff:.0f}% more efficient)")
        
        # Campaign outlier detection
        if campaign_performance and len(campaign_performance) > 0:
            total_spends = []
            cost_per_leads = []
            funded_rates = []
            
            for camp in campaign_performance:
                if isinstance(camp, dict):
                    spend = safe_get(camp, 'total_combined_spend')
                    cpl = safe_get(camp, 'cost_per_lead')
                    funded = safe_get(camp, 'lead_to_funded_rate')
                    
                    if spend > 0:
                        total_spends.append(spend)
                    if cpl > 0:
                        cost_per_leads.append(cpl)
                    if funded > 0:
                        funded_rates.append(funded)
            
            # High spend outlier
            if total_spends and len(total_spends) > 1:
                avg_spend = sum(total_spends) / len(total_spends)
                max_spend_campaign = None
                max_spend = 0
                
                for camp in campaign_performance:
                    if isinstance(camp, dict):
                        spend = safe_get(camp, 'total_combined_spend')
                        if spend > max_spend:
                            max_spend = spend
                            max_spend_campaign = camp
                
                if max_spend_campaign and max_spend > avg_spend * 2:
                    campaign_name = max_spend_campaign.get('campaign_name', max_spend_campaign.get('utm_campaign', 'Unknown'))
                    if isinstance(campaign_name, str):
                        campaign_name = campaign_name[:30]
                    else:
                        campaign_name = 'Unknown'
                    insights.append(f"High spend outlier: '{campaign_name}' at {format_currency(max_spend)} ({(max_spend/avg_spend):.1f}x average)")
    
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        insights = ["Analysis temporarily simplified due to data processing issue"]
    
    # Generate final message
    if not insights:
        message = "Performance relatively stable across channels with no significant cross-platform trends or outliers detected"
    else:
        message = " | ".join(insights[:3])  # Take top 3 insights
    
    # Determine color code
    high_impact_keywords = ['surge', 'dominating', 'significantly outperforming', 'champion', 'efficient performer']
    warning_keywords = ['outlier', 'shift', 'dropped']
    
    if any(keyword in message.lower() for keyword in high_impact_keywords):
        color_code = "green"
    elif any(keyword in message.lower() for keyword in warning_keywords):
        color_code = "yellow"
    else:
        color_code = "green"
    
    return message, color_code

def generate_claude_analysis(data):
    """Generate analysis with enhanced cross-platform insights"""
    
    # Validate input data
    if data is None:
        print("ERROR: Data is None in generate_claude_analysis")
        return {
            "message": "Analysis temporarily unavailable due to data processing issue",
            "colorCode": "yellow",
            "paidSocial": {
                "dayOverDayPulse": {},
                "weekOverWeekPulse": {}
            },
            "paidSearchVideo": {
                "dayOverDayPulse": {},
                "weekOverWeekPulse": {}
            }
        }
    
    if not isinstance(data, dict):
        print(f"ERROR: Data is not a dict in generate_claude_analysis, it's {type(data)}")
        return {
            "message": "Analysis temporarily unavailable due to data format issue",
            "colorCode": "yellow",
            "paidSocial": {
                "dayOverDayPulse": {},
                "weekOverWeekPulse": {}
            },
            "paidSearchVideo": {
                "dayOverDayPulse": {},
                "weekOverWeekPulse": {}
            }
        }
    
    def safe_get(data_dict, key, default=0):
        """Safely get value from dict with multiple fallbacks"""
        if not isinstance(data_dict, dict):
            return default
        val = data_dict.get(key, default)
        if val is None:
            return default
        try:
            return float(val) if val != default else default
        except (TypeError, ValueError):
            return default
    
    def calc_change(current, previous):
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return ((current - previous) / previous) * 100
    
    def format_value_with_change(current, previous, is_percentage=False, is_currency=False):
        change = calc_change(current, previous)
        if is_percentage:
            return f"{current:.1f}% ({change:+.1f}%)"
        elif is_currency:
            return f"${current:.2f} ({change:+.1f}%)"
        else:
            return f"{current:,.0f} ({change:+.1f}%)"
    
    # Extract data with multiple safety checks
    yesterday = data.get('yesterday', {}) or {}
    same_day_last_week = data.get('same_day_last_week', {}) or {}
    last_7_days = data.get('last_7_days', {}) or {}
    previous_7_days = data.get('previous_7_days', {}) or {}
    
    print(f"Data extraction check:")
    print(f"  Yesterday type: {type(yesterday)}, keys: {list(yesterday.keys()) if isinstance(yesterday, dict) else 'Not dict'}")
    print(f"  Same day last week type: {type(same_day_last_week)}")
    print(f"  Last 7 days type: {type(last_7_days)}")
    print(f"  Previous 7 days type: {type(previous_7_days)}")
    
    try:
        # Generate enhanced message and color code
        message, color_code = generate_enhanced_claude_analysis(data)
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        message = "Performance analysis temporarily simplified"
        color_code = "green"
    
    try:
        # PAID SOCIAL DAY-OVER-DAY
        ps_dod = {
            "totalSpend": format_value_with_change(
                safe_get(yesterday, 'paid_social_spend'),
                safe_get(same_day_last_week, 'paid_social_spend'),
                is_currency=True
            ),
            "totalImpressions": format_value_with_change(
                safe_get(yesterday, 'paid_social_impressions'),
                safe_get(same_day_last_week, 'paid_social_impressions')
            ),
            "cpm": format_value_with_change(
                safe_get(yesterday, 'paid_social_cpm'),
                safe_get(same_day_last_week, 'paid_social_cpm'),
                is_currency=True
            ),
            "ctr": format_value_with_change(
                safe_get(yesterday, 'paid_social_ctr'),
                safe_get(same_day_last_week, 'paid_social_ctr'),
                is_percentage=True
            ),
            "totalClicks": format_value_with_change(
                safe_get(yesterday, 'paid_social_clicks'),
                safe_get(same_day_last_week, 'paid_social_clicks')
            ),
            "cpc": format_value_with_change(
                safe_get(yesterday, 'paid_social_cpc'),
                safe_get(same_day_last_week, 'paid_social_cpc'),
                is_currency=True
            ),
            "totalLeads": format_value_with_change(
                safe_get(yesterday, 'paid_social_leads'),
                safe_get(same_day_last_week, 'paid_social_leads')
            ),
            "costPerLead": format_value_with_change(
                safe_get(yesterday, 'paid_social_cost_per_lead'),
                safe_get(same_day_last_week, 'paid_social_cost_per_lead'),
                is_currency=True
            ),
            "estimateCVR": format_value_with_change(
                safe_get(yesterday, 'paid_social_estimate_cvr'),
                safe_get(same_day_last_week, 'paid_social_estimate_cvr'),
                is_percentage=True
            ),
            "totalEstimates": format_value_with_change(
                safe_get(yesterday, 'paid_social_estimates'),
                safe_get(same_day_last_week, 'paid_social_estimates')
            ),
            "costPerEstimate": format_value_with_change(
                safe_get(yesterday, 'paid_social_cost_per_estimate'),
                safe_get(same_day_last_week, 'paid_social_cost_per_estimate'),
                is_currency=True
            ),
            "closingsCVR": format_value_with_change(
                safe_get(yesterday, 'paid_social_closing_cvr'),
                safe_get(same_day_last_week, 'paid_social_closing_cvr'),
                is_percentage=True
            ),
            "totalClosings": format_value_with_change(
                safe_get(yesterday, 'paid_social_closings'),
                safe_get(same_day_last_week, 'paid_social_closings')
            ),
            "costPerClosing": format_value_with_change(
                safe_get(yesterday, 'paid_social_cost_per_closing'),
                safe_get(same_day_last_week, 'paid_social_cost_per_closing'),
                is_currency=True
            ),
            "fundedCVR": format_value_with_change(
                safe_get(yesterday, 'paid_social_funded_cvr'),
                safe_get(same_day_last_week, 'paid_social_funded_cvr'),
                is_percentage=True
            ),
            "totalFunded": format_value_with_change(
                safe_get(yesterday, 'paid_social_funded'),
                safe_get(same_day_last_week, 'paid_social_funded')
            ),
            "costPerFunded": format_value_with_change(
                safe_get(yesterday, 'paid_social_cost_per_funded'),
                safe_get(same_day_last_week, 'paid_social_cost_per_funded'),
                is_currency=True
            ),
            "totalRPTs": format_value_with_change(
                safe_get(yesterday, 'paid_social_rpts'),
                safe_get(same_day_last_week, 'paid_social_rpts')
            )
        }
        
        # PAID SOCIAL WEEK-OVER-WEEK
        ps_wow = {
            "totalSpend": format_value_with_change(
                safe_get(last_7_days, 'paid_social_spend'),
                safe_get(previous_7_days, 'paid_social_spend'),
                is_currency=True
            ),
            "totalImpressions": format_value_with_change(
                safe_get(last_7_days, 'paid_social_impressions'),
                safe_get(previous_7_days, 'paid_social_impressions')
            ),
            "cpm": format_value_with_change(
                safe_get(last_7_days, 'paid_social_cpm'),
                safe_get(previous_7_days, 'paid_social_cpm'),
                is_currency=True
            ),
            "ctr": format_value_with_change(
                safe_get(last_7_days, 'paid_social_ctr'),
                safe_get(previous_7_days, 'paid_social_ctr'),
                is_percentage=True
            ),
            "totalClicks": format_value_with_change(
                safe_get(last_7_days, 'paid_social_clicks'),
                safe_get(previous_7_days, 'paid_social_clicks')
            ),
            "cpc": format_value_with_change(
                safe_get(last_7_days, 'paid_social_cpc'),
                safe_get(previous_7_days, 'paid_social_cpc'),
                is_currency=True
            ),
            "totalLeads": format_value_with_change(
                safe_get(last_7_days, 'paid_social_leads'),
                safe_get(previous_7_days, 'paid_social_leads')
            ),
            "costPerLead": format_value_with_change(
                safe_get(last_7_days, 'paid_social_cost_per_lead'),
                safe_get(previous_7_days, 'paid_social_cost_per_lead'),
                is_currency=True
            ),
            "estimateCVR": format_value_with_change(
                safe_get(last_7_days, 'paid_social_estimate_cvr'),
                safe_get(previous_7_days, 'paid_social_estimate_cvr'),
                is_percentage=True
            ),
            "totalEstimates": format_value_with_change(
                safe_get(last_7_days, 'paid_social_estimates'),
                safe_get(previous_7_days, 'paid_social_estimates')
            ),
            "costPerEstimate": format_value_with_change(
                safe_get(last_7_days, 'paid_social_cost_per_estimate'),
                safe_get(previous_7_days, 'paid_social_cost_per_estimate'),
                is_currency=True
            ),
            "closingsCVR": format_value_with_change(
                safe_get(last_7_days, 'paid_social_closing_cvr'),
                safe_get(previous_7_days, 'paid_social_closing_cvr'),
                is_percentage=True
            ),
            "totalClosings": format_value_with_change(
                safe_get(last_7_days, 'paid_social_closings'),
                safe_get(previous_7_days, 'paid_social_closings')
            ),
            "costPerClosing": format_value_with_change(
                safe_get(last_7_days, 'paid_social_cost_per_closing'),
                safe_get(previous_7_days, 'paid_social_cost_per_closing'),
                is_currency=True
            ),
            "fundedCVR": format_value_with_change(
                safe_get(last_7_days, 'paid_social_funded_cvr'),
                safe_get(previous_7_days, 'paid_social_funded_cvr'),
                is_percentage=True
            ),
            "totalFunded": format_value_with_change(
                safe_get(last_7_days, 'paid_social_funded'),
                safe_get(previous_7_days, 'paid_social_funded')
            ),
            "costPerFunded": format_value_with_change(
                safe_get(last_7_days, 'paid_social_cost_per_funded'),
                safe_get(previous_7_days, 'paid_social_cost_per_funded'),
                is_currency=True
            ),
            "totalRPTs": format_value_with_change(
                safe_get(last_7_days, 'paid_social_rpts'),
                safe_get(previous_7_days, 'paid_social_rpts')
            )
        }
        
        # PAID SEARCH+VIDEO DAY-OVER-DAY
        psv_dod = {
            "totalSpend": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_spend'),
                safe_get(same_day_last_week, 'paid_search_video_spend'),
                is_currency=True
            ),
            "totalImpressions": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_impressions'),
                safe_get(same_day_last_week, 'paid_search_video_impressions')
            ),
            "cpm": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_cpm'),
                safe_get(same_day_last_week, 'paid_search_video_cpm'),
                is_currency=True
            ),
            "ctr": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_ctr'),
                safe_get(same_day_last_week, 'paid_search_video_ctr'),
                is_percentage=True
            ),
            "totalClicks": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_clicks'),
                safe_get(same_day_last_week, 'paid_search_video_clicks')
            ),
            "cpc": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_cpc'),
                safe_get(same_day_last_week, 'paid_search_video_cpc'),
                is_currency=True
            ),
            "totalLeads": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_leads'),
                safe_get(same_day_last_week, 'paid_search_video_leads')
            ),
            "costPerLead": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_cost_per_lead'),
                safe_get(same_day_last_week, 'paid_search_video_cost_per_lead'),
                is_currency=True
            ),
            "estimateCVR": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_estimate_cvr'),
                safe_get(same_day_last_week, 'paid_search_video_estimate_cvr'),
                is_percentage=True
            ),
            "totalEstimates": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_estimates'),
                safe_get(same_day_last_week, 'paid_search_video_estimates')
            ),
            "costPerEstimate": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_cost_per_estimate'),
                safe_get(same_day_last_week, 'paid_search_video_cost_per_estimate'),
                is_currency=True
            ),
            "closingsCVR": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_closing_cvr'),
                safe_get(same_day_last_week, 'paid_search_video_closing_cvr'),
                is_percentage=True
            ),
            "totalClosings": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_closings'),
                safe_get(same_day_last_week, 'paid_search_video_closings')
            ),
            "costPerClosing": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_cost_per_closing'),
                safe_get(same_day_last_week, 'paid_search_video_cost_per_closing'),
                is_currency=True
            ),
            "fundedCVR": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_funded_cvr'),
                safe_get(same_day_last_week, 'paid_search_video_funded_cvr'),
                is_percentage=True
            ),
            "totalFunded": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_funded'),
                safe_get(same_day_last_week, 'paid_search_video_funded')
            ),
            "costPerFunded": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_cost_per_funded'),
                safe_get(same_day_last_week, 'paid_search_video_cost_per_funded'),
                is_currency=True
            ),
            "totalRPTs": format_value_with_change(
                safe_get(yesterday, 'paid_search_video_rpts'),
                safe_get(same_day_last_week, 'paid_search_video_rpts')
            )
        }
        
        # PAID SEARCH+VIDEO WEEK-OVER-WEEK
        psv_wow = {
            "totalSpend": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_spend'),
                safe_get(previous_7_days, 'paid_search_video_spend'),
                is_currency=True
            ),
            "totalImpressions": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_impressions'),
                safe_get(previous_7_days, 'paid_search_video_impressions')
            ),
            "cpm": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_cpm'),
                safe_get(previous_7_days, 'paid_search_video_cpm'),
                is_currency=True
            ),
            "ctr": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_ctr'),
                safe_get(previous_7_days, 'paid_search_video_ctr'),
                is_percentage=True
            ),
            "totalClicks": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_clicks'),
                safe_get(previous_7_days, 'paid_search_video_clicks')
            ),
            "cpc": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_cpc'),
                safe_get(previous_7_days, 'paid_search_video_cpc'),
                is_currency=True
            ),
            "totalLeads": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_leads'),
                safe_get(previous_7_days, 'paid_search_video_leads')
            ),
            "costPerLead": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_cost_per_lead'),
                safe_get(previous_7_days, 'paid_search_video_cost_per_lead'),
                is_currency=True
            ),
            "estimateCVR": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_estimate_cvr'),
                safe_get(previous_7_days, 'paid_search_video_estimate_cvr'),
                is_percentage=True
            ),
            "totalEstimates": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_estimates'),
                safe_get(previous_7_days, 'paid_search_video_estimates')
            ),
            "costPerEstimate": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_cost_per_estimate'),
                safe_get(previous_7_days, 'paid_search_video_cost_per_estimate'),
                is_currency=True
            ),
            "closingsCVR": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_closing_cvr'),
                safe_get(previous_7_days, 'paid_search_video_closing_cvr'),
                is_percentage=True
            ),
            "totalClosings": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_closings'),
                safe_get(previous_7_days, 'paid_search_video_closings')
            ),
            "costPerClosing": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_cost_per_closing'),
                safe_get(previous_7_days, 'paid_search_video_cost_per_closing'),
                is_currency=True
            ),
            "fundedCVR": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_funded_cvr'),
                safe_get(previous_7_days, 'paid_search_video_funded_cvr'),
                is_percentage=True
            ),
            "totalFunded": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_funded'),
                safe_get(previous_7_days, 'paid_search_video_funded')
            ),
            "costPerFunded": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_cost_per_funded'),
                safe_get(previous_7_days, 'paid_search_video_cost_per_funded'),
                is_currency=True
            ),
            "totalRPTs": format_value_with_change(
                safe_get(last_7_days, 'paid_search_video_rpts'),
                safe_get(previous_7_days, 'paid_search_video_rpts')
            )
        }
        
        return {
            "message": message,
            "colorCode": color_code,
            "paidSocial": {
                "dayOverDayPulse": ps_dod,
                "weekOverWeekPulse": ps_wow
            },
            "paidSearchVideo": {
                "dayOverDayPulse": psv_dod,
                "weekOverWeekPulse": psv_wow
            }
        }
        
    except Exception as e:
        print(f"Error building claude analysis response: {e}")
        return {
            "message": "Analysis completed with limited data",
            "colorCode": "yellow",
            "paidSocial": {"dayOverDayPulse": {}, "weekOverWeekPulse": {}},
            "paidSearchVideo": {"dayOverDayPulse": {}, "weekOverWeekPulse": {}}
        }

@marketing_router.post("/analyze-trends", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(request: TrendAnalysisRequest):
    """Execute comprehensive SQL with channel separation and generate Claude analysis"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                message="BigQuery not available",
                error="BigQuery client not initialized"
            )
        
        # Execute the comprehensive SQL
        data = execute_comprehensive_sql(request)
        
        # Generate Claude analysis
        claude_analysis = generate_claude_analysis(data)
        
        return TrendAnalysisResponse(
            status="success",
            message="Comprehensive analysis with enhanced cross-platform insights completed",
            data={
                "claude_analysis": claude_analysis,
                "raw_data": {
                    "yesterday": data.get('yesterday', {}),
                    "same_day_last_week": data.get('same_day_last_week', {}),
                    "last_7_days": data.get('last_7_days', {}),
                    "previous_7_days": data.get('previous_7_days', {})
                },
                "campaign_performance": data.get('campaign_performance', []),
                "debug_info": {
                    "yesterday_paid_social_spend": data.get('yesterday', {}).get('paid_social_spend', 0),
                    "yesterday_paid_search_video_spend": data.get('yesterday', {}).get('paid_search_video_spend', 0),
                    "last_7_days_paid_social_spend": data.get('last_7_days', {}).get('paid_social_spend', 0),
                    "last_7_days_paid_search_video_spend": data.get('last_7_days', {}).get('paid_search_video_spend', 0),
                    "total_campaigns": len(data.get('campaign_performance', []))
                }
            }
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="Comprehensive analysis failed",
            error=str(e)
        )

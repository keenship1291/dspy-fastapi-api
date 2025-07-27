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

def execute_comprehensive_sql(request: TrendAnalysisRequest):
    """Execute comprehensive SQL queries matching the structure from paste.txt"""
    
    # Parse date ranges from request
    try:
        since_date = datetime.strptime(request.date_range["since"], "%Y-%m-%d").date()
        until_date = datetime.strptime(request.date_range["until"], "%Y-%m-%d").date()
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid date format in request: {e}")
    
    # Calculate date ranges based on request
    yesterday = until_date - timedelta(days=1)  # 2025-07-25 (one day before until)
    same_day_last_week = yesterday - timedelta(days=7)  # 2025-07-18
    last_7_days_end = yesterday
    last_7_days_start = yesterday - timedelta(days=6)  # 2025-07-19
    previous_7_days_end = same_day_last_week
    previous_7_days_start = previous_7_days_end - timedelta(days=6)  # 2025-07-12
    
    # Validate that the request date range covers our analysis period
    if since_date > previous_7_days_start:
        print(f"Warning: Request since date {since_date} is after calculated start {previous_7_days_start}")
    if until_date < last_7_days_end:
        print(f"Warning: Request until date {until_date} is before calculated end {last_7_days_end}")
    
    print(f"Date ranges from request:")
    print(f"  Request: {since_date} to {until_date}")
    print(f"  Yesterday: {yesterday}")
    print(f"  Same day last week: {same_day_last_week}")
    print(f"  Last 7 days: {last_7_days_start} to {last_7_days_end}")
    print(f"  Previous 7 days: {previous_7_days_start} to {previous_7_days_end}")
    print(f"  Include historical: {request.include_historical}")
    print(f"  Analysis depth: {request.analysis_depth}")
    
    # FIXED VERSION - Replace the get_comprehensive_query function with this:

    def get_comprehensive_query(date_filter, period_name):
        base_query = f"""
        WITH meta_aggregated AS (
          -- Pre-aggregate Meta data to prevent multiplication
          SELECT 
            mm.utm_campaign,
            m.date,
            SUM(SAFE_CAST(m.spend AS FLOAT64)) as spend,
            SUM(SAFE_CAST(m.impressions AS INT64)) as impressions,
            SUM(SAFE_CAST(m.clicks AS INT64)) as clicks,
            SUM(SAFE_CAST(m.leads AS INT64)) as leads,
            SUM(SAFE_CAST(m.purchases AS INT64)) as purchases,
            SUM(SAFE_CAST(m.landing_page_views AS INT64)) as landing_page_views
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
          JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
            ON mm.adset_name_mapped = m.adset_name
          GROUP BY mm.utm_campaign, m.date
        ),
        google_aggregated AS (
          -- Pre-aggregate Google data to prevent multiplication  
          SELECT 
            campaign_name,
            date,
            SUM(spend_usd) as spend_usd,
            SUM(impressions) as impressions,
            SUM(clicks) as clicks,
            SUM(conversions) as conversions
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
          GROUP BY campaign_name, date
        )
        
        SELECT 
          '{period_name}' as period,
          
          -- Hex Funnel Metrics
          SUM(h.leads) as hex_leads,
          SUM(h.start_flows) as hex_start_flows,
          SUM(h.estimates) as hex_estimates,
          SUM(h.closings) as hex_closings,
          SUM(h.funded) as hex_funded,
          SUM(h.rpts) as hex_rpts,
          
          -- Meta Advertising Metrics (now properly aggregated)
          SUM(ma.spend) as meta_spend,
          SUM(ma.impressions) as meta_impressions,
          SUM(ma.clicks) as meta_clicks,
          SUM(ma.leads) as meta_leads,
          SUM(ma.purchases) as meta_purchases,
          SUM(ma.landing_page_views) as meta_landing_page_views,
          
          -- Google Advertising Metrics (now properly aggregated)
          SUM(ga.spend_usd) as google_spend,
          SUM(ga.impressions) as google_impressions,
          SUM(ga.clicks) as google_clicks,
          SUM(ga.conversions) as google_conversions,
          
          -- Combined Spend
          SUM(COALESCE(ma.spend, 0)) + SUM(COALESCE(ga.spend_usd, 0)) as total_ad_spend,
          
          -- Conversion Rates
          SAFE_DIVIDE(SUM(h.start_flows), SUM(h.leads)) * 100 as lead_to_start_flow_rate,
          SAFE_DIVIDE(SUM(h.estimates), SUM(h.start_flows)) * 100 as start_flow_to_estimate_rate,
          SAFE_DIVIDE(SUM(h.closings), SUM(h.estimates)) * 100 as estimate_to_closing_rate,
          SAFE_DIVIDE(SUM(h.funded), SUM(h.closings)) * 100 as closing_to_funded_rate,
          SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as overall_lead_to_funded_rate,
          
          -- Cost Metrics
          SAFE_DIVIDE(SUM(COALESCE(ma.spend, 0)) + SUM(COALESCE(ga.spend_usd, 0)), SUM(h.leads)) as cost_per_lead,
          SAFE_DIVIDE(SUM(COALESCE(ma.spend, 0)) + SUM(COALESCE(ga.spend_usd, 0)), SUM(h.funded)) as cost_per_funded,
          
          -- CTR
          SAFE_DIVIDE(SUM(COALESCE(ma.clicks, 0)) + SUM(COALESCE(ga.clicks, 0)), 
                      SUM(COALESCE(ma.impressions, 0)) + SUM(COALESCE(ga.impressions, 0))) * 100 as overall_ctr"""
        
        # Add channel-specific metrics only if analysis_depth is not "basic"
        if request.analysis_depth != "basic":
            base_query += f"""
          
          -- Channel-specific metrics (based on utm_medium)
          ,SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.spend, 0) ELSE 0 END) as paid_social_spend,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.spend_usd, 0) ELSE 0 END) as paid_search_video_spend,
          
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.impressions, 0) ELSE 0 END) as paid_social_impressions,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.impressions, 0) ELSE 0 END) as paid_search_video_impressions,
          
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.clicks, 0) ELSE 0 END) as paid_social_clicks,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.clicks, 0) ELSE 0 END) as paid_search_video_clicks,
          
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
          
          -- Channel-specific conversion rates
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.clicks, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.impressions, 0) ELSE 0 END)) * 100 as paid_social_ctr,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.clicks, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.impressions, 0) ELSE 0 END)) * 100 as paid_search_video_ctr,
          
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
          
          -- Channel-specific cost metrics
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.spend, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.impressions, 0) ELSE 0 END) / 1000) as paid_social_cpm,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.spend_usd, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.impressions, 0) ELSE 0 END) / 1000) as paid_search_video_cpm,
          
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.spend, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.clicks, 0) ELSE 0 END)) as paid_social_cpc,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.spend_usd, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.clicks, 0) ELSE 0 END)) as paid_search_video_cpc,
          
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.spend, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.leads ELSE 0 END)) as paid_social_cost_per_lead,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.spend, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.estimates ELSE 0 END)) as paid_social_cost_per_estimate,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.spend, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.closings ELSE 0 END)) as paid_social_cost_per_closing,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN COALESCE(ma.spend, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.funded ELSE 0 END)) as paid_social_cost_per_funded,
          
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.spend_usd, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.leads ELSE 0 END)) as paid_search_video_cost_per_lead,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.spend_usd, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.estimates ELSE 0 END)) as paid_search_video_cost_per_estimate,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.spend_usd, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.closings ELSE 0 END)) as paid_search_video_cost_per_closing,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN COALESCE(ga.spend_usd, 0) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.funded ELSE 0 END)) as paid_search_video_cost_per_funded"""
    
        base_query += f"""
    
        FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
        LEFT JOIN meta_aggregated ma ON h.utm_campaign = ma.utm_campaign AND h.date = ma.date
        LEFT JOIN google_aggregated ga ON h.utm_campaign = ga.campaign_name AND h.date = ga.date
        WHERE {date_filter}
        """
        
        return base_query
    
    # Campaign performance query using the proper column mapping
    def get_campaign_performance_query():
        return f"""
        SELECT 
          h.utm_campaign,
          mm.campaign_name_mapped AS campaign_name,
          mm.adset_name_mapped,
          
          -- Total Metrics
          SUM(h.leads) as total_leads,
          SUM(h.funded) as total_funded,
          SUM(SAFE_CAST(m.spend AS FLOAT64)) as total_meta_spend,
          SUM(g.spend_usd) as total_google_spend,
          SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd) as total_combined_spend,
          
          -- Performance Metrics
          SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as lead_to_funded_rate,
          SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.leads)) as cost_per_lead,
          SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.funded)) as cost_per_funded,
          
          -- Channel Performance
          SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)), SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd)) * 100 as meta_spend_percentage,
          SAFE_DIVIDE(SUM(g.spend_usd), SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd)) * 100 as google_spend_percentage

        FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
        LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm ON h.utm_campaign = mm.utm_campaign
        LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m ON mm.adset_name_mapped = m.adset_name AND h.date = m.date
        LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.utm_campaign = g.campaign_name AND h.date = g.date
        WHERE h.date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"
        GROUP BY h.utm_campaign, mm.campaign_name_mapped, mm.adset_name_mapped
        HAVING SUM(h.leads) > 0
        ORDER BY total_combined_spend DESC
        """
    
    # Define queries for each period
    queries = {}
    
    queries['yesterday'] = get_comprehensive_query(f'h.date = "{yesterday}"', 'Yesterday')
    queries['same_day_last_week'] = get_comprehensive_query(f'h.date = "{same_day_last_week}"', 'Same Day Last Week')
    queries['last_7_days'] = get_comprehensive_query(f'h.date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"', 'Last 7 Days')
    
    # Only include previous 7 days if include_historical is True
    if request.include_historical:
        queries['previous_7_days'] = get_comprehensive_query(f'h.date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}"', 'Previous 7 Days')
    
    # Campaign performance query (always include if analysis_depth is not basic)
    if request.analysis_depth != "basic":
        queries['campaign_performance'] = get_campaign_performance_query()
    
    results = {}
    
    for period, sql in queries.items():
        print(f"\n=== EXECUTING {period.upper()} ===")
        
        try:
            df = bigquery_client.query(sql).to_dataframe()
            print(f"Query returned {len(df)} rows")
            
            if not df.empty:
                if period == 'campaign_performance':
                    # For campaign performance, return all rows
                    campaigns = []
                    for _, row in df.iterrows():
                        campaign_data = row.to_dict()
                        # Clean the data
                        for key, value in campaign_data.items():
                            if pd.isna(value):
                                campaign_data[key] = 0
                            elif hasattr(value, 'item'):
                                campaign_data[key] = value.item()
                        campaigns.append(campaign_data)
                    results[period] = campaigns
                else:
                    # For aggregate queries, return single row
                    result = df.iloc[0].to_dict()
                    # Clean the data
                    for key, value in result.items():
                        if pd.isna(value):
                            result[key] = 0
                        elif hasattr(value, 'item'):
                            result[key] = value.item()
                        else:
                            result[key] = value
                    
                    results[period] = result
                    
                    # Debug print key metrics
                    paid_social_spend = result.get('paid_social_spend', 0) or 0
                    paid_search_video_spend = result.get('paid_search_video_spend', 0) or 0
                    total_spend = result.get('total_ad_spend', 0) or 0
                    total_leads = result.get('hex_leads', 0) or 0
                    
                    print(f"RESULTS {period}:")
                    print(f"  Paid Social: ${paid_social_spend:,.0f}")
                    print(f"  Paid Search+Video: ${paid_search_video_spend:,.0f}")
                    print(f"  Total: ${total_spend:,.0f}, Leads: {total_leads}")
                
            else:
                print(f"No data returned for {period}")
                results[period] = {} if period != 'campaign_performance' else []
                
        except Exception as e:
            print(f"ERROR executing {period}: {e}")
            results[period] = {} if period != 'campaign_performance' else []
    
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
    
    def format_value_with_change(current, previous, is_percentage=False, is_currency=False):
        change = calc_change(current, previous)
        if is_percentage:
            return f"{current:.1f}% ({change:+.1f}%)"
        elif is_currency:
            return f"${current:.2f} ({change:+.1f}%)"
        else:
            return f"{current:,.0f} ({change:+.1f}%)"
    
    def format_currency(value):
        return f"${value:,.0f}" if value >= 1000 else f"${value:.0f}"
    
    def format_number(value):
        return f"{value:,.0f}" if value >= 1000 else f"{value:.0f}"
    
    def format_percentage(value):
        return f"{value:.1f}%" if value else "0.0%"
    
    # Extract data with safety checks
    yesterday = data.get('yesterday', {}) if isinstance(data, dict) else {}
    same_day_last_week = data.get('same_day_last_week', {}) if isinstance(data, dict) else {}
    last_7_days = data.get('last_7_days', {}) if isinstance(data, dict) else {}
    previous_7_days = data.get('previous_7_days', {}) if isinstance(data, dict) else {}
    campaign_performance = data.get('campaign_performance', []) if isinstance(data, dict) else []
    
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
            
            # Cost efficiency outliers
            if cost_per_leads and len(cost_per_leads) > 1:
                avg_cpl = sum(cost_per_leads) / len(cost_per_leads)
                best_campaign = None
                best_cpl = float('inf')
                
                for camp in campaign_performance:
                    if isinstance(camp, dict):
                        cpl = safe_get(camp, 'cost_per_lead')
                        if 0 < cpl < avg_cpl * 0.5 and cpl < best_cpl:
                            best_cpl = cpl
                            best_campaign = camp
                
                if best_campaign:
                    campaign_name = best_campaign.get('campaign_name', best_campaign.get('utm_campaign', 'Unknown'))
                    if isinstance(campaign_name, str):
                        campaign_name = campaign_name[:30]
                    else:
                        campaign_name = 'Unknown'
                    insights.append(f"Efficient performer: '{campaign_name}' at {format_currency(best_cpl)} ({avg_cpl/best_cpl:.1f}x better than average)")
            
            # Conversion champion
            if funded_rates and len(funded_rates) > 1:
                avg_funded_rate = sum(funded_rates) / len(funded_rates)
                best_funded_campaign = None
                best_funded_rate = 0
                
                for camp in campaign_performance:
                    if isinstance(camp, dict):
                        funded = safe_get(camp, 'lead_to_funded_rate')
                        if funded > avg_funded_rate * 1.5 and funded > best_funded_rate:
                            best_funded_rate = funded
                            best_funded_campaign = camp
                
                if best_funded_campaign:
                    campaign_name = best_funded_campaign.get('campaign_name', best_funded_campaign.get('utm_campaign', 'Unknown'))
                    if isinstance(campaign_name, str):
                        campaign_name = campaign_name[:30]
                    else:
                        campaign_name = 'Unknown'
                    insights.append(f"Conversion champion: '{campaign_name}' with {format_percentage(best_funded_rate)} funded rate ({(best_funded_rate/avg_funded_rate):.1f}x average)")
    
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        insights = [f"Analysis temporarily simplified due to data processing issue"]
    
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
    
    def safe_get(data_dict, key, default=0):
        val = data_dict.get(key, default)
        return float(val) if val is not None else default
    
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
    
    # Extract data for both channels and periods
    yesterday = data.get('yesterday', {})
    same_day_last_week = data.get('same_day_last_week', {})
    last_7_days = data.get('last_7_days', {})
    previous_7_days = data.get('previous_7_days', {})
    
    # Generate enhanced message and color code
    message, color_code = generate_enhanced_claude_analysis(data)
    
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

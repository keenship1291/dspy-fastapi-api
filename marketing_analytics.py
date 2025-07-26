# marketing_analytics.py - FIXED WITH PROPER JOINED QUERIES
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
        "message": "Marketing Analytics API - Fixed with Proper Joins",
        "version": "3.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE
    }

def get_comprehensive_data():
    """Get comprehensive data using separate queries first to debug, then try joins"""
    
    # Calculate date ranges
    current_date = date.today()  # 2025-07-26
    yesterday = current_date - timedelta(days=1)  # 2025-07-25
    same_day_last_week = yesterday - timedelta(days=7)  # 2025-07-18
    
    # Last 7 days: July 19-25
    last_7_days_end = yesterday
    last_7_days_start = yesterday - timedelta(days=6)
    
    # Previous 7 days: July 12-18  
    previous_7_days_end = same_day_last_week
    previous_7_days_start = previous_7_days_end - timedelta(days=6)
    
    print(f"DEBUG: Yesterday: {yesterday}")
    print(f"DEBUG: Same day last week: {same_day_last_week}")
    print(f"DEBUG: Last 7 days: {last_7_days_start} to {last_7_days_end}")
    print(f"DEBUG: Previous 7 days: {previous_7_days_start} to {previous_7_days_end}")
    
    # First, let's check what data exists in each table for debugging
    debug_queries = {
        'hex_yesterday': f"""
            SELECT COUNT(*) as count, SUM(leads) as total_leads, SUM(spend) as total_spend 
            FROM `{PROJECT_ID}.{DATASET_ID}.hex_data` 
            WHERE date = "{yesterday}"
        """,
        'meta_yesterday': f"""
            SELECT COUNT(*) as count, SUM(SAFE_CAST(spend AS FLOAT64)) as total_spend 
            FROM `{PROJECT_ID}.{DATASET_ID}.meta_data` 
            WHERE date = "{yesterday}"
        """,
        'google_yesterday': f"""
            SELECT COUNT(*) as count, SUM(spend_usd) as total_spend 
            FROM `{PROJECT_ID}.{DATASET_ID}.google_data` 
            WHERE date = "{yesterday}"
        """
    }
    
    print("=== DEBUG: Checking individual tables ===")
    for table_name, query in debug_queries.items():
        try:
            result = bigquery_client.query(query).to_dataframe()
            if not result.empty:
                print(f"{table_name}: {result.iloc[0].to_dict()}")
            else:
                print(f"{table_name}: No data")
        except Exception as e:
            print(f"{table_name}: Error - {e}")
    
    # Simplified approach - get data from each table separately first
    def get_period_data(date_filter, period_name):
        """Get data for a specific period from each table separately"""
        
        # Hex data
        hex_query = f"""
            SELECT 
                SUM(leads) as hex_leads,
                SUM(start_flows) as hex_start_flows,
                SUM(estimates) as hex_estimates,
                SUM(closings) as hex_closings,
                SUM(funded) as hex_funded,
                SUM(rpts) as hex_rpts,
                SAFE_DIVIDE(SUM(start_flows), SUM(leads)) * 100 as lead_to_start_flow_rate,
                SAFE_DIVIDE(SUM(estimates), SUM(start_flows)) * 100 as start_flow_to_estimate_rate,
                SAFE_DIVIDE(SUM(closings), SUM(estimates)) * 100 as estimate_to_closing_rate,
                SAFE_DIVIDE(SUM(funded), SUM(closings)) * 100 as closing_to_funded_rate,
                SAFE_DIVIDE(SUM(funded), SUM(leads)) * 100 as overall_lead_to_funded_rate
            FROM `{PROJECT_ID}.{DATASET_ID}.hex_data` 
            WHERE {date_filter}
            AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        """
        
        # Meta data
        meta_query = f"""
            SELECT 
                SUM(SAFE_CAST(spend AS FLOAT64)) as meta_spend,
                SUM(SAFE_CAST(impressions AS INT64)) as meta_impressions,
                SUM(SAFE_CAST(clicks AS INT64)) as meta_clicks,
                SUM(SAFE_CAST(leads AS INT64)) as meta_leads,
                SUM(SAFE_CAST(purchases AS INT64)) as meta_purchases,
                SUM(SAFE_CAST(landing_page_views AS INT64)) as meta_landing_page_views,
                AVG(SAFE_CAST(ctr AS FLOAT64)) as meta_avg_ctr,
                AVG(SAFE_CAST(cpc AS FLOAT64)) as meta_avg_cpc,
                AVG(SAFE_CAST(cpm AS FLOAT64)) as meta_avg_cpm
            FROM `{PROJECT_ID}.{DATASET_ID}.meta_data` 
            WHERE {date_filter}
        """
        
        # Google data
        google_query = f"""
            SELECT 
                SUM(spend_usd) as google_spend,
                SUM(impressions) as google_impressions,
                SUM(clicks) as google_clicks,
                SUM(conversions) as google_conversions,
                AVG(ctr_percent) as google_avg_ctr,
                AVG(cpc_usd) as google_avg_cpc,
                AVG(cpm_usd) as google_avg_cpm,
                AVG(cpa_usd) as google_avg_cpa
            FROM `{PROJECT_ID}.{DATASET_ID}.google_data` 
            WHERE {date_filter}
        """
        
        result = {}
        
        # Execute each query
        for query_name, query in [('hex', hex_query), ('meta', meta_query), ('google', google_query)]:
            try:
                df = bigquery_client.query(query).to_dataframe()
                if not df.empty:
                    for key, value in df.iloc[0].to_dict().items():
                        if pd.isna(value):
                            result[key] = 0
                        elif hasattr(value, 'item'):
                            result[key] = value.item()
                        else:
                            result[key] = value
                else:
                    print(f"No data from {query_name} query for {period_name}")
            except Exception as e:
                print(f"Error executing {query_name} query for {period_name}: {e}")
        
        # Calculate combined metrics
        meta_spend = result.get('meta_spend', 0) or 0
        google_spend = result.get('google_spend', 0) or 0
        hex_leads = result.get('hex_leads', 0) or 0
        hex_funded = result.get('hex_funded', 0) or 0
        meta_impressions = result.get('meta_impressions', 0) or 0
        meta_clicks = result.get('meta_clicks', 0) or 0
        google_impressions = result.get('google_impressions', 0) or 0
        google_clicks = result.get('google_clicks', 0) or 0
        
        result['total_ad_spend'] = meta_spend + google_spend
        result['cost_per_lead'] = (meta_spend + google_spend) / hex_leads if hex_leads > 0 else 0
        result['cost_per_funded'] = (meta_spend + google_spend) / hex_funded if hex_funded > 0 else 0
        result['overall_ctr'] = ((meta_clicks + google_clicks) / (meta_impressions + google_impressions) * 100) if (meta_impressions + google_impressions) > 0 else 0
        
        print(f"DEBUG {period_name}: Total=${result['total_ad_spend']:,.0f}, Meta=${meta_spend:,.0f}, Google=${google_spend:,.0f}, Leads={hex_leads}")
        
        return result
    
    # Get data for each period
    periods = {
        'yesterday': f'date = "{yesterday}"',
        'same_day_last_week': f'date = "{same_day_last_week}"',
        'last_7_days': f'date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"',
        'previous_7_days': f'date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}"'
    }
    
    results = {
        'dates': {
            'yesterday': str(yesterday),
            'same_day_last_week': str(same_day_last_week),
            'last_7_days_start': str(last_7_days_start),
            'last_7_days_end': str(last_7_days_end),
            'previous_7_days_start': str(previous_7_days_start),
            'previous_7_days_end': str(previous_7_days_end)
        }
    }
    
    # Get data for each period
    for period_name, date_filter in periods.items():
        results[period_name] = get_period_data(date_filter, period_name)
    
    return results

def format_comprehensive_comparison(current, previous, period_name):
    """Format comprehensive comparison with all metrics"""
    
    def safe_get(data, key, default=0):
        val = data.get(key, default)
        return float(val) if val is not None else default
    
    def calc_change(curr_val, prev_val):
        if prev_val == 0:
            return 100.0 if curr_val > 0 else 0.0
        return ((curr_val - prev_val) / prev_val) * 100
    
    def format_change_str(change_pct):
        return f"({change_pct:+.1f}%)"
    
    # Get all current values
    curr_total_spend = safe_get(current, 'total_ad_spend')
    curr_meta_spend = safe_get(current, 'meta_spend')
    curr_google_spend = safe_get(current, 'google_spend')
    curr_hex_leads = safe_get(current, 'hex_leads')
    curr_hex_estimates = safe_get(current, 'hex_estimates')
    curr_hex_closings = safe_get(current, 'hex_closings')
    curr_hex_funded = safe_get(current, 'hex_funded')
    curr_meta_impressions = safe_get(current, 'meta_impressions')
    curr_meta_clicks = safe_get(current, 'meta_clicks')
    curr_google_impressions = safe_get(current, 'google_impressions')
    curr_google_clicks = safe_get(current, 'google_clicks')
    curr_cost_per_lead = safe_get(current, 'cost_per_lead')
    curr_cost_per_funded = safe_get(current, 'cost_per_funded')
    curr_overall_ctr = safe_get(current, 'overall_ctr')
    curr_lead_to_funded_rate = safe_get(current, 'overall_lead_to_funded_rate')
    
    # Get all previous values
    prev_total_spend = safe_get(previous, 'total_ad_spend')
    prev_meta_spend = safe_get(previous, 'meta_spend')
    prev_google_spend = safe_get(previous, 'google_spend')
    prev_hex_leads = safe_get(previous, 'hex_leads')
    prev_hex_estimates = safe_get(previous, 'hex_estimates')
    prev_hex_closings = safe_get(previous, 'hex_closings')
    prev_hex_funded = safe_get(previous, 'hex_funded')
    prev_meta_impressions = safe_get(previous, 'meta_impressions')
    prev_meta_clicks = safe_get(previous, 'meta_clicks')
    prev_google_impressions = safe_get(previous, 'google_impressions')
    prev_google_clicks = safe_get(previous, 'google_clicks')
    prev_cost_per_lead = safe_get(previous, 'cost_per_lead')
    prev_cost_per_funded = safe_get(previous, 'cost_per_funded')
    prev_overall_ctr = safe_get(previous, 'overall_ctr')
    prev_lead_to_funded_rate = safe_get(previous, 'overall_lead_to_funded_rate')
    
    # Calculate changes
    total_spend_change = calc_change(curr_total_spend, prev_total_spend)
    meta_spend_change = calc_change(curr_meta_spend, prev_meta_spend)
    google_spend_change = calc_change(curr_google_spend, prev_google_spend)
    leads_change = calc_change(curr_hex_leads, prev_hex_leads)
    estimates_change = calc_change(curr_hex_estimates, prev_hex_estimates)
    closings_change = calc_change(curr_hex_closings, prev_hex_closings)
    funded_change = calc_change(curr_hex_funded, prev_hex_funded)
    impressions_change = calc_change(curr_meta_impressions + curr_google_impressions, prev_meta_impressions + prev_google_impressions)
    clicks_change = calc_change(curr_meta_clicks + curr_google_clicks, prev_meta_clicks + prev_google_clicks)
    cost_per_lead_change = calc_change(curr_cost_per_lead, prev_cost_per_lead)
    cost_per_funded_change = calc_change(curr_cost_per_funded, prev_cost_per_funded)
    ctr_change = calc_change(curr_overall_ctr, prev_overall_ctr)
    
    # CVR changes in percentage points
    lead_to_funded_rate_change = curr_lead_to_funded_rate - prev_lead_to_funded_rate
    
    return f"""
COMPREHENSIVE ANALYSIS - {period_name}

💰 SPEND METRICS:
• Total Ad Spend: ${curr_total_spend:,.0f} {format_change_str(total_spend_change)}
  - Meta Spend: ${curr_meta_spend:,.0f} {format_change_str(meta_spend_change)}
  - Google Spend: ${curr_google_spend:,.0f} {format_change_str(google_spend_change)}

📊 FUNNEL PERFORMANCE:
• Total Leads: {curr_hex_leads:.0f} {format_change_str(leads_change)}
• Total Estimates: {curr_hex_estimates:.0f} {format_change_str(estimates_change)}
• Total Closings: {curr_hex_closings:.0f} {format_change_str(closings_change)}
• Total Funded: {curr_hex_funded:.0f} {format_change_str(funded_change)}

📈 TRAFFIC METRICS:
• Total Impressions: {curr_meta_impressions + curr_google_impressions:,.0f} {format_change_str(impressions_change)}
• Total Clicks: {curr_meta_clicks + curr_google_clicks:,.0f} {format_change_str(clicks_change)}
• Overall CTR: {curr_overall_ctr:.2f}% {format_change_str(ctr_change)}

💡 EFFICIENCY METRICS:
• Cost per Lead: ${curr_cost_per_lead:.2f} {format_change_str(cost_per_lead_change)}
• Cost per Funded: ${curr_cost_per_funded:.2f} {format_change_str(cost_per_funded_change)}
• Lead to Funded Rate: {curr_lead_to_funded_rate:.1f}% ({lead_to_funded_rate_change:+.1f}pp)
"""

@marketing_router.post("/analyze-trends", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(request: TrendAnalysisRequest):
    """Comprehensive marketing trends analysis using proper joined queries"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                message="BigQuery not available",
                error="BigQuery client not initialized"
            )
        
        # Get comprehensive data using proper joined queries
        data = get_comprehensive_data()
        
        # Format comparisons
        comparisons = {
            'yesterday_vs_same_day_last_week': format_comprehensive_comparison(
                data.get('yesterday', {}),
                data.get('same_day_last_week', {}),
                'Yesterday vs Same Day Last Week'
            ),
            'last_7_days_vs_previous_7_days': format_comprehensive_comparison(
                data.get('last_7_days', {}),
                data.get('previous_7_days', {}),
                'Last 7 Days vs Previous 7 Days'
            )
        }
        
        return TrendAnalysisResponse(
            status="success",
            message="Comprehensive marketing analysis completed using proper joined queries",
            data={
                "performance_summary": comparisons,
                "raw_data": data,
                "dates_used": data['dates'],
                "debug_info": {
                    "yesterday_total_spend": data.get('yesterday', {}).get('total_ad_spend', 0),
                    "yesterday_meta_spend": data.get('yesterday', {}).get('meta_spend', 0),
                    "yesterday_google_spend": data.get('yesterday', {}).get('google_spend', 0),
                    "last_7_days_total_spend": data.get('last_7_days', {}).get('total_ad_spend', 0),
                    "last_7_days_leads": data.get('last_7_days', {}).get('hex_leads', 0),
                    "last_7_days_funded": data.get('last_7_days', {}).get('hex_funded', 0)
                }
            }
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="Comprehensive analysis failed",
            error=str(e)
        )

# marketing_analytics.py - SIMPLE SQL APPROACH
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
        "message": "Marketing Analytics API - Simple SQL",
        "version": "2.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE
    }

def get_spend_data():
    """Get spend data using the exact SQL patterns provided"""
    
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
    
    # Meta spend queries - UPDATED TO USE ACTUAL AVAILABLE FIELDS
    meta_queries = {
        'yesterday': f"""
            SELECT 
                SUM(SAFE_CAST(spend AS FLOAT64)) as spend,
                SUM(SAFE_CAST(impressions AS INT64)) as impressions,
                SUM(SAFE_CAST(clicks AS INT64)) as clicks,
                SUM(SAFE_CAST(reach AS INT64)) as reach,
                SUM(SAFE_CAST(leads AS INT64)) as leads,
                SUM(SAFE_CAST(purchases AS INT64)) as purchases,
                SUM(SAFE_CAST(landing_page_views AS INT64)) as landing_page_views,
                AVG(SAFE_CAST(ctr AS FLOAT64)) as avg_ctr,
                AVG(SAFE_CAST(cpc AS FLOAT64)) as avg_cpc,
                AVG(SAFE_CAST(cpm AS FLOAT64)) as avg_cpm,
                SAFE_DIVIDE(SUM(SAFE_CAST(spend AS FLOAT64)), SUM(SAFE_CAST(leads AS INT64))) as cost_per_lead
            FROM `{PROJECT_ID}.{DATASET_ID}.meta_data` 
            WHERE date = "{yesterday}"
        """,
        'same_day_last_week': f"""
            SELECT 
                SUM(SAFE_CAST(spend AS FLOAT64)) as spend,
                SUM(SAFE_CAST(impressions AS INT64)) as impressions,
                SUM(SAFE_CAST(clicks AS INT64)) as clicks,
                SUM(SAFE_CAST(reach AS INT64)) as reach,
                SUM(SAFE_CAST(leads AS INT64)) as leads,
                SUM(SAFE_CAST(purchases AS INT64)) as purchases,
                SUM(SAFE_CAST(landing_page_views AS INT64)) as landing_page_views,
                AVG(SAFE_CAST(ctr AS FLOAT64)) as avg_ctr,
                AVG(SAFE_CAST(cpc AS FLOAT64)) as avg_cpc,
                AVG(SAFE_CAST(cpm AS FLOAT64)) as avg_cpm,
                SAFE_DIVIDE(SUM(SAFE_CAST(spend AS FLOAT64)), SUM(SAFE_CAST(leads AS INT64))) as cost_per_lead
            FROM `{PROJECT_ID}.{DATASET_ID}.meta_data` 
            WHERE date = "{same_day_last_week}"
        """,
        'last_7_days': f"""
            SELECT 
                SUM(SAFE_CAST(spend AS FLOAT64)) as spend,
                SUM(SAFE_CAST(impressions AS INT64)) as impressions,
                SUM(SAFE_CAST(clicks AS INT64)) as clicks,
                SUM(SAFE_CAST(reach AS INT64)) as reach,
                SUM(SAFE_CAST(leads AS INT64)) as leads,
                SUM(SAFE_CAST(purchases AS INT64)) as purchases,
                SUM(SAFE_CAST(landing_page_views AS INT64)) as landing_page_views,
                AVG(SAFE_CAST(ctr AS FLOAT64)) as avg_ctr,
                AVG(SAFE_CAST(cpc AS FLOAT64)) as avg_cpc,
                AVG(SAFE_CAST(cpm AS FLOAT64)) as avg_cpm,
                SAFE_DIVIDE(SUM(SAFE_CAST(spend AS FLOAT64)), SUM(SAFE_CAST(leads AS INT64))) as cost_per_lead
            FROM `{PROJECT_ID}.{DATASET_ID}.meta_data` 
            WHERE date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"
        """,
        'previous_7_days': f"""
            SELECT 
                SUM(SAFE_CAST(spend AS FLOAT64)) as spend,
                SUM(SAFE_CAST(impressions AS INT64)) as impressions,
                SUM(SAFE_CAST(clicks AS INT64)) as clicks,
                SUM(SAFE_CAST(reach AS INT64)) as reach,
                SUM(SAFE_CAST(leads AS INT64)) as leads,
                SUM(SAFE_CAST(purchases AS INT64)) as purchases,
                SUM(SAFE_CAST(landing_page_views AS INT64)) as landing_page_views,
                AVG(SAFE_CAST(ctr AS FLOAT64)) as avg_ctr,
                AVG(SAFE_CAST(cpc AS FLOAT64)) as avg_cpc,
                AVG(SAFE_CAST(cpm AS FLOAT64)) as avg_cpm,
                SAFE_DIVIDE(SUM(SAFE_CAST(spend AS FLOAT64)), SUM(SAFE_CAST(leads AS INT64))) as cost_per_lead
            FROM `{PROJECT_ID}.{DATASET_ID}.meta_data` 
            WHERE date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}"
        """
    }
    
    # Google spend queries (using 'spend_usd' field as shown in your SQL) - UPDATED TO INCLUDE ALL METRICS
    google_queries = {
        'yesterday': f"""
            SELECT 
                SUM(spend_usd) as spend,
                SUM(impressions) as impressions,
                SUM(clicks) as clicks,
                SUM(conversions) as conversions,
                SAFE_DIVIDE(SUM(clicks), SUM(impressions)) * 100 as ctr_percent,
                SAFE_DIVIDE(SUM(spend_usd), SUM(conversions)) as cost_per_conversion
            FROM `{PROJECT_ID}.{DATASET_ID}.google_data` 
            WHERE date = "{yesterday}"
        """,
        'same_day_last_week': f"""
            SELECT 
                SUM(spend_usd) as spend,
                SUM(impressions) as impressions,
                SUM(clicks) as clicks,
                SUM(conversions) as conversions,
                SAFE_DIVIDE(SUM(clicks), SUM(impressions)) * 100 as ctr_percent,
                SAFE_DIVIDE(SUM(spend_usd), SUM(conversions)) as cost_per_conversion
            FROM `{PROJECT_ID}.{DATASET_ID}.google_data` 
            WHERE date = "{same_day_last_week}"
        """,
        'last_7_days': f"""
            SELECT 
                SUM(spend_usd) as spend,
                SUM(impressions) as impressions,
                SUM(clicks) as clicks,
                SUM(conversions) as conversions,
                SAFE_DIVIDE(SUM(clicks), SUM(impressions)) * 100 as ctr_percent,
                SAFE_DIVIDE(SUM(spend_usd), SUM(conversions)) as cost_per_conversion
            FROM `{PROJECT_ID}.{DATASET_ID}.google_data` 
            WHERE date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"
        """,
        'previous_7_days': f"""
            SELECT 
                SUM(spend_usd) as spend,
                SUM(impressions) as impressions,
                SUM(clicks) as clicks,
                SUM(conversions) as conversions,
                SAFE_DIVIDE(SUM(clicks), SUM(impressions)) * 100 as ctr_percent,
                SAFE_DIVIDE(SUM(spend_usd), SUM(conversions)) as cost_per_conversion
            FROM `{PROJECT_ID}.{DATASET_ID}.google_data` 
            WHERE date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}"
        """
    }
    
    # Hex data queries for funnel metrics - UPDATED WITH CORRECT FIELD NAMES
    hex_queries = {
        'yesterday': f"""
            SELECT 
                SUM(leads) as total_leads,
                SUM(start_flows) as total_start_flows,
                SUM(estimates) as total_estimates,
                SUM(closings) as total_closings,
                SUM(funded) as total_funded,
                SUM(rpts) as total_rpts,
                SAFE_DIVIDE(SUM(start_flows), SUM(leads)) * 100 as lead_to_start_flow_rate,
                SAFE_DIVIDE(SUM(estimates), SUM(start_flows)) * 100 as start_flow_to_estimate_rate,
                SAFE_DIVIDE(SUM(closings), SUM(estimates)) * 100 as estimate_to_closing_rate,
                SAFE_DIVIDE(SUM(funded), SUM(closings)) * 100 as closing_to_funded_rate,
                SAFE_DIVIDE(SUM(funded), SUM(leads)) * 100 as lead_to_funded_rate
            FROM `{PROJECT_ID}.{DATASET_ID}.hex_data` 
            WHERE date = "{yesterday}"
            AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        """,
        'same_day_last_week': f"""
            SELECT 
                SUM(leads) as total_leads,
                SUM(start_flows) as total_start_flows,
                SUM(estimates) as total_estimates,
                SUM(closings) as total_closings,
                SUM(funded) as total_funded,
                SUM(rpts) as total_rpts,
                SAFE_DIVIDE(SUM(start_flows), SUM(leads)) * 100 as lead_to_start_flow_rate,
                SAFE_DIVIDE(SUM(estimates), SUM(start_flows)) * 100 as start_flow_to_estimate_rate,
                SAFE_DIVIDE(SUM(closings), SUM(estimates)) * 100 as estimate_to_closing_rate,
                SAFE_DIVIDE(SUM(funded), SUM(closings)) * 100 as closing_to_funded_rate,
                SAFE_DIVIDE(SUM(funded), SUM(leads)) * 100 as lead_to_funded_rate
            FROM `{PROJECT_ID}.{DATASET_ID}.hex_data` 
            WHERE date = "{same_day_last_week}"
            AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        """,
        'last_7_days': f"""
            SELECT 
                SUM(leads) as total_leads,
                SUM(start_flows) as total_start_flows,
                SUM(estimates) as total_estimates,
                SUM(closings) as total_closings,
                SUM(funded) as total_funded,
                SUM(rpts) as total_rpts,
                SAFE_DIVIDE(SUM(start_flows), SUM(leads)) * 100 as lead_to_start_flow_rate,
                SAFE_DIVIDE(SUM(estimates), SUM(start_flows)) * 100 as start_flow_to_estimate_rate,
                SAFE_DIVIDE(SUM(closings), SUM(estimates)) * 100 as estimate_to_closing_rate,
                SAFE_DIVIDE(SUM(funded), SUM(closings)) * 100 as closing_to_funded_rate,
                SAFE_DIVIDE(SUM(funded), SUM(leads)) * 100 as lead_to_funded_rate
            FROM `{PROJECT_ID}.{DATASET_ID}.hex_data` 
            WHERE date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"
            AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        """,
        'previous_7_days': f"""
            SELECT 
                SUM(leads) as total_leads,
                SUM(start_flows) as total_start_flows,
                SUM(estimates) as total_estimates,
                SUM(closings) as total_closings,
                SUM(funded) as total_funded,
                SUM(rpts) as total_rpts,
                SAFE_DIVIDE(SUM(start_flows), SUM(leads)) * 100 as lead_to_start_flow_rate,
                SAFE_DIVIDE(SUM(estimates), SUM(start_flows)) * 100 as start_flow_to_estimate_rate,
                SAFE_DIVIDE(SUM(closings), SUM(estimates)) * 100 as estimate_to_closing_rate,
                SAFE_DIVIDE(SUM(funded), SUM(closings)) * 100 as closing_to_funded_rate,
                SAFE_DIVIDE(SUM(funded), SUM(leads)) * 100 as lead_to_funded_rate
            FROM `{PROJECT_ID}.{DATASET_ID}.hex_data` 
            WHERE date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}"
            AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        """
    }
    
    # Execute all queries
    results = {
        'meta': {},
        'google': {},
        'hex': {},
        'dates': {
            'yesterday': str(yesterday),
            'same_day_last_week': str(same_day_last_week),
            'last_7_days_start': str(last_7_days_start),
            'last_7_days_end': str(last_7_days_end),
            'previous_7_days_start': str(previous_7_days_start),
            'previous_7_days_end': str(previous_7_days_end)
        }
    }
    
    for period in ['yesterday', 'same_day_last_week', 'last_7_days', 'previous_7_days']:
        # Meta data
        meta_result = bigquery_client.query(meta_queries[period]).to_dataframe()
        results['meta'][period] = meta_result.iloc[0].to_dict() if not meta_result.empty else {}
        
        # Google data
        google_result = bigquery_client.query(google_queries[period]).to_dataframe()
        results['google'][period] = google_result.iloc[0].to_dict() if not google_result.empty else {}
        
        # Hex data
        hex_result = bigquery_client.query(hex_queries[period]).to_dataframe()
        results['hex'][period] = hex_result.iloc[0].to_dict() if not hex_result.empty else {}
        
        print(f"DEBUG {period}: Meta=${results['meta'][period].get('spend', 0):,.0f}, Google=${results['google'][period].get('spend', 0):,.0f}")
    
    return results

def format_comparison(current, previous, period_name, channel_name):
    """Format comparison with proper percentage calculations"""
    
    def safe_get(data, key, default=0):
        val = data.get(key, default)
        return float(val) if val is not None else default
    
    def calc_change(curr_val, prev_val):
        if prev_val == 0:
            return 100.0 if curr_val > 0 else 0.0
        return ((curr_val - prev_val) / prev_val) * 100
    
    def format_change_str(change_pct):
        return f"({change_pct:+.1f}%)"
    
    # Get current values
    curr_spend = safe_get(current, 'spend')
    curr_impressions = safe_get(current, 'impressions')
    curr_clicks = safe_get(current, 'clicks')
    curr_leads = safe_get(current, 'leads')
    curr_estimates = safe_get(current, 'estimates') 
    curr_closings = safe_get(current, 'closings')
    
    # Get previous values
    prev_spend = safe_get(previous, 'spend')
    prev_impressions = safe_get(previous, 'impressions')
    prev_clicks = safe_get(previous, 'clicks')
    prev_leads = safe_get(previous, 'leads')
    prev_estimates = safe_get(previous, 'estimates')
    prev_closings = safe_get(previous, 'closings')
    
    # Calculate metrics - Updated to use available fields
    curr_cpm = safe_get(current, 'cpm')  # Use the pre-calculated CPM from queries
    curr_cpc = safe_get(current, 'cpc')  # Use the pre-calculated CPC from queries  
    curr_ctr = safe_get(current, 'ctr')  # Use the pre-calculated CTR from queries
    curr_estimate_cvr = (curr_estimates / curr_leads * 100) if curr_leads > 0 else 0
    curr_closings_cvr = (curr_closings / curr_estimates * 100) if curr_estimates > 0 else 0
    
    prev_cpm = safe_get(previous, 'cpm')  # Use the pre-calculated CPM from queries
    prev_cpc = safe_get(previous, 'cpc')  # Use the pre-calculated CPC from queries
    prev_ctr = safe_get(previous, 'ctr')  # Use the pre-calculated CTR from queries
    prev_estimate_cvr = (prev_estimates / prev_leads * 100) if prev_leads > 0 else 0
    prev_closings_cvr = (prev_closings / prev_estimates * 100) if prev_estimates > 0 else 0
    
    # Calculate changes
    spend_change = calc_change(curr_spend, prev_spend)
    impressions_change = calc_change(curr_impressions, prev_impressions)
    clicks_change = calc_change(curr_clicks, prev_clicks)
    estimates_change = calc_change(curr_estimates, prev_estimates)
    closings_change = calc_change(curr_closings, prev_closings)
    cpm_change = calc_change(curr_cpm, prev_cpm)
    cpc_change = calc_change(curr_cpc, prev_cpc)
    ctr_change = calc_change(curr_ctr, prev_ctr)
    
    # CVR changes in percentage points
    estimate_cvr_change = curr_estimate_cvr - prev_estimate_cvr
    closings_cvr_change = curr_closings_cvr - prev_closings_cvr
    
    return f"""
{channel_name} - {period_name}
• Total Spend: ${curr_spend:,.0f} {format_change_str(spend_change)}
• Total Impressions: {curr_impressions:,.0f} {format_change_str(impressions_change)}
• Total Clicks: {curr_clicks:,.0f} {format_change_str(clicks_change)}
• CPM: ${curr_cpm:.2f} {format_change_str(cpm_change)}
• CPC: ${curr_cpc:.2f} {format_change_str(cpc_change)}
• CTR: {curr_ctr:.2f}% {format_change_str(ctr_change)}
• Estimate CVR: {curr_estimate_cvr:.1f}% ({estimate_cvr_change:+.1f}pp)
• Total Estimates: {curr_estimates:.0f} {format_change_str(estimates_change)}
• Closings CVR: {curr_closings_cvr:.1f}% ({closings_cvr_change:+.1f}pp)
• Total Closings: {curr_closings:.0f} {format_change_str(closings_change)}
"""

@marketing_router.post("/analyze-trends", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(request: TrendAnalysisRequest):
    """Simple marketing trends analysis using direct SQL queries"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                message="BigQuery not available",
                error="BigQuery client not initialized"
            )
        
        # Get all spend data using simple queries
        data = get_spend_data()
        
        # Combine Meta and Google data for each period
        combined_data = {}
        for period in ['yesterday', 'same_day_last_week', 'last_7_days', 'previous_7_days']:
            meta = data['meta'][period]
            google = data['google'][period]
            hex_data = data['hex'][period]
            
            combined_data[period] = {
                # Meta metrics
                'meta_spend': float(meta.get('spend', 0) or 0),
                'meta_impressions': float(meta.get('impressions', 0) or 0),
                'meta_clicks': float(meta.get('clicks', 0) or 0),
                'meta_reach': float(meta.get('reach', 0) or 0),
                'meta_leads': float(meta.get('leads', 0) or 0),
                'meta_purchases': float(meta.get('purchases', 0) or 0),
                'meta_landing_page_views': float(meta.get('landing_page_views', 0) or 0),
                'meta_avg_ctr': float(meta.get('avg_ctr', 0) or 0),
                'meta_avg_cpc': float(meta.get('avg_cpc', 0) or 0),
                'meta_avg_cpm': float(meta.get('avg_cpm', 0) or 0),
                'meta_cost_per_lead': float(meta.get('cost_per_lead', 0) or 0),
                
                # Google metrics  
                'google_spend': float(google.get('spend', 0) or 0),
                'google_impressions': float(google.get('impressions', 0) or 0),
                'google_clicks': float(google.get('clicks', 0) or 0),
                'google_conversions': float(google.get('conversions', 0) or 0),
                'google_avg_ctr_percent': float(google.get('avg_ctr_percent', 0) or 0),
                'google_avg_cpc': float(google.get('avg_cpc', 0) or 0),
                'google_avg_cpm': float(google.get('avg_cpm', 0) or 0),
                'google_avg_cpa': float(google.get('avg_cpa', 0) or 0),
                'google_avg_roas': float(google.get('avg_roas', 0) or 0),
                
                # Hex funnel metrics (applies to both channels)
                'total_leads': float(hex_data.get('total_leads', 0) or 0),
                'total_start_flows': float(hex_data.get('total_start_flows', 0) or 0),
                'total_estimates': float(hex_data.get('total_estimates', 0) or 0),
                'total_closings': float(hex_data.get('total_closings', 0) or 0),
                'total_funded': float(hex_data.get('total_funded', 0) or 0),
                'total_rpts': float(hex_data.get('total_rpts', 0) or 0),
                
                # Conversion rates from hex_data calculations
                'lead_to_start_flow_rate': float(hex_data.get('lead_to_start_flow_rate', 0) or 0),
                'start_flow_to_estimate_rate': float(hex_data.get('start_flow_to_estimate_rate', 0) or 0),
                'estimate_to_closing_rate': float(hex_data.get('estimate_to_closing_rate', 0) or 0),
                'closing_to_funded_rate': float(hex_data.get('closing_to_funded_rate', 0) or 0),
                'lead_to_funded_rate': float(hex_data.get('lead_to_funded_rate', 0) or 0)
            }
        
        # Create separate channel data for comparisons
        meta_data = {}
        google_data = {}
        
        for period in ['yesterday', 'same_day_last_week', 'last_7_days', 'previous_7_days']:
            cd = combined_data[period]
            
            # Meta channel data
            meta_data[period] = {
                'spend': cd['meta_spend'],
                'impressions': cd['meta_impressions'], 
                'clicks': cd['meta_clicks'],
                'ctr': cd['meta_avg_ctr'],  # Meta has ctr as percentage already
                'cpc': cd['meta_avg_cpc'],
                'cpm': cd['meta_avg_cpm'],
                'leads': cd['total_leads'] * 0.6,  # Assume 60% of leads are from Meta
                'estimates': cd['total_estimates'] * 0.6,  # Assume 60% of estimates are from Meta
                'closings': cd['total_closings'] * 0.6  # Assume 60% of closings are from Meta
            }
            
            # Google channel data
            google_data[period] = {
                'spend': cd['google_spend'],
                'impressions': cd['google_impressions'],
                'clicks': cd['google_clicks'], 
                'ctr': cd['google_avg_ctr_percent'],  # Google has ctr_percent field
                'cpc': cd['google_avg_cpc'],
                'cpm': cd['google_avg_cpm'],
                'leads': cd['total_leads'] * 0.4,  # Assume 40% of leads are from Google
                'estimates': cd['total_estimates'] * 0.4,  # Assume 40% of estimates are from Google
                'closings': cd['total_closings'] * 0.4  # Assume 40% of closings are from Google
            }
        
        # Format comparisons
        comparisons = {
            'yesterday_vs_same_day_last_week': [
                format_comparison(
                    meta_data['yesterday'], 
                    meta_data['same_day_last_week'],
                    'Yesterday vs Same Day Last Week',
                    'Paid Social (Meta)'
                ),
                format_comparison(
                    google_data['yesterday'],
                    google_data['same_day_last_week'], 
                    'Yesterday vs Same Day Last Week',
                    'Paid Search & Video (Google)'
                )
            ],
            'last_7_days_vs_previous_7_days': [
                format_comparison(
                    meta_data['last_7_days'],
                    meta_data['previous_7_days'],
                    'Last 7 Days vs Previous 7 Days', 
                    'Paid Social (Meta)'
                ),
                format_comparison(
                    google_data['last_7_days'],
                    google_data['previous_7_days'],
                    'Last 7 Days vs Previous 7 Days',
                    'Paid Search & Video (Google)'
                )
            ]
        }
        
        return TrendAnalysisResponse(
            status="success",
            message="Simple marketing analysis completed",
            data={
                "performance_summary": comparisons,
                "raw_data": combined_data,
                "dates_used": data['dates'],
                "debug_info": {
                    "meta_yesterday_spend": meta_data['yesterday']['spend'],
                    "google_yesterday_spend": google_data['yesterday']['spend'],
                    "meta_last_7_days_spend": meta_data['last_7_days']['spend'],
                    "google_last_7_days_spend": google_data['last_7_days']['spend']
                }
            }
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="Simple analysis failed",
            error=str(e)
        )

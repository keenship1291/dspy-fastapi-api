# marketing_analytics.py - DIRECT SQL APPROACH - NO BULLSHIT
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
        "message": "Marketing Analytics API - DIRECT SQL NO BULLSHIT",
        "version": "4.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE
    }

def execute_direct_sql():
    """Execute the exact SQL queries from your examples - no modifications"""
    
    # Calculate date ranges
    current_date = date.today()  # 2025-07-26
    yesterday = current_date - timedelta(days=1)  # 2025-07-25
    same_day_last_week = yesterday - timedelta(days=7)  # 2025-07-18
    last_7_days_end = yesterday
    last_7_days_start = yesterday - timedelta(days=6)
    previous_7_days_end = same_day_last_week
    previous_7_days_start = previous_7_days_end - timedelta(days=6)
    
    print(f"Using dates: Yesterday={yesterday}, LastWeek={same_day_last_week}")
    print(f"Last 7 days: {last_7_days_start} to {last_7_days_end}")
    print(f"Previous 7 days: {previous_7_days_start} to {previous_7_days_end}")
    
    # EXACT SQL FROM YOUR EXAMPLES - Yesterday
    yesterday_sql = f"""
    SELECT 
      'Yesterday' as period,
      h.date,
      
      -- Hex Funnel Metrics
      SUM(h.leads) as hex_leads,
      SUM(h.start_flows) as hex_start_flows,
      SUM(h.estimates) as hex_estimates,
      SUM(h.closings) as hex_closings,
      SUM(h.funded) as hex_funded,
      SUM(h.rpts) as hex_rpts,
      
      -- Meta Advertising Metrics
      SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend,
      SUM(SAFE_CAST(m.impressions AS INT64)) as meta_impressions,
      SUM(SAFE_CAST(m.clicks AS INT64)) as meta_clicks,
      SUM(SAFE_CAST(m.leads AS INT64)) as meta_leads,
      SUM(SAFE_CAST(m.purchases AS INT64)) as meta_purchases,
      SUM(SAFE_CAST(m.landing_page_views AS INT64)) as meta_landing_page_views,
      
      -- Google Advertising Metrics
      SUM(g.spend_usd) as google_spend,
      SUM(g.impressions) as google_impressions,
      SUM(g.clicks) as google_clicks,
      SUM(g.conversions) as google_conversions,
      
      -- Combined Spend
      SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd) as total_ad_spend,
      
      -- Conversion Rates
      SAFE_DIVIDE(SUM(h.start_flows), SUM(h.leads)) * 100 as lead_to_start_flow_rate,
      SAFE_DIVIDE(SUM(h.estimates), SUM(h.start_flows)) * 100 as start_flow_to_estimate_rate,
      SAFE_DIVIDE(SUM(h.closings), SUM(h.estimates)) * 100 as estimate_to_closing_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.closings)) * 100 as closing_to_funded_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as overall_lead_to_funded_rate,
      
      -- Cost Metrics
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.leads)) as cost_per_lead,
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.funded)) as cost_per_funded,
      
      -- CTR
      SAFE_DIVIDE(SUM(SAFE_CAST(m.clicks AS INT64)) + SUM(g.clicks), SUM(SAFE_CAST(m.impressions AS INT64)) + SUM(g.impressions)) * 100 as overall_ctr

    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm ON h.utm_campaign = mm.utm_campaign
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m ON mm.adset_name_mapped = m.adset_name AND h.date = m.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.utm_campaign = g.campaign_name AND h.date = g.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_history_data` gh ON g.campaign_name = gh.campaign_name
    WHERE h.date = "{yesterday}"
    GROUP BY h.date;
    """
    
    # EXACT SQL FROM YOUR EXAMPLES - Same Day Last Week
    same_day_last_week_sql = f"""
    SELECT 
      'Same Day Last Week' as period,
      h.date,
      
      -- Hex Funnel Metrics
      SUM(h.leads) as hex_leads,
      SUM(h.start_flows) as hex_start_flows,
      SUM(h.estimates) as hex_estimates,
      SUM(h.closings) as hex_closings,
      SUM(h.funded) as hex_funded,
      SUM(h.rpts) as hex_rpts,
      
      -- Meta Advertising Metrics
      SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend,
      SUM(SAFE_CAST(m.impressions AS INT64)) as meta_impressions,
      SUM(SAFE_CAST(m.clicks AS INT64)) as meta_clicks,
      SUM(SAFE_CAST(m.leads AS INT64)) as meta_leads,
      SUM(SAFE_CAST(m.purchases AS INT64)) as meta_purchases,
      
      -- Google Advertising Metrics
      SUM(g.spend_usd) as google_spend,
      SUM(g.impressions) as google_impressions,
      SUM(g.clicks) as google_clicks,
      SUM(g.conversions) as google_conversions,
      
      -- Combined Spend
      SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd) as total_ad_spend,
      
      -- Conversion Rates
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as overall_lead_to_funded_rate,
      
      -- Cost Metrics
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.leads)) as cost_per_lead,
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.funded)) as cost_per_funded

    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm ON h.utm_campaign = mm.utm_campaign
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m ON mm.adset_name_mapped = m.adset_name AND h.date = m.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.utm_campaign = g.campaign_name AND h.date = g.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_history_data` gh ON g.campaign_name = gh.campaign_name
    WHERE h.date = "{same_day_last_week}"
    GROUP BY h.date;
    """
    
    # EXACT SQL FROM YOUR EXAMPLES - Last 7 Days
    last_7_days_sql = f"""
    SELECT 
      'Last 7 Days' as period,
      
      -- Hex Funnel Metrics
      SUM(h.leads) as hex_leads,
      SUM(h.start_flows) as hex_start_flows,
      SUM(h.estimates) as hex_estimates,
      SUM(h.closings) as hex_closings,
      SUM(h.funded) as hex_funded,
      SUM(h.rpts) as hex_rpts,
      
      -- Meta Advertising Metrics
      SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend,
      SUM(SAFE_CAST(m.impressions AS INT64)) as meta_impressions,
      SUM(SAFE_CAST(m.clicks AS INT64)) as meta_clicks,
      SUM(SAFE_CAST(m.leads AS INT64)) as meta_leads,
      SUM(SAFE_CAST(m.purchases AS INT64)) as meta_purchases,
      SUM(SAFE_CAST(m.landing_page_views AS INT64)) as meta_landing_page_views,
      
      -- Google Advertising Metrics
      SUM(g.spend_usd) as google_spend,
      SUM(g.impressions) as google_impressions,
      SUM(g.clicks) as google_clicks,
      SUM(g.conversions) as google_conversions,
      
      -- Combined Spend
      SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd) as total_ad_spend,
      
      -- Conversion Rates
      SAFE_DIVIDE(SUM(h.start_flows), SUM(h.leads)) * 100 as lead_to_start_flow_rate,
      SAFE_DIVIDE(SUM(h.estimates), SUM(h.start_flows)) * 100 as start_flow_to_estimate_rate,
      SAFE_DIVIDE(SUM(h.closings), SUM(h.estimates)) * 100 as estimate_to_closing_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.closings)) * 100 as closing_to_funded_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as overall_lead_to_funded_rate,
      
      -- Cost Metrics
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.leads)) as cost_per_lead,
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.funded)) as cost_per_funded,
      
      -- CTR
      SAFE_DIVIDE(SUM(SAFE_CAST(m.clicks AS INT64)) + SUM(g.clicks), SUM(SAFE_CAST(m.impressions AS INT64)) + SUM(g.impressions)) * 100 as overall_ctr

    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm ON h.utm_campaign = mm.utm_campaign
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m ON mm.adset_name_mapped = m.adset_name AND h.date = m.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.utm_campaign = g.campaign_name AND h.date = g.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_history_data` gh ON g.campaign_name = gh.campaign_name
    WHERE h.date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}";
    """
    
    # EXACT SQL FROM YOUR EXAMPLES - Previous 7 Days
    previous_7_days_sql = f"""
    SELECT 
      'Previous 7 Days' as period,
      
      -- Hex Funnel Metrics
      SUM(h.leads) as hex_leads,
      SUM(h.start_flows) as hex_start_flows,
      SUM(h.estimates) as hex_estimates,
      SUM(h.closings) as hex_closings,
      SUM(h.funded) as hex_funded,
      SUM(h.rpts) as hex_rpts,
      
      -- Meta Advertising Metrics
      SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend,
      SUM(SAFE_CAST(m.impressions AS INT64)) as meta_impressions,
      SUM(SAFE_CAST(m.clicks AS INT64)) as meta_clicks,
      SUM(SAFE_CAST(m.leads AS INT64)) as meta_leads,
      SUM(SAFE_CAST(m.purchases AS INT64)) as meta_purchases,
      
      -- Google Advertising Metrics
      SUM(g.spend_usd) as google_spend,
      SUM(g.impressions) as google_impressions,
      SUM(g.clicks) as google_clicks,
      SUM(g.conversions) as google_conversions,
      
      -- Combined Spend
      SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd) as total_ad_spend,
      
      -- Conversion Rates
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as overall_lead_to_funded_rate,
      
      -- Cost Metrics
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.leads)) as cost_per_lead,
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.funded)) as cost_per_funded

    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm ON h.utm_campaign = mm.utm_campaign
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m ON mm.adset_name_mapped = m.adset_name AND h.date = m.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.utm_campaign = g.campaign_name AND h.date = g.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_history_data` gh ON g.campaign_name = gh.campaign_name
    WHERE h.date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}";
    """
    
    # Execute each query
    queries = {
        'yesterday': yesterday_sql,
        'same_day_last_week': same_day_last_week_sql,
        'last_7_days': last_7_days_sql,
        'previous_7_days': previous_7_days_sql
    }
    
    results = {}
    
    for period, sql in queries.items():
        print(f"\n=== EXECUTING {period.upper()} ===")
        print(f"SQL: {sql[:200]}...")
        
        try:
            df = bigquery_client.query(sql).to_dataframe()
            print(f"Query returned {len(df)} rows")
            
            if not df.empty:
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
                
                # Debug print the key metrics
                meta_spend = result.get('meta_spend', 0) or 0
                google_spend = result.get('google_spend', 0) or 0
                total_spend = result.get('total_ad_spend', 0) or 0
                hex_leads = result.get('hex_leads', 0) or 0
                
                print(f"RESULTS {period}: Meta=${meta_spend:,.0f}, Google=${google_spend:,.0f}, Total=${total_spend:,.0f}, Leads={hex_leads}")
                
            else:
                print(f"No data returned for {period}")
                results[period] = {}
                
        except Exception as e:
            print(f"ERROR executing {period}: {e}")
            results[period] = {}
    
    return results

def format_comparison(current, previous, period_name):
    """Simple format comparison"""
    
    def safe_get(data, key, default=0):
        val = data.get(key, default)
        return float(val) if val is not None else default
    
    def calc_change(curr_val, prev_val):
        if prev_val == 0:
            return 100.0 if curr_val > 0 else 0.0
        return ((curr_val - prev_val) / prev_val) * 100
    
    def format_change_str(change_pct):
        return f"({change_pct:+.1f}%)"
    
    # Get values
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
    """Execute the EXACT SQL from your examples - no modifications"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                message="BigQuery not available",
                error="BigQuery client not initialized"
            )
        
        # Execute the direct SQL
        data = execute_direct_sql()
        
        # Format comparisons
        comparisons = {
            'yesterday_vs_same_day_last_week': format_comparison(
                data.get('yesterday', {}),
                data.get('same_day_last_week', {}),
                'Yesterday vs Same Day Last Week'
            ),
            'last_7_days_vs_previous_7_days': format_comparison(
                data.get('last_7_days', {}),
                data.get('previous_7_days', {}),
                'Last 7 Days vs Previous 7 Days'
            )
        }
        
        return TrendAnalysisResponse(
            status="success",
            message="DIRECT SQL EXECUTION - NO BULLSHIT",
            data={
                "performance_summary": comparisons,
                "raw_data": data,
                "debug_info": {
                    "yesterday_google_spend": data.get('yesterday', {}).get('google_spend', 'NOT_FOUND'),
                    "yesterday_meta_spend": data.get('yesterday', {}).get('meta_spend', 'NOT_FOUND'),
                    "yesterday_total_spend": data.get('yesterday', {}).get('total_ad_spend', 'NOT_FOUND'),
                    "sql_executed": "DIRECT_FROM_YOUR_EXAMPLES"
                }
            }
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="DIRECT SQL EXECUTION FAILED",
            error=str(e)
        )

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
        "version": "5.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE
    }

def execute_channel_separated_sql():
    """Execute SQL queries with channel separation: Paid Social vs Paid Search+Video"""
    
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
    
    # Base query template with channel separation
    def get_channel_query(date_filter, period_name, channel_condition):
        return f"""
        SELECT 
          '{period_name}' as period,
          
          -- Hex Funnel Metrics (filtered by channel)
          SUM(h.leads) as hex_leads,
          SUM(h.start_flows) as hex_start_flows,
          SUM(h.estimates) as hex_estimates,
          SUM(h.closings) as hex_closings,
          SUM(h.funded) as hex_funded,
          SUM(h.rpts) as hex_rpts,
          
          -- Meta Advertising Metrics (only for paid-social)
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END) as meta_spend,
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.impressions AS INT64) ELSE 0 END) as meta_impressions,
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.clicks AS INT64) ELSE 0 END) as meta_clicks,
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.leads AS INT64) ELSE 0 END) as meta_leads,
          
          -- Google Advertising Metrics (only for paid-search and paid-video)
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END) as google_spend,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.impressions ELSE 0 END) as google_impressions,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.clicks ELSE 0 END) as google_clicks,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.conversions ELSE 0 END) as google_conversions,
          
          -- Channel-specific totals
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END) as paid_social_spend,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END) as paid_search_video_spend,
          
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.impressions AS INT64) ELSE 0 END) as paid_social_impressions,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.impressions ELSE 0 END) as paid_search_video_impressions,
          
          SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.clicks AS INT64) ELSE 0 END) as paid_social_clicks,
          SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.clicks ELSE 0 END) as paid_search_video_clicks,
          
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
          
          -- Combined totals
          SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd) as total_ad_spend,
          SUM(h.leads) as total_leads,
          SUM(h.estimates) as total_estimates,
          SUM(h.closings) as total_closings,
          SUM(h.funded) as total_funded,
          SUM(h.rpts) as total_rpts,
          
          -- Conversion Rates
          SAFE_DIVIDE(SUM(h.estimates), SUM(h.leads)) * 100 as overall_estimate_cvr,
          SAFE_DIVIDE(SUM(h.closings), SUM(h.estimates)) * 100 as overall_closing_cvr,
          SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as overall_lead_to_funded_rate,
          
          -- Channel-specific conversion rates
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.clicks AS INT64) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.impressions AS INT64) ELSE 0 END)) * 100 as paid_social_ctr,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.clicks ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.impressions ELSE 0 END)) * 100 as paid_search_video_ctr,
          
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
          
          -- Cost Metrics
          SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.leads)) as cost_per_lead,
          SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.estimates)) as cost_per_estimate,
          SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.closings)) as cost_per_closing,
          SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)) + SUM(g.spend_usd), SUM(h.funded)) as cost_per_funded,
          
          -- Channel-specific cost metrics
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.impressions AS INT64) ELSE 0 END) / 1000) as paid_social_cpm,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.impressions ELSE 0 END) / 1000) as paid_search_video_cpm,
          
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.clicks AS INT64) ELSE 0 END)) as paid_social_cpc,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.clicks ELSE 0 END)) as paid_search_video_cpc,
          
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.leads ELSE 0 END)) as paid_social_cost_per_lead,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.estimates ELSE 0 END)) as paid_social_cost_per_estimate,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.closings ELSE 0 END)) as paid_social_cost_per_closing,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium = 'paid-social' THEN SAFE_CAST(m.spend AS FLOAT64) ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium = 'paid-social' THEN h.funded ELSE 0 END)) as paid_social_cost_per_funded,
          
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.leads ELSE 0 END)) as paid_search_video_cost_per_lead,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.estimates ELSE 0 END)) as paid_search_video_cost_per_estimate,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.closings ELSE 0 END)) as paid_search_video_cost_per_closing,
          SAFE_DIVIDE(SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN g.spend_usd ELSE 0 END), 
                      SUM(CASE WHEN h.utm_medium IN ('paid-search', 'paid-video') THEN h.funded ELSE 0 END)) as paid_search_video_cost_per_funded

        FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
        LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm ON h.utm_campaign = mm.utm_campaign
        LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m ON mm.adset_name_mapped = m.adset_name AND h.date = m.date
        LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.utm_campaign = g.campaign_name AND h.date = g.date
        LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_history_data` gh ON g.campaign_name = gh.campaign_name
        WHERE {date_filter}
        AND h.utm_medium IN ('paid-social', 'paid-search', 'paid-video')
        """
    
    # Define queries for each period
    queries = {
        'yesterday': get_channel_query(f'h.date = "{yesterday}"', 'Yesterday', ''),
        'same_day_last_week': get_channel_query(f'h.date = "{same_day_last_week}"', 'Same Day Last Week', ''),
        'last_7_days': get_channel_query(f'h.date BETWEEN "{last_7_days_start}" AND "{last_7_days_end}"', 'Last 7 Days', ''),
        'previous_7_days': get_channel_query(f'h.date BETWEEN "{previous_7_days_start}" AND "{previous_7_days_end}"', 'Previous 7 Days', '')
    }
    
    results = {}
    
    for period, sql in queries.items():
        print(f"\n=== EXECUTING {period.upper()} ===")
        
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
                
                # Debug print the key metrics by channel
                paid_social_spend = result.get('paid_social_spend', 0) or 0
                paid_search_video_spend = result.get('paid_search_video_spend', 0) or 0
                total_spend = result.get('total_ad_spend', 0) or 0
                total_leads = result.get('total_leads', 0) or 0
                
                print(f"RESULTS {period}:")
                print(f"  Paid Social: ${paid_social_spend:,.0f}")
                print(f"  Paid Search+Video: ${paid_search_video_spend:,.0f}")
                print(f"  Total: ${total_spend:,.0f}, Leads: {total_leads}")
                
            else:
                print(f"No data returned for {period}")
                results[period] = {}
                
        except Exception as e:
            print(f"ERROR executing {period}: {e}")
            results[period] = {}
    
    return results

def generate_claude_analysis(data):
    """Generate Claude analysis using DSPy"""
    
    # Format the data for Claude
    analysis_prompt = f"""
You are a marketing performance analyst. Analyze this marketing data and provide insights focused on CAMPAIGN-LEVEL performance within the account, not just overall account metrics.

## DATA ANALYSIS

### DAY-OVER-DAY COMPARISON (Yesterday vs Same Day Last Week):

**PAID SOCIAL (Meta)**:
- Yesterday: Spend=${data.get('yesterday', {}).get('paid_social_spend', 0):,.0f}, Impressions={data.get('yesterday', {}).get('paid_social_impressions', 0):,}, CTR={data.get('yesterday', {}).get('paid_social_ctr', 0):.1f}%, Clicks={data.get('yesterday', {}).get('paid_social_clicks', 0):,}, Leads={data.get('yesterday', {}).get('paid_social_leads', 0)}, Estimates={data.get('yesterday', {}).get('paid_social_estimates', 0)}, Closings={data.get('yesterday', {}).get('paid_social_closings', 0)}, Funded={data.get('yesterday', {}).get('paid_social_funded', 0)}, RPTs={data.get('yesterday', {}).get('paid_social_rpts', 0)}
- Same Day Last Week: Spend=${data.get('same_day_last_week', {}).get('paid_social_spend', 0):,.0f}, Impressions={data.get('same_day_last_week', {}).get('paid_social_impressions', 0):,}, CTR={data.get('same_day_last_week', {}).get('paid_social_ctr', 0):.1f}%, Clicks={data.get('same_day_last_week', {}).get('paid_social_clicks', 0):,}, Leads={data.get('same_day_last_week', {}).get('paid_social_leads', 0)}, Estimates={data.get('same_day_last_week', {}).get('paid_social_estimates', 0)}, Closings={data.get('same_day_last_week', {}).get('paid_social_closings', 0)}, Funded={data.get('same_day_last_week', {}).get('paid_social_funded', 0)}, RPTs={data.get('same_day_last_week', {}).get('paid_social_rpts', 0)}
- Costs: Yesterday CPM=${data.get('yesterday', {}).get('paid_social_cpm', 0):.2f}, CPC=${data.get('yesterday', {}).get('paid_social_cpc', 0):.2f}, CPL=${data.get('yesterday', {}).get('paid_social_cost_per_lead', 0):.2f}, CPE=${data.get('yesterday', {}).get('paid_social_cost_per_estimate', 0):.2f}, CPC=${data.get('yesterday', {}).get('paid_social_cost_per_closing', 0):.2f}, CPF=${data.get('yesterday', {}).get('paid_social_cost_per_funded', 0):.2f}
- Costs: Last Week CPM=${data.get('same_day_last_week', {}).get('paid_social_cpm', 0):.2f}, CPC=${data.get('same_day_last_week', {}).get('paid_social_cpc', 0):.2f}, CPL=${data.get('same_day_last_week', {}).get('paid_social_cost_per_lead', 0):.2f}, CPE=${data.get('same_day_last_week', {}).get('paid_social_cost_per_estimate', 0):.2f}, CPC=${data.get('same_day_last_week', {}).get('paid_social_cost_per_closing', 0):.2f}, CPF=${data.get('same_day_last_week', {}).get('paid_social_cost_per_funded', 0):.2f}
- CVRs: Yesterday Estimate={data.get('yesterday', {}).get('paid_social_estimate_cvr', 0):.1f}%, Closing={data.get('yesterday', {}).get('paid_social_closing_cvr', 0):.1f}%, Funded={data.get('yesterday', {}).get('paid_social_funded_cvr', 0):.1f}%
- CVRs: Last Week Estimate={data.get('same_day_last_week', {}).get('paid_social_estimate_cvr', 0):.1f}%, Closing={data.get('same_day_last_week', {}).get('paid_social_closing_cvr', 0):.1f}%, Funded={data.get('same_day_last_week', {}).get('paid_social_funded_cvr', 0):.1f}%

**PAID SEARCH + VIDEO (Google)**:
- Yesterday: Spend=${data.get('yesterday', {}).get('paid_search_video_spend', 0):,.0f}, Impressions={data.get('yesterday', {}).get('paid_search_video_impressions', 0):,}, CTR={data.get('yesterday', {}).get('paid_search_video_ctr', 0):.1f}%, Clicks={data.get('yesterday', {}).get('paid_search_video_clicks', 0):,}, Leads={data.get('yesterday', {}).get('paid_search_video_leads', 0)}, Estimates={data.get('yesterday', {}).get('paid_search_video_estimates', 0)}, Closings={data.get('yesterday', {}).get('paid_search_video_closings', 0)}, Funded={data.get('yesterday', {}).get('paid_search_video_funded', 0)}, RPTs={data.get('yesterday', {}).get('paid_search_video_rpts', 0)}
- Same Day Last Week: Spend=${data.get('same_day_last_week', {}).get('paid_search_video_spend', 0):,.0f}, Impressions={data.get('same_day_last_week', {}).get('paid_search_video_impressions', 0):,}, CTR={data.get('same_day_last_week', {}).get('paid_search_video_ctr', 0):.1f}%, Clicks={data.get('same_day_last_week', {}).get('paid_search_video_clicks', 0):,}, Leads={data.get('same_day_last_week', {}).get('paid_search_video_leads', 0)}, Estimates={data.get('same_day_last_week', {}).get('paid_search_video_estimates', 0)}, Closings={data.get('same_day_last_week', {}).get('paid_search_video_closings', 0)}, Funded={data.get('same_day_last_week', {}).get('paid_search_video_funded', 0)}, RPTs={data.get('same_day_last_week', {}).get('paid_search_video_rpts', 0)}
- Costs: Yesterday CPM=${data.get('yesterday', {}).get('paid_search_video_cpm', 0):.2f}, CPC=${data.get('yesterday', {}).get('paid_search_video_cpc', 0):.2f}, CPL=${data.get('yesterday', {}).get('paid_search_video_cost_per_lead', 0):.2f}, CPE=${data.get('yesterday', {}).get('paid_search_video_cost_per_estimate', 0):.2f}, CPC=${data.get('yesterday', {}).get('paid_search_video_cost_per_closing', 0):.2f}, CPF=${data.get('yesterday', {}).get('paid_search_video_cost_per_funded', 0):.2f}
- Costs: Last Week CPM=${data.get('same_day_last_week', {}).get('paid_search_video_cpm', 0):.2f}, CPC=${data.get('same_day_last_week', {}).get('paid_search_video_cpc', 0):.2f}, CPL=${data.get('same_day_last_week', {}).get('paid_search_video_cost_per_lead', 0):.2f}, CPE=${data.get('same_day_last_week', {}).get('paid_search_video_cost_per_estimate', 0):.2f}, CPC=${data.get('same_day_last_week', {}).get('paid_search_video_cost_per_closing', 0):.2f}, CPF=${data.get('same_day_last_week', {}).get('paid_search_video_cost_per_funded', 0):.2f}
- CVRs: Yesterday Estimate={data.get('yesterday', {}).get('paid_search_video_estimate_cvr', 0):.1f}%, Closing={data.get('yesterday', {}).get('paid_search_video_closing_cvr', 0):.1f}%, Funded={data.get('yesterday', {}).get('paid_search_video_funded_cvr', 0):.1f}%
- CVRs: Last Week Estimate={data.get('same_day_last_week', {}).get('paid_search_video_estimate_cvr', 0):.1f}%, Closing={data.get('same_day_last_week', {}).get('paid_search_video_closing_cvr', 0):.1f}%, Funded={data.get('same_day_last_week', {}).get('paid_search_video_funded_cvr', 0):.1f}%

### WEEK-OVER-WEEK COMPARISON (Last 7 Days vs Previous 7 Days):

**PAID SOCIAL (Meta)**:
- Last 7 Days: Spend=${data.get('last_7_days', {}).get('paid_social_spend', 0):,.0f}, Impressions={data.get('last_7_days', {}).get('paid_social_impressions', 0):,}, CTR={data.get('last_7_days', {}).get('paid_social_ctr', 0):.1f}%, Clicks={data.get('last_7_days', {}).get('paid_social_clicks', 0):,}, Leads={data.get('last_7_days', {}).get('paid_social_leads', 0)}, Estimates={data.get('last_7_days', {}).get('paid_social_estimates', 0)}, Closings={data.get('last_7_days', {}).get('paid_social_closings', 0)}, Funded={data.get('last_7_days', {}).get('paid_social_funded', 0)}, RPTs={data.get('last_7_days', {}).get('paid_social_rpts', 0)}
- Previous 7 Days: Spend=${data.get('previous_7_days', {}).get('paid_social_spend', 0):,.0f}, Impressions={data.get('previous_7_days', {}).get('paid_social_impressions', 0):,}, CTR={data.get('previous_7_days', {}).get('paid_social_ctr', 0):.1f}%, Clicks={data.get('previous_7_days', {}).get('paid_social_clicks', 0):,}, Leads={data.get('previous_7_days', {}).get('paid_social_leads', 0)}, Estimates={data.get('previous_7_days', {}).get('paid_social_estimates', 0)}, Closings={data.get('previous_7_days', {}).get('paid_social_closings', 0)}, Funded={data.get('previous_7_days', {}).get('paid_social_funded', 0)}, RPTs={data.get('previous_7_days', {}).get('paid_social_rpts', 0)}
- Costs: Last 7 Days CPM=${data.get('last_7_days', {}).get('paid_social_cpm', 0):.2f}, CPC=${data.get('last_7_days', {}).get('paid_social_cpc', 0):.2f}, CPL=${data.get('last_7_days', {}).get('paid_social_cost_per_lead', 0):.2f}, CPE=${data.get('last_7_days', {}).get('paid_social_cost_per_estimate', 0):.2f}, CPC=${data.get('last_7_days', {}).get('paid_social_cost_per_closing', 0):.2f}, CPF=${data.get('last_7_days', {}).get('paid_social_cost_per_funded', 0):.2f}
- Costs: Previous 7 Days CPM=${data.get('previous_7_days', {}).get('paid_social_cpm', 0):.2f}, CPC=${data.get('previous_7_days', {}).get('paid_social_cpc', 0):.2f}, CPL=${data.get('previous_7_days', {}).get('paid_social_cost_per_lead', 0):.2f}, CPE=${data.get('previous_7_days', {}).get('paid_social_cost_per_estimate', 0):.2f}, CPC=${data.get('previous_7_days', {}).get('paid_social_cost_per_closing', 0):.2f}, CPF=${data.get('previous_7_days', {}).get('paid_social_cost_per_funded', 0):.2f}
- CVRs: Last 7 Days Estimate={data.get('last_7_days', {}).get('paid_social_estimate_cvr', 0):.1f}%, Closing={data.get('last_7_days', {}).get('paid_social_closing_cvr', 0):.1f}%, Funded={data.get('last_7_days', {}).get('paid_social_funded_cvr', 0):.1f}%
- CVRs: Previous 7 Days Estimate={data.get('previous_7_days', {}).get('paid_social_estimate_cvr', 0):.1f}%, Closing={data.get('previous_7_days', {}).get('paid_social_closing_cvr', 0):.1f}%, Funded={data.get('previous_7_days', {}).get('paid_social_funded_cvr', 0):.1f}%

**PAID SEARCH + VIDEO (Google)**:
- Last 7 Days: Spend=${data.get('last_7_days', {}).get('paid_search_video_spend', 0):,.0f}, Impressions={data.get('last_7_days', {}).get('paid_search_video_impressions', 0):,}, CTR={data.get('last_7_days', {}).get('paid_search_video_ctr', 0):.1f}%, Clicks={data.get('last_7_days', {}).get('paid_search_video_clicks', 0):,}, Leads={data.get('last_7_days', {}).get('paid_search_video_leads', 0)}, Estimates={data.get('last_7_days', {}).get('paid_search_video_estimates', 0)}, Closings={data.get('last_7_days', {}).get('paid_search_video_closings', 0)}, Funded={data.get('last_7_days', {}).get('paid_search_video_funded', 0)}, RPTs={data.get('last_7_days', {}).get('paid_search_video_rpts', 0)}
- Previous 7 Days: Spend=${data.get('previous_7_days', {}).get('paid_search_video_spend', 0):,.0f}, Impressions={data.get('previous_7_days', {}).get('paid_search_video_impressions', 0):,}, CTR={data.get('previous_7_days', {}).get('paid_search_video_ctr', 0):.1f}%, Clicks={data.get('previous_7_days', {}).get('paid_search_video_clicks', 0):,}, Leads={data.get('previous_7_days', {}).get('paid_search_video_leads', 0)}, Estimates={data.get('previous_7_days', {}).get('paid_search_video_estimates', 0)}, Closings={data.get('previous_7_days', {}).get('paid_search_video_closings', 0)}, Funded={data.get('previous_7_days', {}).get('paid_search_video_funded', 0)}, RPTs={data.get('previous_7_days', {}).get('paid_search_video_rpts', 0)}
- Costs: Last 7 Days CPM=${data.get('last_7_days', {}).get('paid_search_video_cpm', 0):.2f}, CPC=${data.get('last_7_days', {}).get('paid_search_video_cpc', 0):.2f}, CPL=${data.get('last_7_days', {}).get('paid_search_video_cost_per_lead', 0):.2f}, CPE=${data.get('last_7_days', {}).get('paid_search_video_cost_per_estimate', 0):.2f}, CPC=${data.get('last_7_days', {}).get('paid_search_video_cost_per_closing', 0):.2f}, CPF=${data.get('last_7_days', {}).get('paid_search_video_cost_per_funded', 0):.2f}
- Costs: Previous 7 Days CPM=${data.get('previous_7_days', {}).get('paid_search_video_cpm', 0):.2f}, CPC=${data.get('previous_7_days', {}).get('paid_search_video_cpc', 0):.2f}, CPL=${data.get('previous_7_days', {}).get('paid_search_video_cost_per_lead', 0):.2f}, CPE=${data.get('previous_7_days', {}).get('paid_search_video_cost_per_estimate', 0):.2f}, CPC=${data.get('previous_7_days', {}).get('paid_search_video_cost_per_closing', 0):.2f}, CPF=${data.get('previous_7_days', {}).get('paid_search_video_cost_per_funded', 0):.2f}
- CVRs: Last 7 Days Estimate={data.get('last_7_days', {}).get('paid_search_video_estimate_cvr', 0):.1f}%, Closing={data.get('last_7_days', {}).get('paid_search_video_closing_cvr', 0):.1f}%, Funded={data.get('last_7_days', {}).get('paid_search_video_funded_cvr', 0):.1f}%
- CVRs: Previous 7 Days Estimate={data.get('previous_7_days', {}).get('paid_search_video_estimate_cvr', 0):.1f}%, Closing={data.get('previous_7_days', {}).get('paid_search_video_closing_cvr', 0):.1f}%, Funded={data.get('previous_7_days', {}).get('paid_search_video_funded_cvr', 0):.1f}%

## OUTPUT REQUIREMENTS

For the MESSAGE field: Focus on CAMPAIGN-LEVEL insights, optimization opportunities, and specific tactical recommendations. Examples:
- "Video campaigns showing strong CTR gains but higher cost per closing suggests creative refresh needed"
- "Search campaigns efficient on lead gen but estimates dropping - review landing page experience"
- "Social campaigns scaling impressions well but CTR declining - test new creative angles"
- "Cost per estimate increasing across both channels - funnel optimization opportunity"

Don't just summarize the overall account performance - provide specific insights about what's happening within campaigns and what actions to take.

Provide analysis for BOTH channels separately in this exact format:

{{
  "message": "Campaign-level insights and tactical recommendations based on performance patterns",
  "colorCode": "green|yellow|red", 
  "paidSocial": {{
    "dayOverDayPulse": {{
      "totalSpend": "value (+/-X.X%)",
      "totalImpressions": "value (+/-X.X%)",
      "cpm": "$X.XX (+/-X.X%)",
      "ctr": "X.X% (+/-X.X%)",
      "totalClicks": "value (+/-X.X%)",
      "cpc": "$X.XX (+/-X.X%)",
      "totalLeads": "value (+/-X.X%)",
      "costPerLead": "$X.XX (+/-X.X%)",
      "estimateCVR": "X.X% (+/-X.X%)",
      "totalEstimates": "value (+/-X.X%)",
      "costPerEstimate": "$X.XX (+/-X.X%)",
      "closingsCVR": "X.X% (+/-X.X%)",
      "totalClosings": "value (+/-X.X%)",
      "costPerClosing": "$X.XX (+/-X.X%)",
      "fundedCVR": "X.X% (+/-X.X%)",
      "totalFunded": "value (+/-X.X%)",
      "costPerFunded": "$X.XX (+/-X.X%)",
      "totalRPTs": "value (+/-X.X%)"
    }},
    "weekOverWeekPulse": {{
      "totalSpend": "value (+/-X.X%)",
      "totalImpressions": "value (+/-X.X%)",
      "cpm": "$X.XX (+/-X.X%)",
      "ctr": "X.X% (+/-X.X%)",
      "totalClicks": "value (+/-X.X%)",
      "cpc": "$X.XX (+/-X.X%)",
      "totalLeads": "value (+/-X.X%)",
      "costPerLead": "$X.XX (+/-X.X%)",
      "estimateCVR": "X.X% (+/-X.X%)",
      "totalEstimates": "value (+/-X.X%)",
      "costPerEstimate": "$X.XX (+/-X.X%)",
      "closingsCVR": "X.X% (+/-X.X%)",
      "totalClosings": "value (+/-X.X%)",
      "costPerClosing": "$X.XX (+/-X.X%)",
      "fundedCVR": "X.X% (+/-X.X%)",
      "totalFunded": "value (+/-X.X%)",
      "costPerFunded": "$X.XX (+/-X.X%)",
      "totalRPTs": "value (+/-X.X%)"
    }}
  }},
  "paidSearchVideo": {{
    "dayOverDayPulse": {{
      "totalSpend": "value (+/-X.X%)",
      "totalImpressions": "value (+/-X.X%)",
      "cpm": "$X.XX (+/-X.X%)",
      "ctr": "X.X% (+/-X.X%)",
      "totalClicks": "value (+/-X.X%)",
      "cpc": "$X.XX (+/-X.X%)",
      "totalLeads": "value (+/-X.X%)",
      "costPerLead": "$X.XX (+/-X.X%)",
      "estimateCVR": "X.X% (+/-X.X%)",
      "totalEstimates": "value (+/-X.X%)",
      "costPerEstimate": "$X.XX (+/-X.X%)",
      "closingsCVR": "X.X% (+/-X.X%)",
      "totalClosings": "value (+/-X.X%)",
      "costPerClosing": "$X.XX (+/-X.X%)",
      "fundedCVR": "X.X% (+/-X.X%)",
      "totalFunded": "value (+/-X.X%)",
      "costPerFunded": "$X.XX (+/-X.X%)",
      "totalRPTs": "value (+/-X.X%)"
    }},
    "weekOverWeekPulse": {{
      "totalSpend": "value (+/-X.X%)",
      "totalImpressions": "value (+/-X.X%)",
      "cpm": "$X.XX (+/-X.X%)",
      "ctr": "X.X% (+/-X.X%)",
      "totalClicks": "value (+/-X.X%)",
      "cpc": "$X.XX (+/-X.X%)",
      "totalLeads": "value (+/-X.X%)",
      "costPerLead": "$X.XX (+/-X.X%)",
      "estimateCVR": "X.X% (+/-X.X%)",
      "totalEstimates": "value (+/-X.X%)",
      "costPerEstimate": "$X.XX (+/-X.X%)",
      "closingsCVR": "X.X% (+/-X.X%)",
      "totalClosings": "value (+/-X.X%)",
      "costPerClosing": "$X.XX (+/-X.X%)",
      "fundedCVR": "X.X% (+/-X.X%)",
      "totalFunded": "value (+/-X.X%)",
      "costPerFunded": "$X.XX (+/-X.X%)",
      "totalRPTs": "value (+/-X.X%)"
    }}
  }}
}}

Calculate percentage changes using: ((New Value - Old Value) / Old Value) × 100
For conversion rates, show percentage change of the rates, not percentage points.
Use "+" for increases, "-" for decreases.
Round to 1 decimal place.

Output ONLY the JSON object. No additional text.
"""

    try:
        # Use DSPy to generate the analysis
        analysis_result = dspy.Predict("analysis -> json_output")(analysis=analysis_prompt)
        
        # Try to parse as JSON
        if hasattr(analysis_result, 'json_output'):
            try:
                return json.loads(analysis_result.json_output)
            except json.JSONDecodeError:
                return {"error": "Failed to parse Claude analysis as JSON", "raw_output": analysis_result.json_output}
        else:
            return {"error": "No json_output field in Claude response", "raw_response": str(analysis_result)}
            
    except Exception as e:
        return {"error": f"Claude analysis failed: {str(e)}"}

@marketing_router.post("/analyze-trends", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(request: TrendAnalysisRequest):
    """Execute SQL with channel separation and generate Claude analysis"""
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                message="BigQuery not available",
                error="BigQuery client not initialized"
            )
        
        # Execute the channel-separated SQL
        data = execute_channel_separated_sql()
        
        # Generate Claude analysis
        claude_analysis = generate_claude_analysis(data)
        
        return TrendAnalysisResponse(
            status="success",
            message="Channel-separated analysis with Claude insights completed",
            data={
                "claude_analysis": claude_analysis,
                "raw_data": data,
                "debug_info": {
                    "yesterday_paid_social_spend": data.get('yesterday', {}).get('paid_social_spend', 0),
                    "yesterday_paid_search_video_spend": data.get('yesterday', {}).get('paid_search_video_spend', 0),
                    "last_7_days_paid_social_spend": data.get('last_7_days', {}).get('paid_social_spend', 0),
                    "last_7_days_paid_search_video_spend": data.get('last_7_days', {}).get('paid_search_video_spend', 0)
                }
            }
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            message="Channel-separated analysis failed",
            error=str(e)
        )

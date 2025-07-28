# marketing_analytics.py - CLAUDE AUTO-ANALYSIS OF AVAILABLE DATA
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import dspy
import os
from datetime import datetime, timedelta, date
import logging
from google.oauth2 import service_account
import json

# BigQuery imports
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
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            bigquery_client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        else:
            bigquery_client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"BigQuery client initialization failed: {e}")
        bigquery_client = None
else:
    bigquery_client = None

# Pydantic Models
class TrendAnalysisResponse(BaseModel):
    status: str
    trend_summary: str
    trend_insights: Optional[Dict[str, Any]] = None
    significant_changes: Optional[List[Dict[str, Any]]] = None
    data_coverage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Create router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API - Auto-Analysis of Available Data",
        "version": "11.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE,
        "note": "Analyzes all available partitioned data automatically"
    }

def get_all_available_data():
    """Pull all available data from BigQuery partitions using correct schema"""
    
    # Daily aggregated view - combines all data using correct column references
    daily_summary_query = """
    WITH meta_daily AS (
      SELECT 
        m.date,
        SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend,
        SUM(SAFE_CAST(m.impressions AS INT64)) as meta_impressions,
        SUM(SAFE_CAST(m.clicks AS INT64)) as meta_clicks,
        SUM(SAFE_CAST(m.leads AS INT64)) as meta_leads
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
      JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
        ON mm.adset_name_mapped = m.adset_name
      GROUP BY m.date
    ),
    google_daily AS (
      SELECT 
        date,
        SUM(spend_usd) as google_spend,
        SUM(impressions) as google_impressions,
        SUM(clicks) as google_clicks,
        SUM(conversions) as google_conversions,
        AVG(ctr_percent) as avg_ctr,
        AVG(cpc_usd) as avg_cpc
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
      GROUP BY date
    ),
    hex_daily AS (
      SELECT 
        date,
        SUM(leads) as hex_leads,
        SUM(start_flows) as hex_start_flows,
        SUM(estimates) as hex_estimates,
        SUM(closings) as hex_closings,
        SUM(funded) as hex_funded,
        SUM(rpts) as hex_rpts
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data`
      GROUP BY date
    )
    
    SELECT 
      COALESCE(m.date, g.date, h.date) as date,
      
      -- Spend metrics
      COALESCE(m.meta_spend, 0) as meta_spend,
      COALESCE(g.google_spend, 0) as google_spend,
      COALESCE(m.meta_spend, 0) + COALESCE(g.google_spend, 0) as total_spend,
      
      -- High funnel metrics
      COALESCE(m.meta_impressions, 0) as meta_impressions,
      COALESCE(g.google_impressions, 0) as google_impressions,
      COALESCE(m.meta_clicks, 0) as meta_clicks,
      COALESCE(g.google_clicks, 0) as google_clicks,
      COALESCE(g.avg_ctr, 0) as google_ctr,
      COALESCE(g.avg_cpc, 0) as google_cpc,
      
      -- Lower funnel metrics  
      COALESCE(h.hex_leads, 0) as leads,
      COALESCE(h.hex_start_flows, 0) as start_flows,
      COALESCE(h.hex_estimates, 0) as estimates,
      COALESCE(h.hex_closings, 0) as closings,
      COALESCE(h.hex_funded, 0) as funded,
      COALESCE(h.hex_rpts, 0) as rpts,
      
      -- Conversion rates
      SAFE_DIVIDE(COALESCE(h.hex_start_flows, 0), COALESCE(h.hex_leads, 0)) * 100 as lead_to_start_rate,
      SAFE_DIVIDE(COALESCE(h.hex_estimates, 0), COALESCE(h.hex_start_flows, 0)) * 100 as start_to_estimate_rate,
      SAFE_DIVIDE(COALESCE(h.hex_closings, 0), COALESCE(h.hex_estimates, 0)) * 100 as estimate_to_closing_rate,
      SAFE_DIVIDE(COALESCE(h.hex_funded, 0), COALESCE(h.hex_closings, 0)) * 100 as closing_to_funded_rate,
      SAFE_DIVIDE(COALESCE(h.hex_funded, 0), COALESCE(h.hex_leads, 0)) * 100 as overall_funded_rate,
      
      -- Cost metrics
      SAFE_DIVIDE(COALESCE(m.meta_spend, 0) + COALESCE(g.google_spend, 0), COALESCE(h.hex_leads, 0)) as cost_per_lead,
      SAFE_DIVIDE(COALESCE(m.meta_spend, 0) + COALESCE(g.google_spend, 0), COALESCE(h.hex_funded, 0)) as cost_per_funded
      
    FROM meta_daily m
    FULL OUTER JOIN google_daily g ON m.date = g.date
    FULL OUTER JOIN hex_daily h ON COALESCE(m.date, g.date) = h.date
    ORDER BY date DESC
    """
    
    # Campaign level breakdown for additional insights
    campaign_summary_query = """
    SELECT 
      h.utm_campaign,
      h.utm_medium,
      h.date,
      SUM(h.leads) as campaign_leads,
      SUM(h.funded) as campaign_funded,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as campaign_funded_rate
    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    GROUP BY h.utm_campaign, h.utm_medium, h.date
    HAVING SUM(h.leads) > 0
    ORDER BY h.date DESC, campaign_leads DESC
    """
    
    try:
        if not bigquery_client:
            raise Exception("BigQuery client not available")
        
        # Execute queries
        daily_summary_df = bigquery_client.query(daily_summary_query).to_dataframe()
        campaign_summary_df = bigquery_client.query(campaign_summary_query).to_dataframe()
        
        # Convert to lists of dicts
        def df_to_dict_list(df):
            if df.empty:
                return []
            result = []
            for _, row in df.iterrows():
                row_data = {}
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        row_data[col] = 0 if col != 'date' else None
                    elif hasattr(value, 'item'):
                        row_data[col] = value.item()
                    else:
                        row_data[col] = str(value) if col == 'date' else value
                result.append(row_data)
            return result
        
        return {
            "daily_summary": df_to_dict_list(daily_summary_df),
            "campaign_summary": df_to_dict_list(campaign_summary_df),
            "data_coverage": {
                "daily_summary_days": len(daily_summary_df) if not daily_summary_df.empty else 0,
                "total_campaigns": len(campaign_summary_df['utm_campaign'].unique()) if not campaign_summary_df.empty else 0,
                "date_range": {
                    "earliest": str(daily_summary_df['date'].min()) if not daily_summary_df.empty else "N/A",
                    "latest": str(daily_summary_df['date'].max()) if not daily_summary_df.empty else "N/A"
                }
            }
        }
        
    except Exception as e:
        print(f"Error pulling available data: {e}")
        return {
            "daily_summary": [],
            "campaign_summary": [],
            "data_coverage": {"error": str(e)}
        }

def generate_claude_auto_analysis(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Claude analyzes whatever data is available automatically with simplified output"""
    
    daily_summary = all_data.get("daily_summary", [])
    campaign_summary = all_data.get("campaign_summary", [])
    data_coverage = all_data.get("data_coverage", {})
    
    if not daily_summary:
        return {
            "trend_summary": "No marketing data available for analysis",
            "error": "Daily summary data not available"
        }
    
    # Simplified analysis prompt to avoid JSON parsing issues
    analysis_prompt = f"""
    Analyze LeaseEnd marketing performance data and provide insights:

    DATA AVAILABLE: {data_coverage.get('daily_summary_days', 0)} days from {data_coverage.get('date_range', {}).get('earliest', 'N/A')} to {data_coverage.get('date_range', {}).get('latest', 'N/A')}

    RECENT DAILY PERFORMANCE (last 5 days):
    {json.dumps(daily_summary[:5], indent=2)}

    TOP CAMPAIGNS:
    {json.dumps(campaign_summary[:10], indent=2)}

    Provide analysis in this EXACT format - no extra formatting or characters:

    {{
        "summary": "Brief 2-sentence overview of key trends",
        "spend_trend": "increasing/decreasing/stable/volatile",
        "lead_trend": "increasing/decreasing/stable/volatile", 
        "efficiency_trend": "improving/declining/stable",
        "key_insights": [
            "First key insight",
            "Second key insight", 
            "Third key insight"
        ],
        "alerts": [
            "Any concerning patterns"
        ],
        "top_opportunity": "Main optimization opportunity"
    }}
    """
    
    try:
        # Use DSPy configured Claude instance
        response = dspy.settings.lm.basic_request(analysis_prompt)
        
        # Clean response more aggressively
        response_clean = response.strip()
        
        # Remove markdown code blocks
        if '```' in response_clean:
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        # Find JSON content more carefully
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            # Fallback if no JSON found
            return create_fallback_analysis(daily_summary, data_coverage)
        
        json_content = response_clean[json_start:json_end]
        
        # Try to parse
        parsed_analysis = json.loads(json_content)
        
        # Add metadata
        parsed_analysis['analysis_metadata'] = {
            'days_analyzed': data_coverage.get('daily_summary_days', 0),
            'campaigns_analyzed': data_coverage.get('total_campaigns', 0),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return parsed_analysis
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response content: {response_clean[:500]}...")
        return create_fallback_analysis(daily_summary, data_coverage)
    except Exception as e:
        print(f"Analysis error: {e}")
        return create_fallback_analysis(daily_summary, data_coverage)

def create_fallback_analysis(daily_summary, data_coverage):
    """Create a simple fallback analysis when Claude's response can't be parsed"""
    
    if not daily_summary:
        return {
            "summary": "No data available for analysis",
            "error": "No daily summary data found"
        }
    
    # Simple calculations from the data
    recent_days = daily_summary[:3] if len(daily_summary) >= 3 else daily_summary
    older_days = daily_summary[3:6] if len(daily_summary) >= 6 else []
    
    def avg_metric(days, metric):
        if not days:
            return 0
        values = [day.get(metric, 0) for day in days if day.get(metric, 0) > 0]
        return sum(values) / len(values) if values else 0
    
    recent_spend = avg_metric(recent_days, 'total_spend')
    older_spend = avg_metric(older_days, 'total_spend')
    recent_leads = avg_metric(recent_days, 'leads')
    older_leads = avg_metric(older_days, 'leads')
    
    spend_trend = "increasing" if recent_spend > older_spend * 1.1 else "decreasing" if recent_spend < older_spend * 0.9 else "stable"
    lead_trend = "increasing" if recent_leads > older_leads * 1.1 else "decreasing" if recent_leads < older_leads * 0.9 else "stable"
    
    return {
        "summary": f"Analysis of {data_coverage.get('daily_summary_days', 0)} days shows {spend_trend} spend trend and {lead_trend} lead generation",
        "spend_trend": spend_trend,
        "lead_trend": lead_trend,
        "efficiency_trend": "stable",
        "key_insights": [
            f"Average daily spend: ${recent_spend:.0f}",
            f"Average daily leads: {recent_leads:.0f}",
            f"Data coverage: {data_coverage.get('daily_summary_days', 0)} days"
        ],
        "alerts": ["Analysis system using fallback mode - full insights limited"],
        "top_opportunity": "Improve analysis system for deeper insights",
        "analysis_metadata": {
            "fallback_mode": True,
            "days_analyzed": data_coverage.get('daily_summary_days', 0),
            "analysis_timestamp": datetime.now().isoformat()
        }
    }

@marketing_router.get("/auto-analysis", response_model=TrendAnalysisResponse)
async def auto_analyze_available_data():
    """Automatically analyze all available marketing data"""
    
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                trend_summary="BigQuery not available for auto-analysis",
                error="BigQuery client not initialized"
            )
        
        # Pull all available data
        all_data = get_all_available_data()
        
        # Generate Claude's auto-analysis
        trend_insights = generate_claude_auto_analysis(all_data)
        
        # Extract components for response
        if isinstance(trend_insights, dict):
            trend_summary = trend_insights.get('summary', 'Auto-analysis completed')
            significant_changes = trend_insights.get('alerts', [])
            
            # Convert alerts to expected format
            if significant_changes:
                significant_changes = [{"finding": alert, "priority": "medium"} for alert in significant_changes]
        else:
            trend_summary = "Analysis completed with limitations"
            significant_changes = []
        data_coverage = all_data.get('data_coverage', {})
        
        return TrendAnalysisResponse(
            status="success",
            trend_summary=trend_summary,
            trend_insights=trend_insights,
            significant_changes=significant_changes,
            data_coverage=data_coverage
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            trend_summary="Auto-analysis failed - see error details",
            error=str(e)
        )

@marketing_router.get("/data-status")
async def check_data_availability():
    """Check what data is available for analysis"""
    
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return {
                "status": "error",
                "message": "BigQuery not available",
                "data_available": False
            }
        
        # Quick data availability check
        availability_check = """
        SELECT 
          'google_ads' as source,
          COUNT(*) as record_count,
          MIN(date) as earliest_date,
          MAX(date) as latest_date
        FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
        
        UNION ALL
        
        SELECT 
          'meta_data' as source,
          COUNT(*) as record_count,
          MIN(date) as earliest_date,
          MAX(date) as latest_date
        FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data`
        
        UNION ALL
        
        SELECT 
          'hex_data' as source,
          COUNT(*) as record_count,
          MIN(date) as earliest_date,
          MAX(date) as latest_date
        FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data`
        """
        
        df = bigquery_client.query(availability_check).to_dataframe()
        
        availability = {}
        for _, row in df.iterrows():
            availability[row['source']] = {
                'record_count': int(row['record_count']),
                'earliest_date': str(row['earliest_date']),
                'latest_date': str(row['latest_date'])
            }
        
        return {
            "status": "success",
            "message": "Data availability checked",
            "data_available": True,
            "sources": availability,
            "ready_for_analysis": all(info['record_count'] > 0 for info in availability.values())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Data availability check failed: {str(e)}",
            "data_available": False,
            "error": str(e)
        }

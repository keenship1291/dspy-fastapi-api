# marketing_analytics.py - CLAUDE TREND ANALYSIS ACROSS TIMEFRAMES
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
class TrendAnalysisRequest(BaseModel):
    date_range: Dict[str, str]
    analysis_depth: str = "comprehensive"

class TrendAnalysisResponse(BaseModel):
    status: str
    trend_summary: str
    trend_insights: Optional[Dict[str, Any]] = None
    significant_changes: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

# Create router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API - Claude Trend Analysis",
        "version": "10.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE,
        "data_availability": {
            "metrics": "Last 14 days",
            "google_ads_history": "Last 30 days"
        }
    }

def get_google_ads_history_query(days_back=30):
    """Get Google Ads historical data for trend analysis"""
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)
    
    return f"""
    SELECT 
      date,
      SUM(spend_usd) as daily_google_spend,
      SUM(impressions) as daily_google_impressions,
      SUM(clicks) as daily_google_clicks,
      SUM(conversions) as daily_google_conversions,
      SAFE_DIVIDE(SUM(clicks), SUM(impressions)) * 100 as daily_google_ctr,
      SAFE_DIVIDE(SUM(spend_usd), SUM(clicks)) as daily_google_cpc,
      COUNT(DISTINCT campaign_name) as active_campaigns
    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
    WHERE date BETWEEN "{start_date}" AND "{end_date}"
    GROUP BY date
    ORDER BY date DESC
    """

def get_hex_funnel_trends_query(days_back=14):
    """Get hex funnel trends for available 14-day period"""
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)
    
    return f"""
    WITH meta_daily AS (
      SELECT 
        m.date,
        SUM(SAFE_CAST(m.spend AS FLOAT64)) as daily_meta_spend,
        SUM(SAFE_CAST(m.impressions AS INT64)) as daily_meta_impressions,
        SUM(SAFE_CAST(m.clicks AS INT64)) as daily_meta_clicks
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
      JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
        ON mm.adset_name_mapped = m.adset_name
      GROUP BY m.date
    )
    
    SELECT 
      h.date,
      
      -- Spend Levels
      SUM(COALESCE(m.daily_meta_spend, 0)) as daily_meta_spend,
      SUM(COALESCE(g.spend_usd, 0)) as daily_google_spend,
      SUM(COALESCE(m.daily_meta_spend, 0)) + SUM(COALESCE(g.spend_usd, 0)) as daily_total_spend,
      
      -- High Funnel Metrics (Platform Data)
      SUM(COALESCE(m.daily_meta_impressions, 0)) as daily_meta_impressions,
      SUM(COALESCE(g.impressions, 0)) as daily_google_impressions,
      SUM(COALESCE(m.daily_meta_clicks, 0)) as daily_meta_clicks,
      SUM(COALESCE(g.clicks, 0)) as daily_google_clicks,
      SUM(COALESCE(m.daily_meta_impressions, 0)) + SUM(COALESCE(g.impressions, 0)) as daily_total_impressions,
      SUM(COALESCE(m.daily_meta_clicks, 0)) + SUM(COALESCE(g.clicks, 0)) as daily_total_clicks,
      
      -- Lower Funnel Metrics (Hex Data)
      SUM(h.leads) as daily_leads,
      SUM(h.start_flows) as daily_start_flows,
      SUM(h.estimates) as daily_estimates,
      SUM(h.closings) as daily_closings,
      SUM(h.funded) as daily_funded,
      SUM(h.rpts) as daily_rpts,
      
      -- Calculated Rates
      SAFE_DIVIDE(SUM(COALESCE(m.daily_meta_clicks, 0)) + SUM(COALESCE(g.clicks, 0)), 
                  SUM(COALESCE(m.daily_meta_impressions, 0)) + SUM(COALESCE(g.impressions, 0))) * 100 as daily_overall_ctr,
      SAFE_DIVIDE(SUM(h.start_flows), SUM(h.leads)) * 100 as daily_lead_to_start_rate,
      SAFE_DIVIDE(SUM(h.estimates), SUM(h.start_flows)) * 100 as daily_start_to_estimate_rate,
      SAFE_DIVIDE(SUM(h.closings), SUM(h.estimates)) * 100 as daily_estimate_to_closing_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.closings)) * 100 as daily_closing_to_funded_rate,
      SAFE_DIVIDE(SUM(h.funded), SUM(h.leads)) * 100 as daily_overall_funded_rate,
      
      -- Cost Metrics
      SAFE_DIVIDE(SUM(COALESCE(m.daily_meta_spend, 0)) + SUM(COALESCE(g.spend_usd, 0)), SUM(h.leads)) as daily_cost_per_lead,
      SAFE_DIVIDE(SUM(COALESCE(m.daily_meta_spend, 0)) + SUM(COALESCE(g.spend_usd, 0)), SUM(h.funded)) as daily_cost_per_funded
      
    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data` h
    LEFT JOIN meta_daily m ON h.date = m.date
    LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g ON h.date = g.date
    WHERE h.date BETWEEN "{start_date}" AND "{end_date}"
    GROUP BY h.date
    ORDER BY h.date DESC
    """

def execute_trend_analysis_queries():
    """Execute queries to get trend data across available timeframes"""
    
    try:
        if not bigquery_client:
            raise Exception("BigQuery client not available")
        
        # Get Google Ads history (30 days)
        google_history_query = get_google_ads_history_query(30)
        google_history_df = bigquery_client.query(google_history_query).to_dataframe()
        
        # Get comprehensive funnel trends (14 days)
        funnel_trends_query = get_hex_funnel_trends_query(14)
        funnel_trends_df = bigquery_client.query(funnel_trends_query).to_dataframe()
        
        # Convert to lists of dicts for easier processing
        google_history = []
        if not google_history_df.empty:
            for _, row in google_history_df.iterrows():
                day_data = {}
                for col in google_history_df.columns:
                    value = row[col]
                    if pd.isna(value):
                        day_data[col] = 0
                    elif hasattr(value, 'item'):
                        day_data[col] = value.item()
                    else:
                        day_data[col] = str(value) if col == 'date' else value
                google_history.append(day_data)
        
        funnel_trends = []
        if not funnel_trends_df.empty:
            for _, row in funnel_trends_df.iterrows():
                day_data = {}
                for col in funnel_trends_df.columns:
                    value = row[col]
                    if pd.isna(value):
                        day_data[col] = 0
                    elif hasattr(value, 'item'):
                        day_data[col] = value.item()
                    else:
                        day_data[col] = str(value) if col == 'date' else value
                funnel_trends.append(day_data)
        
        return {
            "google_ads_history": google_history,
            "funnel_trends": funnel_trends
        }
        
    except Exception as e:
        print(f"Error executing trend analysis queries: {e}")
        return {
            "google_ads_history": [],
            "funnel_trends": []
        }

def generate_claude_trend_analysis(trend_data: Dict[str, Any]) -> Dict[str, Any]:
    """Claude analyzes trends across the available data timeframes"""
    
    google_history = trend_data.get("google_ads_history", [])
    funnel_trends = trend_data.get("funnel_trends", [])
    
    if not google_history and not funnel_trends:
        return {
            "trend_summary": "No trend data available for analysis",
            "error": "No data returned from queries"
        }
    
    # Build trend analysis prompt for Claude
    analysis_prompt = f"""
    Analyze marketing trends for LeaseEnd across available data timeframes:
    
    GOOGLE ADS HISTORY (Last 30 days available):
    {json.dumps(google_history[:10], indent=2) if google_history else "No Google Ads history data"}
    
    COMPLETE FUNNEL TRENDS (Last 14 days available):
    {json.dumps(funnel_trends[:7], indent=2) if funnel_trends else "No funnel trend data"}
    
    Your task: Identify significant TRENDS and CHANGES across these timeframes:
    
    1. SPEND LEVEL TRENDS
       - Google Ads spend patterns over 30 days
       - Meta + Google combined spend trends over 14 days
       - Any notable spending shifts or patterns
    
    2. HIGH FUNNEL METRIC TRENDS
       - Impressions and clicks trends (both platforms)
       - CTR changes over time
       - CPC trends and efficiency shifts
    
    3. LOWER FUNNEL METRIC TRENDS  
       - Lead generation patterns
       - Start flows, estimates, closings progression
       - Funded and RPT trends
       - Conversion rate changes through funnel stages
    
    4. SIGNIFICANT CHANGES
       - Day-over-day changes > 20%
       - Week-over-week patterns
       - Trend reversals or inflection points
       - Outlier days or unusual patterns
    
    Focus on TREND ANALYSIS - what's changing, improving, declining, or showing patterns.
    
    Respond in JSON format:
    {{
        "trend_summary": "2-3 sentence overview of key trends observed",
        "spend_trends": {{
            "google_ads_30day_pattern": "increasing|decreasing|stable|volatile",
            "combined_14day_pattern": "increasing|decreasing|stable|volatile", 
            "notable_spend_changes": ["specific changes observed"]
        }},
        "high_funnel_trends": {{
            "impression_trend": "increasing|decreasing|stable|volatile",
            "click_trend": "increasing|decreasing|stable|volatile",
            "ctr_trend": "improving|declining|stable|volatile",
            "efficiency_changes": ["specific CTR/CPC changes"]
        }},
        "lower_funnel_trends": {{
            "lead_generation_trend": "increasing|decreasing|stable|volatile",
            "conversion_rate_trends": {{
                "lead_to_start": "improving|declining|stable",
                "start_to_estimate": "improving|declining|stable", 
                "estimate_to_closing": "improving|declining|stable",
                "closing_to_funded": "improving|declining|stable"
            }},
            "funnel_health_changes": ["specific conversion changes"]
        }},
        "significant_changes": [
            {{
                "metric": "metric_name",
                "change_type": "spike|drop|trend_reversal|outlier",
                "magnitude": "percentage_or_description",
                "timeframe": "when_it_occurred",
                "impact": "business_impact"
            }}
        ],
        "trend_alerts": ["concerning_patterns_or_opportunities"],
        "data_quality_notes": "any_data_gaps_or_limitations"
    }}
    """
    
    try:
        # Use DSPy configured Claude instance
        response = dspy.settings.lm.basic_request(analysis_prompt)
        
        # Clean and parse response
        response_clean = response.strip()
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        # Find JSON content
        if not response_clean.startswith('{'):
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                response_clean = response_clean[json_start:json_end]
        
        trend_analysis = json.loads(response_clean)
        
        # Add metadata
        trend_analysis['analysis_metadata'] = {
            'google_ads_days_analyzed': len(google_history),
            'funnel_days_analyzed': len(funnel_trends),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return trend_analysis
        
    except json.JSONDecodeError as e:
        return {
            "trend_summary": "Trend analysis completed with parsing limitations - key patterns identified",
            "spend_trends": {
                "google_ads_30day_pattern": "data_processing_error",
                "notable_spend_changes": ["Analysis parsing failed - manual review needed"]
            },
            "high_funnel_trends": {
                "impression_trend": "unknown",
                "click_trend": "unknown", 
                "ctr_trend": "unknown",
                "efficiency_changes": ["Unable to analyze due to parsing error"]
            },
            "lower_funnel_trends": {
                "lead_generation_trend": "unknown",
                "conversion_rate_trends": {
                    "lead_to_start": "unknown",
                    "start_to_estimate": "unknown",
                    "estimate_to_closing": "unknown", 
                    "closing_to_funded": "unknown"
                },
                "funnel_health_changes": ["Parsing error prevented analysis"]
            },
            "significant_changes": [
                {
                    "metric": "analysis_system",
                    "change_type": "parsing_error",
                    "magnitude": "100%",
                    "timeframe": "current_analysis",
                    "impact": "Manual review required"
                }
            ],
            "trend_alerts": [f"Analysis parsing error: {str(e)}"],
            "data_quality_notes": "JSON parsing failed - review analysis system",
            "error": str(e)
        }
    except Exception as e:
        return {
            "trend_summary": "Trend analysis failed due to technical error",
            "error": str(e)
        }

@marketing_router.post("/trend-analysis", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(request: TrendAnalysisRequest):
    """Execute trend analysis across Google Ads history and funnel metrics"""
    
    try:
        if not BIGQUERY_AVAILABLE or not bigquery_client:
            return TrendAnalysisResponse(
                status="error",
                trend_summary="BigQuery not available for trend analysis",
                error="BigQuery client not initialized"
            )
        
        # Execute trend analysis queries
        trend_data = execute_trend_analysis_queries()
        
        # Generate Claude's trend analysis
        trend_insights = generate_claude_trend_analysis(trend_data)
        
        # Extract components for response
        trend_summary = trend_insights.get('trend_summary', 'Trend analysis completed')
        significant_changes = trend_insights.get('significant_changes', [])
        
        return TrendAnalysisResponse(
            status="success",
            trend_summary=trend_summary,
            trend_insights=trend_insights,
            significant_changes=significant_changes
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            trend_summary="Trend analysis failed - see error details",
            error=str(e)
        )

@marketing_router.post("/quick-trends")
async def get_quick_trend_insights(request: TrendAnalysisRequest):
    """Get quick trend insights focusing on major changes"""
    
    try:
        # Get recent trend data
        trend_data = execute_trend_analysis_queries()
        
        funnel_trends = trend_data.get("funnel_trends", [])
        
        if not funnel_trends:
            return {
                "status": "error",
                "trends": ["No recent data available for trend analysis"],
                "top_change": "Data unavailable",
                "alert_level": "high"
            }
        
        # Quick analysis of recent vs previous periods
        recent_days = funnel_trends[:3]  # Last 3 days
        previous_days = funnel_trends[3:6] if len(funnel_trends) >= 6 else []
        
        def avg_metric(days, metric):
            if not days:
                return 0
            return sum(day.get(metric, 0) for day in days) / len(days)
        
        recent_leads = avg_metric(recent_days, 'daily_leads')
        previous_leads = avg_metric(previous_days, 'daily_leads')
        recent_spend = avg_metric(recent_days, 'daily_total_spend')
        previous_spend = avg_metric(previous_days, 'daily_total_spend')
        
        # Quick Claude analysis
        quick_prompt = f"""
        Quick trend analysis for LeaseEnd:
        
        Recent 3 days avg: {recent_leads:.0f} leads, ${recent_spend:.0f} spend
        Previous 3 days avg: {previous_leads:.0f} leads, ${previous_spend:.0f} spend
        
        Identify the biggest trend change and alert level.
        
        Respond in JSON: {{"trends": ["...", "...", "..."], "top_change": "...", "alert_level": "high|medium|low"}}
        """
        
        response = dspy.settings.lm.basic_request(quick_prompt)
        response_clean = response.strip()
        
        if not response_clean.startswith('{'):
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                response_clean = response_clean[json_start:json_end]
        
        quick_analysis = json.loads(response_clean)
        
        return {
            "status": "success",
            "trends": quick_analysis.get('trends', []),
            "top_change": quick_analysis.get('top_change', 'No major changes detected'),
            "alert_level": quick_analysis.get('alert_level', 'low'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "trends": [f"Quick trend analysis failed: {str(e)}"],
            "top_change": "Analysis error",
            "alert_level": "high",
            "timestamp": datetime.now().isoformat()
        }

@marketing_router.get("/test-trends")
async def test_trend_analysis():
    """Test endpoint for trend analysis"""
    
    test_request = TrendAnalysisRequest(
        date_range={"since": "2025-01-14", "until": "2025-01-28"},
        analysis_depth="comprehensive"
    )
    
    try:
        return await analyze_marketing_trends(test_request)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Test failed: {str(e)}",
            "error": str(e)
        }

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
    """Pull all available data from BigQuery partitions"""
    
    # Google Ads data query - pulls all available data
    google_ads_query = """
    SELECT 
      date,
      campaign_name,
      SUM(spend_usd) as daily_spend,
      SUM(impressions) as daily_impressions,
      SUM(clicks) as daily_clicks,
      SUM(conversions) as daily_conversions,
      SAFE_DIVIDE(SUM(clicks), SUM(impressions)) * 100 as daily_ctr,
      SAFE_DIVIDE(SUM(spend_usd), SUM(clicks)) as daily_cpc
    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
    GROUP BY date, campaign_name
    ORDER BY date DESC, daily_spend DESC
    """
    
    # Meta data query - pulls all available data
    meta_data_query = """
    SELECT 
      m.date,
      mm.utm_campaign,
      SUM(SAFE_CAST(m.spend AS FLOAT64)) as daily_spend,
      SUM(SAFE_CAST(m.impressions AS INT64)) as daily_impressions,
      SUM(SAFE_CAST(m.clicks AS INT64)) as daily_clicks,
      SUM(SAFE_CAST(m.leads AS INT64)) as daily_leads,
      SAFE_DIVIDE(SUM(SAFE_CAST(m.clicks AS INT64)), SUM(SAFE_CAST(m.impressions AS INT64))) * 100 as daily_ctr,
      SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)), SUM(SAFE_CAST(m.clicks AS INT64))) as daily_cpc
    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
    JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
      ON mm.adset_name_mapped = m.adset_name
    GROUP BY m.date, mm.utm_campaign
    ORDER BY m.date DESC, daily_spend DESC
    """
    
    # Hex funnel data query - pulls all available data
    hex_funnel_query = """
    SELECT 
      date,
      utm_campaign,
      utm_medium,
      SUM(leads) as daily_leads,
      SUM(start_flows) as daily_start_flows,
      SUM(estimates) as daily_estimates,
      SUM(closings) as daily_closings,
      SUM(funded) as daily_funded,
      SUM(rpts) as daily_rpts,
      
      -- Conversion rates
      SAFE_DIVIDE(SUM(start_flows), SUM(leads)) * 100 as lead_to_start_rate,
      SAFE_DIVIDE(SUM(estimates), SUM(start_flows)) * 100 as start_to_estimate_rate,
      SAFE_DIVIDE(SUM(closings), SUM(estimates)) * 100 as estimate_to_closing_rate,
      SAFE_DIVIDE(SUM(funded), SUM(closings)) * 100 as closing_to_funded_rate,
      SAFE_DIVIDE(SUM(funded), SUM(leads)) * 100 as overall_funded_rate
      
    FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data`
    GROUP BY date, utm_campaign, utm_medium
    ORDER BY date DESC, daily_leads DESC
    """
    
    # Daily aggregated view - combines all data
    daily_summary_query = """
    WITH meta_daily AS (
      SELECT 
        m.date,
        SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend,
        SUM(SAFE_CAST(m.impressions AS INT64)) as meta_impressions,
        SUM(SAFE_CAST(m.clicks AS INT64)) as meta_clicks
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
        SUM(conversions) as google_conversions
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
      COALESCE(m.meta_impressions, 0) + COALESCE(g.google_impressions, 0) as total_impressions,
      COALESCE(m.meta_clicks, 0) + COALESCE(g.google_clicks, 0) as total_clicks,
      SAFE_DIVIDE(COALESCE(m.meta_clicks, 0) + COALESCE(g.google_clicks, 0), 
                  COALESCE(m.meta_impressions, 0) + COALESCE(g.google_impressions, 0)) * 100 as overall_ctr,
      
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
    
    try:
        if not bigquery_client:
            raise Exception("BigQuery client not available")
        
        # Execute all queries
        google_ads_df = bigquery_client.query(google_ads_query).to_dataframe()
        meta_data_df = bigquery_client.query(meta_data_query).to_dataframe()
        hex_funnel_df = bigquery_client.query(hex_funnel_query).to_dataframe()
        daily_summary_df = bigquery_client.query(daily_summary_query).to_dataframe()
        
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
            "google_ads_data": df_to_dict_list(google_ads_df),
            "meta_data": df_to_dict_list(meta_data_df),
            "hex_funnel_data": df_to_dict_list(hex_funnel_df),
            "daily_summary": df_to_dict_list(daily_summary_df),
            "data_coverage": {
                "google_ads_days": len(google_ads_df['date'].unique()) if not google_ads_df.empty else 0,
                "meta_data_days": len(meta_data_df['date'].unique()) if not meta_data_df.empty else 0,
                "hex_funnel_days": len(hex_funnel_df['date'].unique()) if not hex_funnel_df.empty else 0,
                "daily_summary_days": len(daily_summary_df) if not daily_summary_df.empty else 0,
                "date_range": {
                    "earliest": str(min([
                        google_ads_df['date'].min() if not google_ads_df.empty else date.max,
                        meta_data_df['date'].min() if not meta_data_df.empty else date.max,
                        hex_funnel_df['date'].min() if not hex_funnel_df.empty else date.max
                    ])),
                    "latest": str(max([
                        google_ads_df['date'].max() if not google_ads_df.empty else date.min,
                        meta_data_df['date'].max() if not meta_data_df.empty else date.min,
                        hex_funnel_df['date'].max() if not hex_funnel_df.empty else date.min
                    ]))
                }
            }
        }
        
    except Exception as e:
        print(f"Error pulling available data: {e}")
        return {
            "google_ads_data": [],
            "meta_data": [],
            "hex_funnel_data": [],
            "daily_summary": [],
            "data_coverage": {"error": str(e)}
        }

def generate_claude_auto_analysis(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Claude analyzes whatever data is available automatically"""
    
    daily_summary = all_data.get("daily_summary", [])
    google_ads = all_data.get("google_ads_data", [])
    meta_data = all_data.get("meta_data", [])
    hex_funnel = all_data.get("hex_funnel_data", [])
    data_coverage = all_data.get("data_coverage", {})
    
    if not daily_summary and not google_ads and not meta_data and not hex_funnel:
        return {
            "trend_summary": "No marketing data available for analysis",
            "error": "All data sources returned empty"
        }
    
    # Build comprehensive analysis prompt
    analysis_prompt = f"""
    Auto-analyze all available LeaseEnd marketing data and identify key trends, patterns, and insights:
    
    DATA COVERAGE:
    {json.dumps(data_coverage, indent=2)}
    
    DAILY PERFORMANCE SUMMARY:
    {json.dumps(daily_summary[:10], indent=2) if daily_summary else "No daily summary data"}
    
    GOOGLE ADS DETAIL (Sample):
    {json.dumps(google_ads[:5], indent=2) if google_ads else "No Google Ads data"}
    
    META ADVERTISING DETAIL (Sample):
    {json.dumps(meta_data[:5], indent=2) if meta_data else "No Meta data"}
    
    HEX FUNNEL DETAIL (Sample):
    {json.dumps(hex_funnel[:5], indent=2) if hex_funnel else "No funnel data"}
    
    Your task: Analyze ALL available data to identify:
    
    1. SPEND PATTERNS & TRENDS
       - Overall spending trends across time
       - Channel allocation patterns (Meta vs Google)
       - Budget efficiency and waste identification
    
    2. HIGH FUNNEL PERFORMANCE
       - Impression and click volume trends
       - CTR performance across channels and time
       - CPC trends and cost efficiency changes
    
    3. CONVERSION FUNNEL ANALYSIS
       - Lead generation patterns and quality
       - Funnel stage conversion rates and trends
       - Drop-off points and optimization opportunities
    
    4. CROSS-CHANNEL INSIGHTS
       - Channel performance comparison
       - Attribution and measurement considerations
       - Synergy effects between platforms
    
    5. SIGNIFICANT ANOMALIES
       - Unusual spikes, drops, or patterns
       - Outlier performance days
       - Trend reversals or inflection points
    
    Focus on actionable insights and trend identification from the available data.
    
    Respond in JSON format:
    {{
        "trend_summary": "Executive summary of key findings across all available data",
        "spend_analysis": {{
            "overall_trend": "increasing|decreasing|stable|volatile",
            "channel_allocation": "meta_dominant|google_dominant|balanced|shifting",
            "efficiency_trend": "improving|declining|stable",
            "spend_insights": ["key spend patterns observed"]
        }},
        "high_funnel_performance": {{
            "impression_trends": "summary of impression patterns",
            "click_trends": "summary of click patterns", 
            "ctr_analysis": "CTR performance insights",
            "cpc_efficiency": "Cost efficiency observations"
        }},
        "conversion_funnel": {{
            "lead_generation": "lead volume and quality trends",
            "funnel_health": {{
                "lead_to_start": "performance assessment",
                "start_to_estimate": "performance assessment",
                "estimate_to_closing": "performance assessment", 
                "closing_to_funded": "performance assessment"
            }},
            "bottlenecks": ["identified conversion bottlenecks"],
            "opportunities": ["funnel optimization opportunities"]
        }},
        "cross_channel_insights": {{
            "performance_leader": "meta|google|balanced",
            "attribution_notes": "measurement and attribution observations",
            "synergy_effects": "cross-channel interaction insights",
            "reallocation_opportunities": ["budget optimization suggestions"]
        }},
        "significant_findings": [
            {{
                "finding": "specific_insight_or_anomaly",
                "impact": "business_impact_assessment",
                "action_needed": "recommended_response",
                "priority": "high|medium|low"
            }}
        ],
        "data_quality_assessment": "assessment of data completeness and reliability",
        "recommended_actions": ["prioritized list of actions based on analysis"]
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
        
        auto_analysis = json.loads(response_clean)
        
        # Add analysis metadata
        auto_analysis['analysis_metadata'] = {
            'data_sources_analyzed': len([k for k, v in all_data.items() if v and k != 'data_coverage']),
            'total_data_points': sum(len(v) if isinstance(v, list) else 0 for v in all_data.values()),
            'analysis_timestamp': datetime.now().isoformat(),
            'data_coverage': data_coverage
        }
        
        return auto_analysis
        
    except json.JSONDecodeError as e:
        return {
            "trend_summary": f"Auto-analysis completed with parsing limitations. Data coverage: {data_coverage.get('daily_summary_days', 0)} days analyzed.",
            "spend_analysis": {
                "overall_trend": "analysis_error",
                "spend_insights": ["JSON parsing failed - manual review needed"]
            },
            "significant_findings": [
                {
                    "finding": "Analysis parsing error occurred",
                    "impact": "Strategic insights not fully accessible", 
                    "action_needed": "Review analysis system and retry",
                    "priority": "high"
                }
            ],
            "data_quality_assessment": f"Data available but parsing failed: {str(e)}",
            "error": str(e)
        }
    except Exception as e:
        return {
            "trend_summary": "Auto-analysis failed due to technical error",
            "error": str(e)
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
        trend_summary = trend_insights.get('trend_summary', 'Auto-analysis completed')
        significant_changes = trend_insights.get('significant_findings', [])
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

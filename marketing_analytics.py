# marketing_analytics.py - FUNNEL ANALYSIS + ANOMALY DETECTION
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import json
from datetime import datetime, timedelta

# BigQuery imports (for spend data)
try:
    from google.cloud import bigquery
    import pandas as pd
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

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
        bigquery_client = None
else:
    bigquery_client = None

# Models
class FunnelDataPoint(BaseModel):
    date: str
    leads: int
    start_flows: int
    estimates: int
    closings: int
    funded: int
    rpts: int

class FunnelAnalysisRequest(BaseModel):
    funnel_data: List[FunnelDataPoint]

class FunnelAnalysisResponse(BaseModel):
    status: str
    colorCode: Optional[str] = None
    formatted_metrics: Optional[Dict[str, Any]] = None
    day_over_day_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class CampaignAnomaly(BaseModel):
    platform: str
    campaign_name: str
    adset_name: Optional[str] = None
    utm_campaign: str
    metric_issues: List[str]
    severity: str
    recent_spend: float
    baseline_spend: float
    recent_cac: Optional[float] = None
    baseline_cac: Optional[float] = None
    recent_closings: Optional[int] = None
    baseline_closings: Optional[int] = None
    cpc_change_pct: Optional[float] = None
    cac_change_pct: Optional[float] = None
    closings_change_pct: Optional[float] = None
    performance_status: str
    campaign_changes: Optional[str] = None

class AnomalyQueryData(BaseModel):
    platform: str
    campaign_name: str
    adset_name: Optional[str] = None
    utm_campaign: Optional[str] = None
    recent_spend: float
    baseline_spend: float
    recent_cpc: Optional[float] = None
    baseline_cpc: Optional[float] = None
    recent_closings: Optional[int] = None
    baseline_closings: Optional[int] = None
    recent_cac: Optional[float] = None
    baseline_cac: Optional[float] = None
    cpc_change_pct: Optional[float] = None
    cac_change_pct: Optional[float] = None
    closings_change_pct: Optional[float] = None
    has_anomaly: Optional[int] = None
    performance_status: Optional[str] = None

class AnomalyAnalysisRequest(BaseModel):
    query_results: List[AnomalyQueryData]
    analysis_date: Optional[str] = None
    baseline_date: Optional[str] = None

class AnomalyAnalysisResponse(BaseModel):
    status: str
    analysis_date: Optional[str] = None
    anomaly_count: Optional[int] = None
    analysis_period: Optional[str] = None
    executive_summary: Optional[str] = None
    critical_issues: Optional[List[str]] = None
    immediate_actions: Optional[List[str]] = None
    campaign_recommendations: Optional[str] = None
    root_cause_analysis: Optional[str] = None
    market_insights: Optional[str] = None
    optimization_strategy: Optional[str] = None
    budget_allocation: Optional[str] = None
    raw_anomalies: Optional[List[CampaignAnomaly]] = None
    error: Optional[str] = None

# Router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API - Funnel Analysis + Anomaly Detection",
        "version": "15.0.0",
        "status": "running"
    }

def get_spend_data():
    """Get total spend from BigQuery"""
    if not bigquery_client:
        return []
    
    try:
        query = """
        WITH meta_daily AS (
          SELECT 
            m.date,
            SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
          JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
            ON mm.adset_name_mapped = m.adset_name
          GROUP BY m.date
        ),
        google_daily AS (
          SELECT 
            date,
            SUM(spend_usd) as google_spend
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
          GROUP BY date
        )
        
        SELECT 
          COALESCE(m.date, g.date) as date,
          COALESCE(m.meta_spend, 0) + COALESCE(g.google_spend, 0) as total_spend
        FROM meta_daily m
        FULL OUTER JOIN google_daily g ON m.date = g.date
        ORDER BY date DESC
        """
        
        df = bigquery_client.query(query).to_dataframe()
        return [{"date": str(row['date']), "total_spend": float(row['total_spend'])} for _, row in df.iterrows()]
    except:
        return []

def calculate_changes(funnel_data: List[FunnelDataPoint]):
    """Calculate both week-over-week and day-over-day changes"""
    
    if len(funnel_data) < 8:
        return {"error": "Need at least 8 days of data for day-over-day comparison"}
    
    # Sort by date (most recent first)
    sorted_data = sorted(funnel_data, key=lambda x: x.date, reverse=True)
    
    # Week-over-week: Last 7 days vs previous 7 days
    if len(sorted_data) >= 14:
        last_7 = sorted_data[:7]
        prev_7 = sorted_data[7:14]
        week_over_week = True
    else:
        week_over_week = False
    
    # Day-over-day: Most recent day vs same day last week (-7 days)
    most_recent_day = sorted_data[0]
    same_day_last_week = sorted_data[7] if len(sorted_data) > 7 else None
    
    def sum_metric(days, metric):
        return sum(getattr(day, metric) for day in days)
    
    def get_metric(day, metric):
        return getattr(day, metric)
    
    # Calculate week-over-week totals
    wow_data = {}
    if week_over_week:
        last_totals = {
            'leads': sum_metric(last_7, 'leads'),
            'start_flows': sum_metric(last_7, 'start_flows'), 
            'estimates': sum_metric(last_7, 'estimates'),
            'closings': sum_metric(last_7, 'closings'),
            'funded': sum_metric(last_7, 'funded'),
            'rpts': sum_metric(last_7, 'rpts')
        }
        
        prev_totals = {
            'leads': sum_metric(prev_7, 'leads'),
            'start_flows': sum_metric(prev_7, 'start_flows'),
            'estimates': sum_metric(prev_7, 'estimates'), 
            'closings': sum_metric(prev_7, 'closings'),
            'funded': sum_metric(prev_7, 'funded'),
            'rpts': sum_metric(prev_7, 'rpts')
        }
        
        wow_data = {
            'last_totals': last_totals,
            'prev_totals': prev_totals
        }
    
    # Calculate day-over-day totals
    dod_data = {}
    if same_day_last_week:
        recent_day_totals = {
            'leads': get_metric(most_recent_day, 'leads'),
            'start_flows': get_metric(most_recent_day, 'start_flows'),
            'estimates': get_metric(most_recent_day, 'estimates'),
            'closings': get_metric(most_recent_day, 'closings'),
            'funded': get_metric(most_recent_day, 'funded'),
            'rpts': get_metric(most_recent_day, 'rpts')
        }
        
        same_day_totals = {
            'leads': get_metric(same_day_last_week, 'leads'),
            'start_flows': get_metric(same_day_last_week, 'start_flows'),
            'estimates': get_metric(same_day_last_week, 'estimates'),
            'closings': get_metric(same_day_last_week, 'closings'),
            'funded': get_metric(same_day_last_week, 'funded'),
            'rpts': get_metric(same_day_last_week, 'rpts')
        }
        
        dod_data = {
            'recent_day_totals': recent_day_totals,
            'same_day_totals': same_day_totals,
            'recent_date': most_recent_day.date,
            'comparison_date': same_day_last_week.date
        }
    
    # Get spend data
    spend_data = get_spend_data()
    spend_dict = {item['date']: item['total_spend'] for item in spend_data}
    
    # Add spend to week-over-week
    if week_over_week:
        last_7_dates = [day.date for day in last_7]
        prev_7_dates = [day.date for day in prev_7]
        
        last_spend = sum(spend_dict.get(date, 0) for date in last_7_dates)
        prev_spend = sum(spend_dict.get(date, 0) for date in prev_7_dates)
        
        wow_data.update({
            'last_spend': last_spend,
            'prev_spend': prev_spend
        })
    
    # Add spend to day-over-day
    if same_day_last_week:
        recent_spend = spend_dict.get(most_recent_day.date, 0)
        comparison_spend = spend_dict.get(same_day_last_week.date, 0)
        
        dod_data.update({
            'recent_spend': recent_spend,
            'comparison_spend': comparison_spend
        })
    
    return {
        'week_over_week': wow_data if week_over_week else {},
        'day_over_day': dod_data if same_day_last_week else {},
        'has_week_data': week_over_week,
        'has_day_data': bool(same_day_last_week)
    }

def format_metric(current, previous, is_currency=False):
    """Format metric with change percentage"""
    if previous == 0:
        change_pct = 100.0 if current > 0 else 0.0
    else:
        change_pct = ((current - previous) / previous) * 100
    
    # Format current value
    if is_currency:
        if current >= 1000:
            current_str = f"${current/1000:,.0f}k"
        else:
            current_str = f"${current:,.0f}"
    else:
        current_str = f"{current:,.0f}"
    
    # Format change
    sign = "+" if change_pct >= 0 else ""
    return f"{current_str} ({sign}{change_pct:.1f}%)"

def determine_color(changes):
    """Determine color based on performance"""
    
    # Use week-over-week data if available, otherwise day-over-day
    if changes.get('has_week_data'):
        last = changes['week_over_week']['last_totals']
        prev = changes['week_over_week']['prev_totals']
    elif changes.get('has_day_data'):
        last = changes['day_over_day']['recent_day_totals']
        prev = changes['day_over_day']['same_day_totals']
    else:
        return "yellow"  # No data for comparison
    
    def pct_change(curr, prev):
        return ((curr - prev) / prev) * 100 if prev > 0 else 0
    
    funded_change = pct_change(last['funded'], prev['funded'])
    revenue_change = pct_change(last['rpts'], prev['rpts'])
    
    if funded_change > 10 and revenue_change > 15:
        return "green"
    elif funded_change < -10 or revenue_change < -15:
        return "red"
    else:
        return "yellow"

# ANOMALY DETECTION FUNCTIONS
def get_analysis_dates():
    """Get the most recent date and same day last week from the data"""
    if not bigquery_client:
        return None, None
        
    try:
        latest_date_query = """
        SELECT MAX(date) as latest_date
        FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data`
        """
        
        result = bigquery_client.query(latest_date_query).to_dataframe()
        latest_date = result['latest_date'].iloc[0]
        baseline_date = latest_date - timedelta(days=7)
        
        return latest_date.strftime('%Y-%m-%d'), baseline_date.strftime('%Y-%m-%d')
    except:
        return None, None

def get_meta_anomalies(recent_date: str, baseline_date: str) -> List[CampaignAnomaly]:
    """Get Meta campaign anomalies from BigQuery"""
    if not bigquery_client:
        return []
        
    try:
        meta_query = f"""
        WITH hex_summary AS (
          SELECT 
            date,
            utm_campaign,
            SUM(leads) as leads,
            SUM(closings) as closings
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data`
          WHERE date IN ('{recent_date}', '{baseline_date}')
          GROUP BY date, utm_campaign
        ),

        meta_with_funnel AS (
          SELECT 
            'meta' as platform,
            m.campaign_name,
            m.adset_name,
            mm.utm_campaign,
            
            MAX(CASE WHEN m.date = '{recent_date}' THEN SAFE_CAST(m.spend AS FLOAT64) END) as recent_spend,
            MAX(CASE WHEN m.date = '{recent_date}' THEN SAFE_DIVIDE(SAFE_CAST(m.spend AS FLOAT64), SAFE_CAST(m.clicks AS INT64)) END) as recent_cpc,
            MAX(CASE WHEN h.date = '{recent_date}' THEN h.closings END) as recent_closings,
            
            MAX(CASE WHEN m.date = '{baseline_date}' THEN SAFE_CAST(m.spend AS FLOAT64) END) as baseline_spend,
            MAX(CASE WHEN m.date = '{baseline_date}' THEN SAFE_DIVIDE(SAFE_CAST(m.spend AS FLOAT64), SAFE_CAST(m.clicks AS INT64)) END) as baseline_cpc,
            MAX(CASE WHEN h.date = '{baseline_date}' THEN h.closings END) as baseline_closings
            
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m
          LEFT JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
            ON (m.campaign_name = mm.campaign_name_mapped AND m.adset_name = mm.adset_name_mapped)
            OR (m.campaign_name = mm.campaign_name_mapped AND mm.adset_name_mapped IS NULL)
          LEFT JOIN hex_summary h 
            ON m.date = h.date AND mm.utm_campaign = h.utm_campaign
          WHERE m.date IN ('{recent_date}', '{baseline_date}')
          GROUP BY m.campaign_name, m.adset_name, mm.utm_campaign
          HAVING recent_spend IS NOT NULL AND baseline_spend IS NOT NULL
        )

        SELECT 
          platform,
          campaign_name,
          adset_name,
          utm_campaign,
          recent_spend,
          baseline_spend,
          recent_cpc,
          baseline_cpc,
          recent_closings,
          baseline_closings,
          
          CASE 
            WHEN recent_closings > 0 THEN ROUND(recent_spend / recent_closings, 2)
            ELSE NULL 
          END as recent_cac,
          
          CASE 
            WHEN baseline_closings > 0 THEN ROUND(baseline_spend / baseline_closings, 2)
            ELSE NULL 
          END as baseline_cac,
          
          CASE 
            WHEN baseline_cpc > 0 THEN ROUND(((recent_cpc - baseline_cpc) / baseline_cpc) * 100, 1)
            ELSE NULL 
          END as cpc_change_pct,
          
          CASE 
            WHEN baseline_closings > 0 AND recent_closings > 0 AND baseline_spend > 0 THEN 
              ROUND((((recent_spend / recent_closings) - (baseline_spend / baseline_closings)) / (baseline_spend / baseline_closings)) * 100, 1)
            ELSE NULL 
          END as cac_change_pct,
          
          CASE 
            WHEN baseline_closings > 0 THEN ROUND(((recent_closings - baseline_closings) / baseline_closings) * 100, 1)
            ELSE NULL 
          END as closings_change_pct,
          
          CASE 
            WHEN recent_spend = 0 AND baseline_spend > 10 THEN 1
            WHEN baseline_cpc > 0 AND ((recent_cpc - baseline_cpc) / baseline_cpc) * 100 >= 30 THEN 1
            WHEN recent_closings = 0 AND baseline_closings > 0 AND recent_spend > 50 THEN 1
            WHEN baseline_closings > 0 AND recent_closings > 0 AND baseline_spend > 0 AND 
                 (((recent_spend / recent_closings) - (baseline_spend / baseline_closings)) / (baseline_spend / baseline_closings)) * 100 >= 25 THEN 1
            ELSE 0
          END as has_anomaly,
          
          CASE 
            WHEN recent_closings > baseline_closings THEN 'IMPROVED'
            WHEN recent_closings = 0 AND baseline_closings > 0 THEN 'BROKEN'
            WHEN recent_closings < baseline_closings THEN 'DECLINED' 
            WHEN recent_spend > baseline_spend * 1.5 AND recent_closings <= baseline_closings THEN 'INEFFICIENT'
            ELSE 'STABLE'
          END as performance_status

        FROM meta_with_funnel
        WHERE has_anomaly = 1 OR performance_status IN ('BROKEN', 'DECLINED', 'INEFFICIENT')
        ORDER BY has_anomaly DESC, cac_change_pct DESC, recent_spend DESC
        LIMIT 20;
        """
        
        df = bigquery_client.query(meta_query).to_dataframe()
        anomalies = []
        
        for _, row in df.iterrows():
            severity = "high" if row.get('performance_status') == 'BROKEN' else "medium"
            
            issues = []
            if row.get('cpc_change_pct', 0) >= 100:
                issues.append(f"CPC spiked {row['cpc_change_pct']:.1f}%")
            if row.get('cac_change_pct', 0) >= 50:
                issues.append(f"CAC increased {row['cac_change_pct']:.1f}%")
            if row.get('closings_change_pct', 0) <= -50:
                issues.append(f"Closings dropped {abs(row['closings_change_pct']):.1f}%")
            if row.get('recent_closings', 0) == 0 and row.get('baseline_closings', 0) > 0:
                issues.append("Zero closings generated")
            
            anomaly = CampaignAnomaly(
                platform="meta",
                campaign_name=row['campaign_name'],
                adset_name=row.get('adset_name'),
                utm_campaign=row.get('utm_campaign', ''),
                metric_issues=issues,
                severity=severity,
                recent_spend=row.get('recent_spend', 0),
                baseline_spend=row.get('baseline_spend', 0),
                recent_cac=row.get('recent_cac'),
                baseline_cac=row.get('baseline_cac'),
                recent_closings=row.get('recent_closings'),
                baseline_closings=row.get('baseline_closings'),
                cpc_change_pct=row.get('cpc_change_pct'),
                cac_change_pct=row.get('cac_change_pct'),
                closings_change_pct=row.get('closings_change_pct'),
                performance_status=row.get('performance_status', 'UNKNOWN')
            )
            anomalies.append(anomaly)
        
        return anomalies
    except Exception as e:
        print(f"Error getting meta anomalies: {e}")
        return []

def get_google_anomalies(recent_date: str, baseline_date: str) -> List[CampaignAnomaly]:
    """Get Google campaign anomalies from BigQuery"""
    if not bigquery_client:
        return []
        
    try:
        google_query = f"""
        WITH hex_closings AS (
          SELECT 
            date,
            utm_campaign,
            SUM(closings) as closings
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_data`
          WHERE date IN ('{recent_date}', '{baseline_date}')
            AND platform IN ('google_ads', 'youtube')
            AND utm_campaign IS NOT NULL
          GROUP BY date, utm_campaign
        ),

        google_with_funnel AS (
          SELECT 
            'google' as platform,
            g.campaign_name,
            g.campaign_name as utm_campaign,
            
            MAX(CASE WHEN g.date = '{recent_date}' THEN g.spend_usd END) as recent_spend,
            MAX(CASE WHEN g.date = '{recent_date}' THEN g.cpc_usd END) as recent_cpc,
            MAX(CASE WHEN h.date = '{recent_date}' THEN h.closings END) as recent_closings,
            
            MAX(CASE WHEN g.date = '{baseline_date}' THEN g.spend_usd END) as baseline_spend,
            MAX(CASE WHEN g.date = '{baseline_date}' THEN g.cpc_usd END) as baseline_cpc,
            MAX(CASE WHEN h.date = '{baseline_date}' THEN h.closings END) as baseline_closings
            
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g
          LEFT JOIN hex_closings h 
            ON g.date = h.date AND g.campaign_name = h.utm_campaign
          WHERE g.date IN ('{recent_date}', '{baseline_date}')
          GROUP BY g.campaign_name
          HAVING recent_spend IS NOT NULL AND baseline_spend IS NOT NULL
        )

        SELECT 
          platform,
          campaign_name,
          utm_campaign,
          recent_spend,
          baseline_spend,
          recent_cpc,
          baseline_cpc,
          recent_closings,
          baseline_closings,
          
          CASE 
            WHEN recent_closings > 0 THEN ROUND(recent_spend / recent_closings, 2)
            ELSE NULL 
          END as recent_cac,
          
          CASE 
            WHEN baseline_closings > 0 THEN ROUND(baseline_spend / baseline_closings, 2)
            ELSE NULL 
          END as baseline_cac,
          
          CASE 
            WHEN baseline_cpc > 0 THEN ROUND(((recent_cpc - baseline_cpc) / baseline_cpc) * 100, 1)
            ELSE NULL 
          END as cpc_change_pct,
          
          CASE 
            WHEN baseline_closings > 0 AND recent_closings > 0 AND baseline_spend > 0 THEN 
              ROUND((((recent_spend / recent_closings) - (baseline_spend / baseline_closings)) / (baseline_spend / baseline_closings)) * 100, 1)
            ELSE NULL 
          END as cac_change_pct,
          
          CASE 
            WHEN baseline_closings > 0 THEN ROUND(((recent_closings - baseline_closings) / baseline_closings) * 100, 1)
            ELSE NULL 
          END as closings_change_pct,
          
          CASE 
            WHEN recent_spend = 0 AND baseline_spend > 25 THEN 1
            WHEN baseline_cpc > 0 AND ((recent_cpc - baseline_cpc) / baseline_cpc) * 100 >= 35 THEN 1
            WHEN recent_closings = 0 AND baseline_closings > 0 AND recent_spend > 100 THEN 1
            WHEN baseline_closings > 0 AND recent_closings > 0 AND baseline_spend > 0 AND 
                 (((recent_spend / recent_closings) - (baseline_spend / baseline_closings)) / (baseline_spend / baseline_closings)) * 100 >= 30 THEN 1
            ELSE 0
          END as has_anomaly,
          
          CASE 
            WHEN recent_closings > baseline_closings THEN 'IMPROVED'
            WHEN recent_closings = 0 AND baseline_closings > 0 THEN 'BROKEN'
            WHEN recent_closings < baseline_closings THEN 'DECLINED' 
            WHEN recent_spend > baseline_spend * 1.5 AND recent_closings <= baseline_closings THEN 'INEFFICIENT'
            ELSE 'STABLE'
          END as performance_status

        FROM google_with_funnel
        WHERE has_anomaly = 1 OR performance_status IN ('BROKEN', 'DECLINED', 'INEFFICIENT')
        ORDER BY has_anomaly DESC, cac_change_pct DESC, recent_spend DESC
        LIMIT 20;
        """
        
        df = bigquery_client.query(google_query).to_dataframe()
        anomalies = []
        
        for _, row in df.iterrows():
            severity = "high" if row.get('performance_status') == 'BROKEN' else "medium"
            
            issues = []
            if row.get('cpc_change_pct', 0) >= 35:
                issues.append(f"CPC increased {row['cpc_change_pct']:.1f}%")
            if row.get('cac_change_pct', 0) >= 50:
                issues.append(f"CAC spiked {row['cac_change_pct']:.1f}%")
            if row.get('closings_change_pct', 0) <= -50:
                issues.append(f"Closings declined {abs(row['closings_change_pct']):.1f}%")
            if row.get('recent_closings', 0) == 0 and row.get('baseline_closings', 0) > 0:
                issues.append("No closings despite spend")
            
            anomaly = CampaignAnomaly(
                platform="google",
                campaign_name=row['campaign_name'],
                utm_campaign=row.get('utm_campaign', ''),
                metric_issues=issues,
                severity=severity,
                recent_spend=row.get('recent_spend', 0),
                baseline_spend=row.get('baseline_spend', 0),
                recent_cac=row.get('recent_cac'),
                baseline_cac=row.get('baseline_cac'),
                recent_closings=row.get('recent_closings'),
                baseline_closings=row.get('baseline_closings'),
                cpc_change_pct=row.get('cpc_change_pct'),
                cac_change_pct=row.get('cac_change_pct'),
                closings_change_pct=row.get('closings_change_pct'),
                performance_status=row.get('performance_status', 'UNKNOWN')
            )
            anomalies.append(anomaly)
        
        return anomalies
    except Exception as e:
        print(f"Error getting google anomalies: {e}")
        return []

def process_query_results_to_anomalies(query_results: List[AnomalyQueryData]) -> List[CampaignAnomaly]:
    """Convert query results to CampaignAnomaly objects"""
    anomalies = []
    
    for row in query_results:
        # Only process if it has anomaly flag or problematic status
        if (row.has_anomaly == 1 or 
            row.performance_status in ['BROKEN', 'DECLINED', 'INEFFICIENT']):
            
            # Determine severity
            severity = "high" if row.performance_status == 'BROKEN' else "medium"
            
            # Build issue list
            issues = []
            if row.cpc_change_pct and row.cpc_change_pct >= 30:
                issues.append(f"CPC {'spiked' if row.cpc_change_pct >= 100 else 'increased'} {row.cpc_change_pct:.1f}%")
            if row.cac_change_pct and row.cac_change_pct >= 25:
                issues.append(f"CAC {'spiked' if row.cac_change_pct >= 50 else 'increased'} {row.cac_change_pct:.1f}%")
            if row.closings_change_pct and row.closings_change_pct <= -25:
                issues.append(f"Closings dropped {abs(row.closings_change_pct):.1f}%")
            if row.recent_closings == 0 and row.baseline_closings and row.baseline_closings > 0:
                issues.append("Zero closings generated")
            if row.recent_spend == 0 and row.baseline_spend > 10:
                issues.append("Campaign stopped spending")
            
            # If no specific issues identified, add generic one
            if not issues and row.performance_status:
                issues.append(f"Performance status: {row.performance_status}")
            
            anomaly = CampaignAnomaly(
                platform=row.platform,
                campaign_name=row.campaign_name,
                adset_name=row.adset_name,
                utm_campaign=row.utm_campaign or '',
                metric_issues=issues,
                severity=severity,
                recent_spend=row.recent_spend,
                baseline_spend=row.baseline_spend,
                recent_cac=row.recent_cac,
                baseline_cac=row.baseline_cac,
                recent_closings=row.recent_closings,
                baseline_closings=row.baseline_closings,
                cpc_change_pct=row.cpc_change_pct,
                cac_change_pct=row.cac_change_pct,
                closings_change_pct=row.closings_change_pct,
                performance_status=row.performance_status or 'UNKNOWN'
            )
            anomalies.append(anomaly)
    
    return anomalies

def create_simple_analysis(anomalies: List[CampaignAnomaly], recent_date: str, baseline_date: str) -> Dict[str, Any]:
    """Create a simple analysis without Claude/DSPy for now"""
    
    # Basic summary
    high_severity = [a for a in anomalies if a.severity == "high"]
    broken_campaigns = [a for a in anomalies if a.performance_status == "BROKEN"]
    total_affected_spend = sum(a.recent_spend for a in anomalies)
    
    # Generate simple insights
    executive_summary = f"Found {len(anomalies)} campaign anomalies. {len(high_severity)} are high severity with {len(broken_campaigns)} completely broken campaigns affecting ${total_affected_spend:,.0f} in daily spend."
    
    critical_issues = []
    immediate_actions = []
    
    for anomaly in anomalies[:5]:  # Top 5 issues
        if anomaly.performance_status == "BROKEN":
            critical_issues.append(f"{anomaly.platform.title()} campaign '{anomaly.campaign_name}' is broken - no closings despite ${anomaly.recent_spend:,.0f} spend")
            immediate_actions.append(f"Pause or investigate {anomaly.campaign_name} immediately")
        elif anomaly.cac_change_pct and anomaly.cac_change_pct >= 50:
            critical_issues.append(f"{anomaly.platform.title()} campaign '{anomaly.campaign_name}' CAC spiked {anomaly.cac_change_pct:.1f}%")
            immediate_actions.append(f"Review targeting and bids for {anomaly.campaign_name}")
    
    return {
        "analysis_date": recent_date,
        "anomaly_count": len(anomalies),
        "analysis_period": f"{recent_date} vs {baseline_date} (same day last week)",
        "executive_summary": executive_summary,
        "critical_issues": critical_issues[:5],
        "immediate_actions": immediate_actions[:5],
        "campaign_recommendations": "Review top performing campaigns and reallocate budget from underperforming ones",
        "root_cause_analysis": "Anomalies likely due to audience fatigue, increased competition, or recent platform changes",
        "market_insights": "Monitor for broader market trends affecting performance",
        "optimization_strategy": "Focus on campaigns with stable CAC and good closing rates",
        "budget_allocation": f"Consider reallocating from ${total_affected_spend:,.0f} in affected spend to better performing campaigns",
        "raw_anomalies": anomalies
    }

# ENDPOINTS
@marketing_router.post("/funnel-analysis", response_model=FunnelAnalysisResponse)
async def analyze_funnel(request: FunnelAnalysisRequest):
    """Simple funnel analysis"""
    
    try:
        # Calculate changes
        changes = calculate_changes(request.funnel_data)
        
        if 'error' in changes:
            return FunnelAnalysisResponse(
                status="error",
                error=changes['error']
            )
        
        # Format week-over-week metrics (if available)
        formatted_metrics = {}
        if changes.get('has_week_data'):
            wow = changes['week_over_week']
            last = wow['last_totals']
            prev = wow['prev_totals']
            
            formatted_metrics = {
                "spendMetrics": {
                    "channel": "adSpend",
                    "period_type": "weekOverWeekPulse",
                    "totalSpend": format_metric(wow['last_spend'], wow['prev_spend'], True)
                },
                "funnelMetrics": {
                    "channel": "funnelPerformance", 
                    "period_type": "weekOverWeekPulse",
                    "totalLeads": format_metric(last['leads'], prev['leads']),
                    "startFlows": format_metric(last['start_flows'], prev['start_flows']),
                    "estimates": format_metric(last['estimates'], prev['estimates']),
                    "closings": format_metric(last['closings'], prev['closings']),
                    "funded": format_metric(last['funded'], prev['funded']),
                    "revenue": format_metric(last['rpts'], prev['rpts'], True)
                }
            }
        
        # Format day-over-day metrics (if available)
        day_over_day_metrics = {}
        if changes.get('has_day_data'):
            dod = changes['day_over_day']
            recent = dod['recent_day_totals']
            comparison = dod['same_day_totals']
            
            day_over_day_metrics = {
                "spendMetrics": {
                    "channel": "adSpend",
                    "period_type": "dayOverDayPulse",
                    "totalSpend": format_metric(dod['recent_spend'], dod['comparison_spend'], True)
                },
                "funnelMetrics": {
                    "channel": "funnelPerformance",
                    "period_type": "dayOverDayPulse", 
                    "totalLeads": format_metric(recent['leads'], comparison['leads']),
                    "startFlows": format_metric(recent['start_flows'], comparison['start_flows']),
                    "estimates": format_metric(recent['estimates'], comparison['estimates']),
                    "closings": format_metric(recent['closings'], comparison['closings']),
                    "funded": format_metric(recent['funded'], comparison['funded']),
                    "revenue": format_metric(recent['rpts'], comparison['rpts'], True)
                },
                "comparison_info": {
                    "recent_date": dod['recent_date'],
                    "comparison_date": dod['comparison_date']
                }
            }
        
        # Determine color
        color_code = determine_color(changes)
        
        return FunnelAnalysisResponse(
            status="success",
            colorCode=color_code,
            formatted_metrics=formatted_metrics if formatted_metrics else None,
            day_over_day_metrics=day_over_day_metrics if day_over_day_metrics else None
        )
        
    except Exception as e:
        return FunnelAnalysisResponse(
            status="error",
            error=str(e)
        )

@marketing_router.post("/anomaly-analysis", response_model=AnomalyAnalysisResponse)
async def analyze_campaign_anomalies(request: AnomalyAnalysisRequest):
    """Analyze campaign anomalies from query results"""
    
    try:
        # Process query results into anomaly objects
        all_anomalies = process_query_results_to_anomalies(request.query_results)
        
        # Determine dates
        analysis_date = request.analysis_date or datetime.now().strftime('%Y-%m-%d')
        baseline_date = request.baseline_date or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        if not all_anomalies:
            return AnomalyAnalysisResponse(
                status="success",
                analysis_date=analysis_date,
                anomaly_count=0,
                analysis_period=f"{analysis_date} vs {baseline_date}",
                executive_summary="No significant anomalies detected in campaign performance",
                critical_issues=[],
                immediate_actions=["Continue monitoring campaign performance"],
                raw_anomalies=[]
            )
        
        # Create analysis
        analysis = create_simple_analysis(all_anomalies, analysis_date, baseline_date)
        
        return AnomalyAnalysisResponse(
            status="success",
            **analysis
        )
        
    except Exception as e:
        return AnomalyAnalysisResponse(
            status="error",
            error=str(e)
        )

@marketing_router.get("/anomaly-analysis", response_model=AnomalyAnalysisResponse)
async def analyze_campaign_anomalies_get():
    """Detect and analyze marketing campaign anomalies (BigQuery version)"""
    
    try:
        if not bigquery_client:
            return AnomalyAnalysisResponse(
                status="error",
                error="BigQuery client not available"
            )
        
        # Get analysis dates
        recent_date, baseline_date = get_analysis_dates()
        if not recent_date or not baseline_date:
            return AnomalyAnalysisResponse(
                status="error", 
                error="Could not determine analysis dates"
            )
        
        # Get anomalies from both platforms
        meta_anomalies = get_meta_anomalies(recent_date, baseline_date)
        google_anomalies = get_google_anomalies(recent_date, baseline_date)
        
        all_anomalies = meta_anomalies + google_anomalies
        
        if not all_anomalies:
            return AnomalyAnalysisResponse(
                status="success",
                analysis_date=recent_date,
                anomaly_count=0,
                analysis_period=f"{recent_date} vs {baseline_date}",
                executive_summary="No significant anomalies detected in campaign performance",
                critical_issues=[],
                immediate_actions=["Continue monitoring campaign performance"],
                raw_anomalies=[]
            )
        
        # Create analysis
        analysis = create_simple_analysis(all_anomalies, recent_date, baseline_date)
        
        return AnomalyAnalysisResponse(
            status="success",
            **analysis
        )
        
    except Exception as e:
        return AnomalyAnalysisResponse(
            status="error",
            error=str(e)
        )

# marketing_analytics.py - SIMPLE FUNNEL ANALYSIS
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import json
from datetime import datetime

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

# Router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API - Simple Funnel Analysis",
        "version": "13.0.0",
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

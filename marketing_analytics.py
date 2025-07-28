# marketing_analytics.py - SIMPLE FUNNEL ANALYSIS + ANOMALY DETECTION
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import os
import json
import asyncio
import httpx
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

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

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

class AnomalyAlert(BaseModel):
    campaign_name: str
    adset_name: Optional[str] = None
    platform: str  # "meta" or "google"
    metric: str
    current_value: float
    previous_value: float
    change_percent: float
    severity: str  # "yellow", "orange", "red"
    message: str
    claude_analysis: Optional[str] = None

class ClaudeAnalysisRequest(BaseModel):
    alerts: List[AnomalyAlert]
    context: Dict[str, Any]

class AnomalyCheckResponse(BaseModel):
    status: str
    alerts: List[AnomalyAlert]
    summary: Dict[str, int]  # Count of alerts by severity
    claude_summary: Optional[str] = None  # Overall analysis from Claude
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

# NEW ANOMALY DETECTION CODE BELOW

@marketing_router.get("/check-anomalies", response_model=AnomalyCheckResponse)
async def check_campaign_anomalies():
    """Check for performance anomalies in ad campaigns compared to same day last week"""
    
    if not bigquery_client:
        return AnomalyCheckResponse(
            status="error",
            alerts=[],
            summary={},
            error="BigQuery client not available"
        )
    
    try:
        alerts = []
        
        # Check Meta anomalies
        meta_alerts = check_meta_anomalies()
        alerts.extend(meta_alerts)
        
        # Check Google anomalies  
        google_alerts = check_google_anomalies()
        alerts.extend(google_alerts)
        
        # Get additional context for Claude analysis
        context = get_campaign_context()
        
        # Analyze alerts with Claude if we have any
        if alerts and ANTHROPIC_API_KEY:
            try:
                # Get Claude's analysis for individual alerts
                for alert in alerts:
                    alert.claude_analysis = await analyze_alert_with_claude(alert, context)
                
                # Get overall summary from Claude
                claude_summary = await get_claude_summary(alerts, context)
            except Exception as e:
                # Don't fail the whole request if Claude analysis fails
                claude_summary = f"Claude analysis unavailable: {str(e)}"
        else:
            claude_summary = None
        
        # Create summary
        summary = {
            "yellow": len([a for a in alerts if a.severity == "yellow"]),
            "orange": len([a for a in alerts if a.severity == "orange"]),
            "red": len([a for a in alerts if a.severity == "red"])
        }
        
        return AnomalyCheckResponse(
            status="success",
            alerts=alerts,
            summary=summary,
            claude_summary=claude_summary
        )
        
    except Exception as e:
        return AnomalyCheckResponse(
            status="error",
            alerts=[],
            summary={},
            claude_summary=None,
            error=str(e)
        )

def get_campaign_context() -> Dict[str, Any]:
    """Get additional context about campaigns for Claude analysis"""
    
    if not bigquery_client:
        return {}
    
    try:
        # Get recent performance trends (last 7 days)
        trend_query = """
        WITH meta_trends AS (
          SELECT 
            'meta' as platform,
            mm.campaign_name,
            m.date,
            SUM(SAFE_CAST(m.spend AS FLOAT64)) as daily_spend,
            SUM(SAFE_CAST(m.impressions AS INT64)) as daily_impressions,
            SUM(SAFE_CAST(m.clicks AS INT64)) as daily_clicks
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
          JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
            ON mm.adset_name_mapped = m.adset_name
          WHERE m.date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
          GROUP BY mm.campaign_name, m.date
        ),
        google_trends AS (
          SELECT 
            'google' as platform,
            campaign_name,
            date,
            SUM(spend_usd) as daily_spend,
            SUM(impressions) as daily_impressions,
            SUM(clicks) as daily_clicks
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
          WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
          GROUP BY campaign_name, date
        )
        SELECT * FROM meta_trends
        UNION ALL
        SELECT * FROM google_trends
        ORDER BY platform, campaign_name, date DESC
        """
        
        df = bigquery_client.query(trend_query).to_dataframe()
        
        # Process trends by campaign
        trends = {}
        for _, row in df.iterrows():
            key = f"{row['platform']}_{row['campaign_name']}"
            if key not in trends:
                trends[key] = []
            trends[key].append({
                'date': str(row['date']),
                'spend': float(row['daily_spend']) if pd.notna(row['daily_spend']) else 0,
                'impressions': int(row['daily_impressions']) if pd.notna(row['daily_impressions']) else 0,
                'clicks': int(row['daily_clicks']) if pd.notna(row['daily_clicks']) else 0
            })
        
        return {
            'recent_trends': trends,
            'analysis_date': str(datetime.now().date())
        }
        
    except Exception as e:
        return {'error': f"Could not fetch context: {str(e)}"}

async def analyze_alert_with_claude(alert: AnomalyAlert, context: Dict[str, Any]) -> str:
    """Get Claude's analysis for a specific alert"""
    
    if not ANTHROPIC_API_KEY:
        return None
    
    # Get relevant trend data for this campaign
    campaign_key = f"{alert.platform}_{alert.campaign_name}"
    trends = context.get('recent_trends', {}).get(campaign_key, [])
    
    prompt = f"""You are analyzing a digital marketing performance anomaly. Provide a brief, actionable analysis.

Alert Details:
- Platform: {alert.platform.title()}
- Campaign: {alert.campaign_name}
{f"- Ad Set: {alert.adset_name}" if alert.adset_name else ""}
- Metric: {alert.metric}
- Current Value: {alert.current_value:.2f}
- Previous Value (same day last week): {alert.previous_value:.2f}
- Change: {alert.change_percent:.1f}%
- Severity: {alert.severity}

Recent 7-day trend data:
{json.dumps(trends[-7:], indent=2) if trends else "No trend data available"}

Please provide a concise analysis (2-3 sentences) covering:
1. Most likely cause(s) for this anomaly (especially for CAC spikes: audience saturation, creative fatigue, increased competition, bid changes, etc.)
2. Whether to investigate immediately or monitor for a few more days
3. Specific area to investigate first (creative refresh, audience expansion, bid adjustments, etc.)

Focus on actionable insights for a marketing manager managing CAC and performance."""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                CLAUDE_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 200,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text'].strip()
            else:
                return f"Analysis unavailable (API error: {response.status_code})"
                
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"

async def get_claude_summary(alerts: List[AnomalyAlert], context: Dict[str, Any]) -> str:
    """Get Claude's overall summary of all alerts"""
    
    if not ANTHROPIC_API_KEY or not alerts:
        return None
    
    # Organize alerts by severity and platform
    alert_summary = {
        'red': [a for a in alerts if a.severity == 'red'],
        'orange': [a for a in alerts if a.severity == 'orange'], 
        'yellow': [a for a in alerts if a.severity == 'yellow']
    }
    
    prompt = f"""You are analyzing multiple digital marketing performance anomalies. Provide a strategic overview.

Alert Summary:
- Red (Critical): {len(alert_summary['red'])} alerts
- Orange (Moderate): {len(alert_summary['orange'])} alerts  
- Yellow (Minor): {len(alert_summary['yellow'])} alerts

Critical Issues (Red):
{chr(10).join([f"• {a.platform.title()}: {a.campaign_name} - {a.metric} {a.change_percent:.1f}% ({f'${a.current_value:.2f} CAC' if a.metric == 'cost_per_closing' else f'{a.current_value:.2f}'})" for a in alert_summary['red']]) if alert_summary['red'] else "None"}

Moderate Issues (Orange):
{chr(10).join([f"• {a.platform.title()}: {a.campaign_name} - {a.metric} {a.change_percent:.1f}% ({f'${a.current_value:.2f} CAC' if a.metric == 'cost_per_closing' else f'{a.current_value:.2f}'})" for a in alert_summary['orange'][:3]]) if alert_summary['orange'] else "None"}
{'...' if len(alert_summary['orange']) > 3 else ''}

Please provide a strategic summary (3-4 sentences) covering:
1. Overall priority level for these issues (focus on CAC spikes as highest priority)
2. Any patterns you notice across platforms/campaigns (audience fatigue, competition, seasonality)
3. Recommended immediate action plan (prioritize CAC optimization first)

Keep it executive-friendly and action-oriented. Remember that CAC (cost per closing) is the most critical metric to address."""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                CLAUDE_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text'].strip()
            else:
                return f"Summary unavailable (API error: {response.status_code})"
                
    except Exception as e:
        return f"Summary unavailable: {str(e)}"

def check_meta_anomalies() -> List[AnomalyAlert]:
    """Check for anomalies in Meta ad data, focusing on CAC (cost per closing)"""
    
    query = """
    WITH hex_closings AS (
      SELECT 
        date,
        utm_adset as adset_name,
        COUNT(*) as closings
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_table`
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
        AND utm_adset IS NOT NULL
        AND utm_adset != ''
      GROUP BY date, utm_adset
    ),
    daily_metrics AS (
      SELECT 
        m.date,
        mm.campaign_name,
        m.adset_name,
        SUM(SAFE_CAST(m.spend AS FLOAT64)) as spend,
        SUM(SAFE_CAST(m.impressions AS INT64)) as impressions,
        SUM(SAFE_CAST(m.clicks AS INT64)) as clicks,
        SUM(SAFE_CAST(m.leads AS INT64)) as leads,
        COALESCE(h.closings, 0) as closings,
        -- Calculate CAC (Cost per Closing) and other metrics
        SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)), COALESCE(h.closings, 0)) as cost_per_closing,
        SAFE_DIVIDE(SUM(SAFE_CAST(m.clicks AS INT64)), SUM(SAFE_CAST(m.impressions AS INT64))) * 100 as ctr,
        SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)), SUM(SAFE_CAST(m.clicks AS INT64))) as cpc,
        SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)), SUM(SAFE_CAST(m.impressions AS INT64))) * 1000 as cpm,
        SAFE_DIVIDE(SUM(SAFE_CAST(m.spend AS FLOAT64)), SUM(SAFE_CAST(m.leads AS INT64))) as cost_per_lead
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
      JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
        ON mm.adset_name_mapped = m.adset_name
      LEFT JOIN hex_closings h 
        ON m.date = h.date AND m.adset_name = h.adset_name
      WHERE m.date >= DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
      GROUP BY m.date, mm.campaign_name, m.adset_name, h.closings
      HAVING spend > 0  -- Only include adsets that spent money
    ),
    comparison_data AS (
      SELECT 
        campaign_name,
        adset_name,
        -- Most recent day
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN spend END) as recent_spend,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN closings END) as recent_closings,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cost_per_closing END) as recent_cac,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN ctr END) as recent_ctr,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cpc END) as recent_cpc,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cpm END) as recent_cpm,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cost_per_lead END) as recent_cost_per_lead,
        -- Same day last week
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN spend END) as prev_spend,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN closings END) as prev_closings,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cost_per_closing END) as prev_cac,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN ctr END) as prev_ctr,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cpc END) as prev_cpc,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cpm END) as prev_cpm,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cost_per_lead END) as prev_cost_per_lead
      FROM daily_metrics
      GROUP BY campaign_name, adset_name
    )
    SELECT * FROM comparison_data
    WHERE recent_spend IS NOT NULL AND prev_spend IS NOT NULL
    """
    
    try:
        df = bigquery_client.query(query).to_dataframe()
        alerts = []
        
        for _, row in df.iterrows():
            # PRIORITY 1: Check CAC (Cost per Closing) - Most Important
            if pd.notna(row['recent_cac']) and pd.notna(row['prev_cac']) and row['prev_cac'] > 0:
                cac_change = ((row['recent_cac'] - row['prev_cac']) / row['prev_cac']) * 100
                if cac_change >= 25:  # CAC increased by 25%+
                    severity = "red" if cac_change >= 75 else "orange" if cac_change >= 50 else "yellow"
                    alerts.append(AnomalyAlert(
                        campaign_name=row['campaign_name'],
                        adset_name=row['adset_name'],
                        platform="meta",
                        metric="cost_per_closing",
                        current_value=row['recent_cac'],
                        previous_value=row['prev_cac'],
                        change_percent=cac_change,
                        severity=severity,
                        message=f"CAC spiked: ${row['recent_cac']:.2f} vs ${row['prev_cac']:.2f} (closings: {row['recent_closings']} vs {row['prev_closings']})"
                    ))
            
            # Check for no closings when there were closings before
            elif row['recent_closings'] == 0 and row['prev_closings'] > 0 and row['recent_spend'] > 50:
                alerts.append(AnomalyAlert(
                    campaign_name=row['campaign_name'],
                    adset_name=row['adset_name'],
                    platform="meta",
                    metric="closings",
                    current_value=0,
                    previous_value=row['prev_closings'],
                    change_percent=-100,
                    severity="red",
                    message=f"No closings generated despite ${row['recent_spend']:.2f} spend (had {row['prev_closings']} closings last week)"
                ))
            
            # Check for spending issues (campaign not spending)
            if row['recent_spend'] == 0 and row['prev_spend'] > 10:
                alerts.append(AnomalyAlert(
                    campaign_name=row['campaign_name'],
                    adset_name=row['adset_name'],
                    platform="meta",
                    metric="spend",
                    current_value=0,
                    previous_value=row['prev_spend'],
                    change_percent=-100,
                    severity="red",
                    message=f"Adset stopped spending completely (was ${row['prev_spend']:.2f})"
                ))
            
            # LOWER PRIORITY: Other metrics (only if no CAC issues)
            # Only check these if CAC is not spiking significantly
            has_major_cac_issue = (pd.notna(row['recent_cac']) and pd.notna(row['prev_cac']) and 
                                 row['prev_cac'] > 0 and 
                                 ((row['recent_cac'] - row['prev_cac']) / row['prev_cac']) * 100 >= 50)
            
            if not has_major_cac_issue:
                # Check CTR (lower is bad)
                if pd.notna(row['recent_ctr']) and pd.notna(row['prev_ctr']) and row['prev_ctr'] > 0:
                    ctr_change = ((row['recent_ctr'] - row['prev_ctr']) / row['prev_ctr']) * 100
                    if ctr_change <= -30:  # CTR dropped by 30%+
                        severity = "orange" if ctr_change <= -50 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=row['adset_name'],
                            platform="meta",
                            metric="ctr",
                            current_value=row['recent_ctr'],
                            previous_value=row['prev_ctr'],
                            change_percent=ctr_change,
                            severity=severity,
                            message=f"CTR dropped: {row['recent_ctr']:.2f}% vs {row['prev_ctr']:.2f}%"
                        ))
                
                # Check CPC (higher is bad)
                if pd.notna(row['recent_cpc']) and pd.notna(row['prev_cpc']) and row['prev_cpc'] > 0:
                    cpc_change = ((row['recent_cpc'] - row['prev_cpc']) / row['prev_cpc']) * 100
                    if cpc_change >= 30:  # CPC increased by 30%+
                        severity = "orange" if cpc_change >= 60 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=row['adset_name'],
                            platform="meta",
                            metric="cpc",
                            current_value=row['recent_cpc'],
                            previous_value=row['prev_cpc'],
                            change_percent=cpc_change,
                            severity=severity,
                            message=f"CPC increased: ${row['recent_cpc']:.2f} vs ${row['prev_cpc']:.2f}"
                        ))
                
                # Check CPM (higher is bad)
                if pd.notna(row['recent_cpm']) and pd.notna(row['prev_cpm']) and row['prev_cpm'] > 0:
                    cpm_change = ((row['recent_cpm'] - row['prev_cpm']) / row['prev_cpm']) * 100
                    if cpm_change >= 40:  # CPM spiked by 40%+
                        severity = "orange" if cpm_change >= 80 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=row['adset_name'],
                            platform="meta",
                            metric="cpm",
                            current_value=row['recent_cpm'],
                            previous_value=row['prev_cpm'],
                            change_percent=cpm_change,
                            severity=severity,
                            message=f"CPM increased: ${row['recent_cpm']:.2f} vs ${row['prev_cpm']:.2f}"
                        ))
                
                # Check Cost per Lead (higher is bad)
                if pd.notna(row['recent_cost_per_lead']) and pd.notna(row['prev_cost_per_lead']) and row['prev_cost_per_lead'] > 0:
                    cpl_change = ((row['recent_cost_per_lead'] - row['prev_cost_per_lead']) / row['prev_cost_per_lead']) * 100
                    if cpl_change >= 40:  # Cost per lead increased by 40%+
                        severity = "orange" if cpl_change >= 70 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=row['adset_name'],
                            platform="meta",
                            metric="cost_per_lead",
                            current_value=row['recent_cost_per_lead'],
                            previous_value=row['prev_cost_per_lead'],
                            change_percent=cpl_change,
                            severity=severity,
                            message=f"Cost per lead increased: ${row['recent_cost_per_lead']:.2f} vs ${row['prev_cost_per_lead']:.2f}"
                        ))
        
        return alerts
        
    except Exception as e:
        return []

def check_google_anomalies() -> List[AnomalyAlert]:
    """Check for anomalies in Google Ads data, focusing on CAC (cost per closing)"""
    
    query = """
    WITH hex_closings AS (
      SELECT 
        date,
        utm_campaign as campaign_name,
        COUNT(*) as closings
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.hex_table`
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
        AND utm_campaign IS NOT NULL
        AND utm_campaign != ''
        AND utm_source = 'google'  -- Only Google campaigns
      GROUP BY date, utm_campaign
    ),
    daily_metrics AS (
      SELECT 
        date,
        campaign_name,
        SUM(spend_usd) as spend,
        SUM(impressions) as impressions,
        SUM(clicks) as clicks,
        SUM(conversions) as conversions,
        COALESCE(h.closings, 0) as closings,
        -- Calculate CAC (Cost per Closing) and other metrics
        SAFE_DIVIDE(SUM(spend_usd), COALESCE(h.closings, 0)) as cost_per_closing,
        SAFE_DIVIDE(SUM(clicks), SUM(impressions)) * 100 as ctr,
        SAFE_DIVIDE(SUM(spend_usd), SUM(clicks)) as cpc,
        SAFE_DIVIDE(SUM(spend_usd), SUM(impressions)) * 1000 as cpm,
        SAFE_DIVIDE(SUM(spend_usd), SUM(conversions)) as cost_per_conversion
      FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data` g
      LEFT JOIN hex_closings h 
        ON g.date = h.date AND g.campaign_name = h.campaign_name
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
      GROUP BY date, campaign_name, h.closings
      HAVING spend > 0  -- Only include campaigns that spent money
    ),
    comparison_data AS (
      SELECT 
        campaign_name,
        -- Most recent day
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN spend END) as recent_spend,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN closings END) as recent_closings,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cost_per_closing END) as recent_cac,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN ctr END) as recent_ctr,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cpc END) as recent_cpc,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cpm END) as recent_cpm,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) THEN cost_per_conversion END) as recent_cost_per_conversion,
        -- Same day last week
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN spend END) as prev_spend,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN closings END) as prev_closings,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cost_per_closing END) as prev_cac,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN ctr END) as prev_ctr,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cpc END) as prev_cpc,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cpm END) as prev_cpm,
        MAX(CASE WHEN date = DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY) THEN cost_per_conversion END) as prev_cost_per_conversion
      FROM daily_metrics
      GROUP BY campaign_name
    )
    SELECT * FROM comparison_data
    WHERE recent_spend IS NOT NULL AND prev_spend IS NOT NULL
    """
    
    try:
        df = bigquery_client.query(query).to_dataframe()
        alerts = []
        
        for _, row in df.iterrows():
            # PRIORITY 1: Check CAC (Cost per Closing) - Most Important
            if pd.notna(row['recent_cac']) and pd.notna(row['prev_cac']) and row['prev_cac'] > 0:
                cac_change = ((row['recent_cac'] - row['prev_cac']) / row['prev_cac']) * 100
                if cac_change >= 30:  # CAC increased by 30%+ (slightly higher threshold for Google)
                    severity = "red" if cac_change >= 85 else "orange" if cac_change >= 60 else "yellow"
                    alerts.append(AnomalyAlert(
                        campaign_name=row['campaign_name'],
                        adset_name=None,
                        platform="google",
                        metric="cost_per_closing",
                        current_value=row['recent_cac'],
                        previous_value=row['prev_cac'],
                        change_percent=cac_change,
                        severity=severity,
                        message=f"CAC spiked: ${row['recent_cac']:.2f} vs ${row['prev_cac']:.2f} (closings: {row['recent_closings']} vs {row['prev_closings']})"
                    ))
            
            # Check for no closings when there were closings before
            elif row['recent_closings'] == 0 and row['prev_closings'] > 0 and row['recent_spend'] > 100:  # Higher spend threshold for Google
                alerts.append(AnomalyAlert(
                    campaign_name=row['campaign_name'],
                    adset_name=None,
                    platform="google",
                    metric="closings",
                    current_value=0,
                    previous_value=row['prev_closings'],
                    change_percent=-100,
                    severity="red",
                    message=f"No closings generated despite ${row['recent_spend']:.2f} spend (had {row['prev_closings']} closings last week)"
                ))
            
            # Check for spending issues (campaign not spending)
            if row['recent_spend'] == 0 and row['prev_spend'] > 25:  # Higher threshold for Google
                alerts.append(AnomalyAlert(
                    campaign_name=row['campaign_name'],
                    adset_name=None,
                    platform="google",
                    metric="spend",
                    current_value=0,
                    previous_value=row['prev_spend'],
                    change_percent=-100,
                    severity="red",
                    message=f"Campaign stopped spending completely (was ${row['prev_spend']:.2f})"
                ))
            
            # LOWER PRIORITY: Other metrics (only if no major CAC issues)
            has_major_cac_issue = (pd.notna(row['recent_cac']) and pd.notna(row['prev_cac']) and 
                                 row['prev_cac'] > 0 and 
                                 ((row['recent_cac'] - row['prev_cac']) / row['prev_cac']) * 100 >= 60)
            
            if not has_major_cac_issue:
                # Check CTR (lower is bad)
                if pd.notna(row['recent_ctr']) and pd.notna(row['prev_ctr']) and row['prev_ctr'] > 0:
                    ctr_change = ((row['recent_ctr'] - row['prev_ctr']) / row['prev_ctr']) * 100
                    if ctr_change <= -25:  # CTR dropped by 25%+
                        severity = "orange" if ctr_change <= -45 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=None,
                            platform="google",
                            metric="ctr",
                            current_value=row['recent_ctr'],
                            previous_value=row['prev_ctr'],
                            change_percent=ctr_change,
                            severity=severity,
                            message=f"CTR dropped: {row['recent_ctr']:.2f}% vs {row['prev_ctr']:.2f}%"
                        ))
                
                # Check CPC (higher is bad)
                if pd.notna(row['recent_cpc']) and pd.notna(row['prev_cpc']) and row['prev_cpc'] > 0:
                    cpc_change = ((row['recent_cpc'] - row['prev_cpc']) / row['prev_cpc']) * 100
                    if cpc_change >= 35:  # CPC increased by 35%+
                        severity = "orange" if cpc_change >= 65 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=None,
                            platform="google",
                            metric="cpc",
                            current_value=row['recent_cpc'],
                            previous_value=row['prev_cpc'],
                            change_percent=cpc_change,
                            severity=severity,
                            message=f"CPC increased: ${row['recent_cpc']:.2f} vs ${row['prev_cpc']:.2f}"
                        ))
                
                # Check CPM (higher is bad)
                if pd.notna(row['recent_cpm']) and pd.notna(row['prev_cpm']) and row['prev_cpm'] > 0:
                    cpm_change = ((row['recent_cpm'] - row['prev_cpm']) / row['prev_cpm']) * 100
                    if cpm_change >= 45:  # CPM spiked by 45%+
                        severity = "orange" if cpm_change >= 90 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=None,
                            platform="google",
                            metric="cpm",
                            current_value=row['recent_cpm'],
                            previous_value=row['prev_cpm'],
                            change_percent=cpm_change,
                            severity=severity,
                            message=f"CPM increased: ${row['recent_cpm']:.2f} vs ${row['prev_cpm']:.2f}"
                        ))
                
                # Check Cost per Conversion (higher is bad)
                if pd.notna(row['recent_cost_per_conversion']) and pd.notna(row['prev_cost_per_conversion']) and row['prev_cost_per_conversion'] > 0:
                    cpc_change = ((row['recent_cost_per_conversion'] - row['prev_cost_per_conversion']) / row['prev_cost_per_conversion']) * 100
                    if cpc_change >= 45:  # Cost per conversion increased by 45%+
                        severity = "orange" if cpc_change >= 80 else "yellow"
                        alerts.append(AnomalyAlert(
                            campaign_name=row['campaign_name'],
                            adset_name=None,
                            platform="google",
                            metric="cost_per_conversion",
                            current_value=row['recent_cost_per_conversion'],
                            previous_value=row['prev_cost_per_conversion'],
                            change_percent=cpc_change,
                            severity=severity,
                            message=f"Cost per conversion increased: ${row['recent_cost_per_conversion']:.2f} vs ${row['prev_cost_per_conversion']:.2f}"
                        ))
        
        return alerts
        
    except Exception as e:
        return []

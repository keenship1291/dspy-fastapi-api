# marketing_analytics.py - CLAUDE TREND ANALYSIS WITH DIRECT FUNNEL DATA INPUT
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import dspy
import os
from datetime import datetime, timedelta, date
import logging
from google.oauth2 import service_account
import json

# BigQuery imports (for spend data)
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

class TrendAnalysisResponse(BaseModel):
    status: str
    trend_summary: str
    colorCode: Optional[str] = None
    formatted_metrics: Optional[Dict[str, Any]] = None
    trend_insights: Optional[Dict[str, Any]] = None
    week_over_week_changes: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Create router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API - Funnel Data Trend Analysis",
        "version": "12.0.0",
        "status": "running",
        "bigquery_available": BIGQUERY_AVAILABLE,
        "input_format": "Accepts funnel_data array with date, leads, start_flows, estimates, closings, funded, rpts"
    }

def get_platform_spend_data():
    """Get platform spend data from BigQuery for comparison"""
    
    if not BIGQUERY_AVAILABLE or not bigquery_client:
        return []
    
    try:
        spend_query = """
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
          COALESCE(m.meta_spend, 0) as meta_spend,
          COALESCE(g.google_spend, 0) as google_spend,
          COALESCE(m.meta_spend, 0) + COALESCE(g.google_spend, 0) as total_spend
        FROM meta_daily m
        FULL OUTER JOIN google_daily g ON m.date = g.date
        ORDER BY date DESC
        """
        
        df = bigquery_client.query(spend_query).to_dataframe()
        
        spend_data = []
        for _, row in df.iterrows():
            spend_data.append({
                'date': str(row['date']),
                'meta_spend': float(row['meta_spend']) if not pd.isna(row['meta_spend']) else 0,
                'google_spend': float(row['google_spend']) if not pd.isna(row['google_spend']) else 0,
                'total_spend': float(row['total_spend']) if not pd.isna(row['total_spend']) else 0
            })
        
        return spend_data
        
    except Exception as e:
        print(f"Error getting spend data: {e}")
        return []

def calculate_week_over_week_changes(funnel_data: List[FunnelDataPoint], spend_data: List[Dict]) -> Dict[str, Any]:
    """Format metrics in structured JSON format like the example"""
    
    def format_value_with_change(current, previous, is_currency=False, is_percentage=False):
        """Format current value with percentage change"""
        if previous == 0:
            change_pct = 100.0 if current > 0 else 0.0
        else:
            change_pct = ((current - previous) / previous) * 100
        
        # Format the current value
        if is_currency:
            if current >= 1000:
                current_formatted = f"${current/1000:,.1f}k"
            else:
                current_formatted = f"${current:,.0f}"
        elif is_percentage:
            current_formatted = f"{current:.1f}%"
        else:
            current_formatted = f"{current:,.0f}"
        
        # Format the change
        change_sign = "+" if change_pct >= 0 else ""
        change_formatted = f"({change_sign}{change_pct:.1f}%)"
        
        return f"{current_formatted} {change_formatted}"
    
    # Extract values from week_changes
    funnel_metrics = {
        "channel": "funnelPerformance",
        "period_type": "weekOverWeekPulse",
        "totalLeads": format_value_with_change(
            week_changes.get('last_7_leads', 0),
            week_changes.get('previous_7_leads', 0)
        ),
        "startFlows": format_value_with_change(
            week_changes.get('last_7_start_flows', 0),
            week_changes.get('previous_7_start_flows', 0)
        ),
        "estimates": format_value_with_change(
            week_changes.get('last_7_estimates', 0),
            week_changes.get('previous_7_estimates', 0)
        ),
        "closings": format_value_with_change(
            week_changes.get('last_7_closings', 0),
            week_changes.get('previous_7_closings', 0)
        ),
        "funded": format_value_with_change(
            week_changes.get('last_7_funded', 0),
            week_changes.get('previous_7_funded', 0)
        ),
        "revenue": format_value_with_change(
            week_changes.get('last_7_rpts', 0),
            week_changes.get('previous_7_rpts', 0),
            is_currency=True
        )
    }
    
    # Add spend metrics from BigQuery if available
    spend_metrics = {}
    if 'last_7_spend' in week_changes and 'previous_7_spend' in week_changes:
        spend_metrics = {
            "channel": "adSpend",
            "period_type": "weekOverWeekPulse", 
            "totalSpend": format_value_with_change(
                week_changes.get('last_7_spend', 0),
                week_changes.get('previous_7_spend', 0),
                is_currency=True
            )
        }
    
    # Calculate conversion rates for current and previous periods
    def safe_divide(numerator, denominator):
        return (numerator / denominator * 100) if denominator > 0 else 0
    
    last_7_lead_to_start = safe_divide(
        week_changes.get('last_7_start_flows', 0),
        week_changes.get('last_7_leads', 0)
    )
    previous_7_lead_to_start = safe_divide(
        week_changes.get('previous_7_start_flows', 0),
        week_changes.get('previous_7_leads', 0)
    )
    
    last_7_start_to_estimate = safe_divide(
        week_changes.get('last_7_estimates', 0),
        week_changes.get('last_7_start_flows', 0)
    )
    previous_7_start_to_estimate = safe_divide(
        week_changes.get('previous_7_estimates', 0),
        week_changes.get('previous_7_start_flows', 0)
    )
    
    last_7_estimate_to_closing = safe_divide(
        week_changes.get('last_7_closings', 0),
        week_changes.get('last_7_estimates', 0)
    )
    previous_7_estimate_to_closing = safe_divide(
        week_changes.get('previous_7_closings', 0),
        week_changes.get('previous_7_estimates', 0)
    )
    
    last_7_closing_to_funded = safe_divide(
        week_changes.get('last_7_funded', 0),
        week_changes.get('last_7_closings', 0)
    )
    previous_7_closing_to_funded = safe_divide(
        week_changes.get('previous_7_funded', 0),
        week_changes.get('previous_7_closings', 0)
    )
    
    conversion_metrics = {
        "channel": "conversionRates",
        "period_type": "weekOverWeekPulse",
        "leadToStartFlowRate": format_value_with_change(
            last_7_lead_to_start,
            previous_7_lead_to_start,
            is_percentage=True
        ),
        "startFlowToEstimateRate": format_value_with_change(
            last_7_start_to_estimate,
            previous_7_start_to_estimate,
            is_percentage=True
        ),
        "estimateToClosingRate": format_value_with_change(
            last_7_estimate_to_closing,
            previous_7_estimate_to_closing,
            is_percentage=True
        ),
        "closingToFundedRate": format_value_with_change(
            last_7_closing_to_funded,
            previous_7_closing_to_funded,
            is_percentage=True
        )
    }
    """Calculate week-over-week changes like the example"""
    
    # Sort by date (most recent first)
    sorted_funnel = sorted(funnel_data, key=lambda x: x.date, reverse=True)
    
    if len(sorted_funnel) < 14:
        return {"error": "Need at least 14 days of data for week-over-week comparison"}
    
    # Split into last 7 days vs previous 7 days
    last_7_days = sorted_funnel[:7]
    previous_7_days = sorted_funnel[7:14]
    
    # Calculate funnel totals
    def sum_funnel_metrics(days):
        return {
            'leads': sum(day.leads for day in days),
            'start_flows': sum(day.start_flows for day in days),
            'estimates': sum(day.estimates for day in days),
            'closings': sum(day.closings for day in days),
            'funded': sum(day.funded for day in days),
            'rpts': sum(day.rpts for day in days)
        }
    
    last_7_totals = sum_funnel_metrics(last_7_days)
    previous_7_totals = sum_funnel_metrics(previous_7_days)
    
    # Calculate spend totals if available
    spend_changes = {}
    if spend_data:
        spend_dict = {item['date']: item for item in spend_data}
        
        last_7_dates = [day.date for day in last_7_days]
        previous_7_dates = [day.date for day in previous_7_days]
        
        last_7_spend = sum(spend_dict.get(date, {}).get('total_spend', 0) for date in last_7_dates)
        previous_7_spend = sum(spend_dict.get(date, {}).get('total_spend', 0) for date in previous_7_dates)
        
        spend_changes = {
            'last_7_spend': last_7_spend,
            'previous_7_spend': previous_7_spend,
            'spend_change': last_7_spend - previous_7_spend
        }
    
    # Calculate changes
    changes = {}
    for metric in ['leads', 'start_flows', 'estimates', 'closings', 'funded', 'rpts']:
        change = last_7_totals[metric] - previous_7_totals[metric]
        changes[f'{metric}_change'] = change
        changes[f'last_7_{metric}'] = last_7_totals[metric]
        changes[f'previous_7_{metric}'] = previous_7_totals[metric]
    
    return {
        **changes,
        **spend_changes,
        'analysis_period': {
            'last_7_start': last_7_days[-1].date,
            'last_7_end': last_7_days[0].date,
            'previous_7_start': previous_7_days[-1].date,
            'previous_7_end': previous_7_days[0].date
        }
    }

def generate_claude_funnel_analysis(funnel_data: List[FunnelDataPoint], week_changes: Dict[str, Any]) -> Dict[str, Any]:
    """Claude analyzes funnel trends and generates insights"""
    
    # Prepare data for Claude
    recent_data = sorted(funnel_data, key=lambda x: x.date, reverse=True)[:7]
    
    analysis_prompt = f"""
    Analyze LeaseEnd marketing funnel performance trends:

    WEEK-OVER-WEEK CHANGES:
    {json.dumps(week_changes, indent=2)}

    RECENT 7 DAYS DAILY BREAKDOWN:
    {json.dumps([{'date': day.date, 'leads': day.leads, 'start_flows': day.start_flows, 'estimates': day.estimates, 'closings': day.closings, 'funded': day.funded, 'rpts': day.rpts} for day in recent_data], indent=2)}

    Provide analysis in this EXACT format:

    {{
        "summary": "Brief trend analysis highlighting key changes like the example format",
        "spend_analysis": "Analysis of spend changes and efficiency",
        "funnel_performance": {{
            "leads_trend": "assessment of lead generation changes",
            "conversion_trends": "analysis of funnel stage performance",
            "bottlenecks": "identification of problem areas",
            "improvements": "areas showing positive trends"
        }},
        "key_insights": [
            "Most significant finding",
            "Secondary insight",
            "Additional observation"
        ],
        "formatted_changes": "Example: ['+$15k spend past 7 days', '+2,333 leads', '+539 start flows', '+499 estimates', '-25 closings', '+39 funded', '+$150k rpts']",
        "priority_actions": [
            "Top recommended action based on trends",
            "Secondary recommendation"
        ]
    }}
    """
    
    try:
        # Use DSPy configured Claude instance
        response = dspy.settings.lm.basic_request(analysis_prompt)
        
        # Clean response
        response_clean = response.strip()
        
        # Remove markdown code blocks
        if '```' in response_clean:
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        # Find JSON content
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            return create_fallback_funnel_analysis(week_changes)
        
        json_content = response_clean[json_start:json_end]
        parsed_analysis = json.loads(json_content)
        
        return parsed_analysis
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return create_fallback_funnel_analysis(week_changes)
    except Exception as e:
        print(f"Analysis error: {e}")
        return create_fallback_funnel_analysis(week_changes)

def create_fallback_funnel_analysis(week_changes: Dict[str, Any]) -> Dict[str, Any]:
    """Create fallback analysis when Claude parsing fails"""
    
    def format_change(value, is_currency=False):
        """Format change values with proper currency/number formatting"""
        if is_currency:
            if abs(value) >= 1000:
                return f"${value/1000:,.0f}k" if value >= 0 else f"-${abs(value)/1000:,.0f}k"
            else:
                return f"${value:,.0f}"
        else:
            return f"{value:,.0f}"
    
    def format_change_with_sign(value, is_currency=False):
        """Add + or - sign to formatted value"""
        formatted = format_change(abs(value), is_currency)
        if value > 0:
            return f"+{formatted}"
        elif value < 0:
            return f"-{formatted}"
        else:
            return formatted
    
    formatted_changes = []
    
    # Spend change (currency)
    if 'spend_change' in week_changes:
        formatted_changes.append(f"{format_change_with_sign(week_changes['spend_change'], True)} spend past 7 days")
    
    # Funnel metrics (non-currency)
    funnel_metrics = [
        ('leads_change', 'leads'),
        ('start_flows_change', 'start flows'),
        ('estimates_change', 'estimates'),
        ('closings_change', 'closings'),
        ('funded_change', 'funded')
    ]
    
    for change_key, display_name in funnel_metrics:
        if change_key in week_changes:
            formatted_changes.append(f"{format_change_with_sign(week_changes[change_key])} {display_name}")
    
    # RPTs (currency)
    if 'rpts_change' in week_changes:
        formatted_changes.append(f"{format_change_with_sign(week_changes['rpts_change'], True)} rpts")
    
    return {
        "summary": "Week-over-week analysis shows mixed funnel performance with some positive and negative trends",
        "spend_analysis": "Spend data analysis limited in fallback mode",
        "funnel_performance": {
            "leads_trend": "Lead generation showing variation",
            "conversion_trends": "Conversion rates fluctuating across funnel stages",
            "bottlenecks": "Analysis system limitations",
            "improvements": "Data collection and processing"
        },
        "key_insights": [
            "Week-over-week comparison completed",
            "Funnel metrics show normal variation",
            "Enhanced analysis requires system improvements"
        ],
        "formatted_changes": formatted_changes,
        "priority_actions": [
            "Review detailed funnel performance by stage",
            "Investigate any significant metric changes"
        ],
        "fallback_mode": True
    }

@marketing_router.post("/funnel-analysis", response_model=TrendAnalysisResponse)
async def analyze_funnel_trends(request: FunnelAnalysisRequest):
    """Analyze funnel data trends with week-over-week comparison"""
    
    try:
        # Get platform spend data
        spend_data = get_platform_spend_data()
        
        # Calculate week-over-week changes
        week_changes = calculate_week_over_week_changes(request.funnel_data, spend_data)
        
        if 'error' in week_changes:
            return TrendAnalysisResponse(
                status="error",
                trend_summary=week_changes['error'],
                error=week_changes['error']
            )
        
        # Determine color code based on performance (after week_changes is calculated)
        color_code = determine_color_code(week_changes)
        
        # Format metrics in structured format
        formatted_metrics = format_metrics_structured(week_changes)
        
        # Generate Claude analysis
        trend_insights = generate_claude_funnel_analysis(request.funnel_data, week_changes)
        
        # Extract trend summary
        trend_summary = trend_insights.get('summary', 'Funnel trend analysis completed')
        
        return TrendAnalysisResponse(
            status="success",
            trend_summary=trend_summary,
            colorCode=color_code,
            formatted_metrics=formatted_metrics,
            trend_insights=trend_insights,
            week_over_week_changes=week_changes
        )
        
    except Exception as e:
        return TrendAnalysisResponse(
            status="error",
            trend_summary="Funnel analysis failed - see error details",
            error=str(e)
        )

@marketing_router.post("/quick-funnel-trends")
async def get_quick_funnel_trends(request: FunnelAnalysisRequest):
    """Get quick formatted changes like the example"""
    
    try:
        # Get spend data
        spend_data = get_platform_spend_data()
        
        # Calculate changes
        week_changes = calculate_week_over_week_changes(request.funnel_data, spend_data)
        
        if 'error' in week_changes:
            return {
                "status": "error",
                "error": week_changes['error']
            }
        
        # Format in structured format
        formatted_metrics = format_metrics_structured(week_changes)
        
        # Format changes like the example
        def format_change(value, is_currency=False):
            """Format change values with proper currency/number formatting"""
            if is_currency:
                if abs(value) >= 1000:
                    return f"${value/1000:,.0f}k" if value >= 0 else f"-${abs(value)/1000:,.0f}k"
                else:
                    return f"${value:,.0f}"
            else:
                return f"{value:,.0f}"
        
        def format_change_with_sign(value, is_currency=False):
            """Add + or - sign to formatted value"""
            formatted = format_change(abs(value), is_currency)
            if value > 0:
                return f"+{formatted}"
            elif value < 0:
                return f"-{formatted}"
            else:
                return formatted
        
        formatted_output = []
        
        # Add spend change if available (currency)
        if 'spend_change' in week_changes:
            formatted_output.append(f"* {format_change_with_sign(week_changes['spend_change'], True)} spend past 7 days")
        
        # Add funnel changes (non-currency)
        funnel_metrics = [
            ('leads_change', 'leads'),
            ('start_flows_change', 'start flows'),
            ('estimates_change', 'estimates'),
            ('closings_change', 'closings'),
            ('funded_change', 'funded')
        ]
        
        for change_key, display_name in funnel_metrics:
            if change_key in week_changes:
                formatted_output.append(f"* {format_change_with_sign(week_changes[change_key])} {display_name}")
        
        # Add RPTs (currency)
        if 'rpts_change' in week_changes:
            formatted_output.append(f"* {format_change_with_sign(week_changes['rpts_change'], True)} rpts")
        
        return {
            "status": "success",
            "formatted_metrics": formatted_metrics,
            "raw_changes": week_changes
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@marketing_router.post("/test-simple")
async def test_simple_endpoint(request: dict):
    """Simple test endpoint to verify routing works"""
    return {
        "status": "success",
        "message": "Marketing router is working",
        "received_data": str(request)[:200] + "..." if len(str(request)) > 200 else str(request)
    }

@marketing_router.get("/test-funnel")
async def test_funnel_analysis():
    """Test endpoint - requires funnel_data to be sent via POST"""
    
    return {
        "status": "info",
        "message": "Use POST /marketing/funnel-analysis with funnel_data in request body",
        "example_request": {
            "funnel_data": [
                {
                    "date": "2025-07-26",
                    "leads": 7535,
                    "start_flows": 491,
                    "estimates": 537,
                    "closings": 66,
                    "funded": 18,
                    "rpts": 46327
                }
            ]
        }
    }

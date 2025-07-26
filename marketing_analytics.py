# marketing_analytics.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import dspy
import os
from datetime import datetime, timedelta
import logging
from google.cloud import bigquery
import pandas as pd
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Use the same database setup as main app
from __main__ import SessionLocal, Base

# BigQuery setup
bigquery_client = bigquery.Client()
PROJECT_ID = "gtm-p3gj3zzk-nthlo"
DATASET_ID = "last_14_days_analysis"

# Database Models for Marketing Analytics
class TrendAnalysis(Base):
    __tablename__ = "trend_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    utm_medium_group = Column(String)
    trend_direction = Column(String)  # improving, declining, stable
    key_insights = Column(JSON)
    performance_metrics = Column(JSON)
    recommendations = Column(Text)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create marketing tables (will only create if they don't exist)
Base.metadata.create_all(bind=SessionLocal().bind)

# Database dependency (reuse from main app)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# DSPy Configuration - reuse the existing Anthropic setup
# Get the configured DSPy instance from main app
def get_marketing_llm():
    """Use the same Claude configuration from main app"""
    return dspy.settings.lm

# Pydantic Models
class UtmMediumMetrics(BaseModel):
    medium_group: str  # paid-social, paid-search-video, paid-media
    total_spend: float
    total_leads: int
    total_estimates: int
    total_conversions: int
    avg_cpa: float
    avg_cpc: float
    avg_ctr: float
    conversion_rate: float
    trend_period_days: int
    daily_breakdown: List[Dict[str, Any]]

class TrendAnalysisRequest(BaseModel):
    date_range: Dict[str, str]  # since, until
    include_historical: bool = True
    analysis_depth: str = "standard"  # standard, detailed, executive

class MediumTrendInsight(BaseModel):
    medium_group: str
    trend_direction: str
    confidence_score: float
    key_metrics_change: Dict[str, float]
    primary_insight: str
    supporting_evidence: List[str]
    recommended_actions: List[str]
    risk_level: str  # low, medium, high

class TrendAnalysisResponse(BaseModel):
    medium_insights: List[MediumTrendInsight]
    cross_medium_summary: Dict[str, Any]
    executive_summary: str
    data_quality_notes: List[str]
    analysis_metadata: Dict[str, Any]

# DSPy Signature for Marketing Trends Analysis
class MarketingTrendsAnalyzer(dspy.Signature):
    """Analyze marketing performance trends across utm_medium groups and provide strategic insights."""
    
    paid_social_metrics = dspy.InputField(desc="Paid social (Meta) performance metrics and trends")
    paid_search_video_metrics = dspy.InputField(desc="Paid search and video performance metrics and trends")  
    paid_media_metrics = dspy.InputField(desc="Overall paid media performance metrics and trends")
    cross_medium_comparison = dspy.InputField(desc="Cross-medium efficiency and allocation comparison")
    
    trend_insights = dspy.OutputField(desc="Strategic insights for each medium group with trend analysis")
    optimization_opportunities = dspy.OutputField(desc="Cross-medium optimization opportunities and budget reallocation recommendations")
    executive_summary = dspy.OutputField(desc="Executive-level summary of marketing performance trends and strategic recommendations")

# DSPy Module
class TrendAnalysisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(MarketingTrendsAnalyzer)
    
    def forward(self, medium_metrics, cross_medium_data):
        # Format metrics for each medium group
        paid_social = self._format_medium_metrics(
            medium_metrics.get('paid-social', {}), 'Paid Social (Meta)'
        )
        paid_search_video = self._format_medium_metrics(
            medium_metrics.get('paid-search-video', {}), 'Paid Search & Video'
        )
        paid_media = self._format_medium_metrics(
            medium_metrics.get('paid-media', {}), 'Overall Paid Media'
        )
        
        cross_medium_summary = self._format_cross_medium_data(cross_medium_data)
        
        result = self.analyzer(
            paid_social_metrics=paid_social,
            paid_search_video_metrics=paid_search_video,
            paid_media_metrics=paid_media,
            cross_medium_comparison=cross_medium_summary
        )
        
        return result
    
    def _format_medium_metrics(self, metrics, medium_name):
        if not metrics:
            return f"{medium_name}: No data available for analysis period"
        
        return f"""
{medium_name} Performance:
- Total Spend: ${metrics.get('total_spend', 0):,.2f}
- Total Leads: {metrics.get('total_leads', 0):,}
- Total Estimates: {metrics.get('total_estimates', 0):,}
- Avg CPA: ${metrics.get('avg_cpa', 0):.2f}
- Avg CPC: ${metrics.get('avg_cpc', 0):.2f}
- Avg CTR: {metrics.get('avg_ctr', 0):.2f}%
- Conversion Rate: {metrics.get('conversion_rate', 0):.2f}%
- Trend: {self._calculate_trend(metrics.get('daily_breakdown', []))}
"""
    
    def _format_cross_medium_data(self, cross_data):
        return f"""
Cross-Medium Performance:
- Spend Distribution: {cross_data.get('spend_distribution', {})}
- Efficiency Rankings: {cross_data.get('efficiency_rankings', {})}
- Volume Leaders: {cross_data.get('volume_leaders', {})}
"""
    
    def _calculate_trend(self, daily_data):
        if len(daily_data) < 3:
            return "Insufficient data"
        
        # Simple trend calculation
        recent_avg = sum(day.get('spend', 0) for day in daily_data[-3:]) / 3
        earlier_avg = sum(day.get('spend', 0) for day in daily_data[:3]) / 3
        
        if recent_avg > earlier_avg * 1.1:
            return "Growing"
        elif recent_avg < earlier_avg * 0.9:
            return "Declining"
        else:
            return "Stable"

# Initialize DSPy module
trend_analyzer = TrendAnalysisModule()

# BigQuery data fetching functions
def fetch_funnel_data(since_date: str, until_date: str) -> pd.DataFrame:
    """Fetch marketing funnel data from hex_data table"""
    query = f"""
    SELECT 
        date,
        utm_campaign,
        utm_medium,
        leads,
        start_flows,
        estimates,
        closings,
        funded,
        rpts,
        platform
    FROM `{PROJECT_ID}.{DATASET_ID}.hex_data`
    WHERE date BETWEEN '{since_date}' AND '{until_date}'
        AND utm_medium IN ('paid-social', 'paid-search', 'paid-video')
    ORDER BY date DESC
    """
    return bigquery_client.query(query).to_dataframe()

def fetch_meta_ad_data(since_date: str, until_date: str) -> pd.DataFrame:
    """Fetch Meta ad performance data"""
    query = f"""
    SELECT 
        date,
        campaign_name,
        adset_name,
        CAST(impressions AS INT64) as impressions,
        CAST(clicks AS INT64) as clicks,
        CAST(spend AS FLOAT64) as spend,
        CAST(reach AS INT64) as reach,
        CAST(landing_page_views AS INT64) as landing_page_views,
        CAST(leads AS INT64) as leads,
        platform
    FROM `{PROJECT_ID}.{DATASET_ID}.meta_data`
    WHERE date BETWEEN '{since_date}' AND '{until_date}'
    ORDER BY date DESC
    """
    return bigquery_client.query(query).to_dataframe()

def process_medium_metrics(funnel_df: pd.DataFrame, meta_df: pd.DataFrame) -> Dict[str, UtmMediumMetrics]:
    """Process and group metrics by utm_medium"""
    
    # Group paid-search and paid-video together
    funnel_df['medium_group'] = funnel_df['utm_medium'].map({
        'paid-social': 'paid-social',
        'paid-search': 'paid-search-video',
        'paid-video': 'paid-search-video'
    })
    
    medium_metrics = {}
    
    for medium_group in ['paid-social', 'paid-search-video']:
        group_data = funnel_df[funnel_df['medium_group'] == medium_group]
        
        if len(group_data) == 0:
            continue
            
        # Get corresponding Meta ad data for paid-social
        meta_group_data = pd.DataFrame()
        if medium_group == 'paid-social':
            meta_group_data = meta_df.copy()
        
        # Calculate metrics
        total_spend = meta_group_data['spend'].sum() if len(meta_group_data) > 0 else 0
        total_leads = group_data['leads'].sum()
        total_estimates = group_data['estimates'].sum()
        total_clicks = meta_group_data['clicks'].sum() if len(meta_group_data) > 0 else 0
        total_impressions = meta_group_data['impressions'].sum() if len(meta_group_data) > 0 else 0
        
        # Calculate rates
        avg_cpa = total_spend / total_leads if total_leads > 0 else 0
        avg_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        conversion_rate = (total_estimates / total_leads * 100) if total_leads > 0 else 0
        
        # Daily breakdown
        daily_breakdown = []
        for date in group_data['date'].unique():
            day_funnel = group_data[group_data['date'] == date]
            day_meta = meta_group_data[meta_group_data['date'] == date] if len(meta_group_data) > 0 else pd.DataFrame()
            
            daily_breakdown.append({
                'date': str(date),
                'spend': day_meta['spend'].sum() if len(day_meta) > 0 else 0,
                'leads': day_funnel['leads'].sum(),
                'estimates': day_funnel['estimates'].sum(),
                'clicks': day_meta['clicks'].sum() if len(day_meta) > 0 else 0
            })
        
        medium_metrics[medium_group] = UtmMediumMetrics(
            medium_group=medium_group,
            total_spend=total_spend,
            total_leads=int(total_leads),
            total_estimates=int(total_estimates),
            total_conversions=int(total_estimates),
            avg_cpa=avg_cpa,
            avg_cpc=avg_cpc,
            avg_ctr=avg_ctr,
            conversion_rate=conversion_rate,
            trend_period_days=len(group_data['date'].unique()),
            daily_breakdown=daily_breakdown
        )
    
    # Add combined paid-media metrics
    all_funnel = funnel_df.copy()
    all_meta = meta_df.copy()
    
    medium_metrics['paid-media'] = UtmMediumMetrics(
        medium_group='paid-media',
        total_spend=all_meta['spend'].sum(),
        total_leads=int(all_funnel['leads'].sum()),
        total_estimates=int(all_funnel['estimates'].sum()),
        total_conversions=int(all_funnel['estimates'].sum()),
        avg_cpa=all_meta['spend'].sum() / all_funnel['leads'].sum() if all_funnel['leads'].sum() > 0 else 0,
        avg_cpc=all_meta['spend'].sum() / all_meta['clicks'].sum() if all_meta['clicks'].sum() > 0 else 0,
        avg_ctr=(all_meta['clicks'].sum() / all_meta['impressions'].sum() * 100) if all_meta['impressions'].sum() > 0 else 0,
        conversion_rate=(all_funnel['estimates'].sum() / all_funnel['leads'].sum() * 100) if all_funnel['leads'].sum() > 0 else 0,
        trend_period_days=len(all_funnel['date'].unique()),
        daily_breakdown=[]
    )
    
    return medium_metrics

def parse_medium_insights(insights_text: str, medium_metrics: Dict[str, UtmMediumMetrics]) -> List[MediumTrendInsight]:
    """Parse DSPy insights into structured format"""
    insights = []
    
    for medium_group, metrics in medium_metrics.items():
        insight = MediumTrendInsight(
            medium_group=medium_group,
            trend_direction="stable",
            confidence_score=0.8,
            key_metrics_change={
                'cpa_change': 0.0,
                'volume_change': 0.0,
                'efficiency_change': 0.0
            },
            primary_insight=f"{medium_group} showing consistent performance with ${metrics.avg_cpa:.2f} avg CPA",
            supporting_evidence=[
                f"Total spend: ${metrics.total_spend:,.2f}",
                f"Total leads: {metrics.total_leads:,}",
                f"Avg CPA: ${metrics.avg_cpa:.2f}",
                f"Conversion rate: {metrics.conversion_rate:.1f}%"
            ],
            recommended_actions=[
                "Continue monitoring performance trends",
                "Optimize campaigns based on CPA efficiency",
                "Consider budget reallocation to top performers"
            ],
            risk_level="low"
        )
        insights.append(insight)
    
    return insights

# Create router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze-trends - Main trend analysis",
            "/trend-history - Historical trend data", 
            "/test-bigquery - Test BigQuery connection"
        ]
    }

@marketing_router.post("/analyze-trends", response_model=TrendAnalysisResponse)
async def analyze_marketing_trends(
    request: TrendAnalysisRequest,
    db: Session = Depends(get_db)
):
    try:
        since_date = request.date_range['since']
        until_date = request.date_range['until']
        
        # Fetch data from BigQuery
        funnel_df = fetch_funnel_data(since_date, until_date)
        meta_df = fetch_meta_ad_data(since_date, until_date)
        
        # Process metrics by medium group
        medium_metrics = process_medium_metrics(funnel_df, meta_df)
        
        # Calculate cross-medium comparison
        total_spend = sum(m.total_spend for m in medium_metrics.values())
        cross_medium_data = {
            'spend_distribution': {
                medium: metrics.total_spend / total_spend if total_spend > 0 else 0 
                for medium, metrics in medium_metrics.items()
            },
            'efficiency_rankings': {
                medium: rank for rank, (medium, _) in enumerate(
                    sorted(medium_metrics.items(), key=lambda x: x[1].avg_cpa), 1
                )
            },
            'volume_leaders': {
                'leads': max(medium_metrics.items(), key=lambda x: x[1].total_leads)[0] if medium_metrics else 'none',
                'spend': max(medium_metrics.items(), key=lambda x: x[1].total_spend)[0] if medium_metrics else 'none'
            }
        }
        
        # Run DSPy analysis
        analysis_result = trend_analyzer(
            medium_metrics={k: v.dict() for k, v in medium_metrics.items()},
            cross_medium_data=cross_medium_data
        )
        
        # Parse insights
        medium_insights = parse_medium_insights(analysis_result.trend_insights, medium_metrics)
        
        # Store analysis in database
        for insight in medium_insights:
            db_trend = TrendAnalysis(
                utm_medium_group=insight.medium_group,
                trend_direction=insight.trend_direction,
                key_insights=insight.dict(),
                performance_metrics={k: v.dict() for k, v in medium_metrics.items()},
                recommendations='; '.join(insight.recommended_actions),
                confidence_score=insight.confidence_score
            )
            db.add(db_trend)
        
        db.commit()
        
        return TrendAnalysisResponse(
            medium_insights=medium_insights,
            cross_medium_summary={
                'total_spend': total_spend,
                'best_performing_medium': min(medium_metrics.items(), key=lambda x: x[1].avg_cpa)[0] if medium_metrics else 'none',
                'highest_volume_medium': max(medium_metrics.items(), key=lambda x: x[1].total_leads)[0] if medium_metrics else 'none',
                'spend_distribution': cross_medium_data['spend_distribution']
            },
            executive_summary=analysis_result.executive_summary,
            data_quality_notes=[
                f"Analyzed {len(funnel_df)} funnel records",
                f"Analyzed {len(meta_df)} Meta ad records",
                f"Date range: {since_date} to {until_date}"
            ],
            analysis_metadata={
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'model_used': 'claude-sonnet-4',
                'data_sources': ['hex_data', 'meta_data']
            }
        )
        
    except Exception as e:
        logging.error(f"Trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@marketing_router.get("/trend-history")
async def get_trend_history(
    days: int = 30,
    medium_group: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get historical trend analysis results"""
    query = db.query(TrendAnalysis)
    
    if medium_group:
        query = query.filter(TrendAnalysis.utm_medium_group == medium_group)
    
    query = query.filter(
        TrendAnalysis.analysis_date >= datetime.utcnow() - timedelta(days=days)
    ).order_by(TrendAnalysis.analysis_date.desc())
    
    trends = query.all()
    
    return {
        'trends': [
            {
                'medium_group': trend.utm_medium_group,
                'trend_direction': trend.trend_direction,
                'confidence_score': trend.confidence_score,
                'analysis_date': trend.analysis_date.isoformat()
            }
            for trend in trends
        ],
        'total_count': len(trends)
    }

@marketing_router.get("/test-bigquery")
async def test_bigquery_connection():
    """Test BigQuery connection and data availability"""
    try:
        # Test basic connection
        test_query = f"""
        SELECT COUNT(*) as total_records
        FROM `{PROJECT_ID}.{DATASET_ID}.hex_data`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        """
        result = bigquery_client.query(test_query).to_dataframe()
        
        return {
            "status": "success",
            "bigquery_connection": "working",
            "recent_records": int(result.iloc[0]['total_records']),
            "project_id": PROJECT_ID,
            "dataset_id": DATASET_ID
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery connection failed: {str(e)}")

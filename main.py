from fastapi import FastAPI, Request, HTTPException
import os
from datetime import datetime, timezone, timedelta
from anthropic import Anthropic
import dspy
from pydantic import BaseModel, field_validator
import json
from typing import List, Dict, Optional, Union
import uuid
import re
import time
from collections import defaultdict

# Database imports
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class FBPost(Base):
    __tablename__ = "fb_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    ad_account_name = Column(String, index=True)
    campaign_name = Column(String, index=True)
    ad_set_name = Column(String)
    ad_name = Column(String)
    page_id = Column(String, index=True)
    post_id = Column(String, unique=True, index=True)
    object_story_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResponseEntry(Base):
    __tablename__ = "responses"
    
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(Text)
    action = Column(String, index=True)
    reply = Column(Text)
    reasoning = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class BatchJob(Base):
    __tablename__ = "batch_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    status = Column(String, default="processing")
    total_comments = Column(Integer, default=0)
    processed_comments = Column(Integer, default=0)
    results = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

# Updated LeaseEnd Response Guidelines
LEASEEND_GUIDELINES = """
COMPANY POSITION - BE CONFIDENT:
- LeaseEnd helps drivers get loans in their name with competitive options, completely online
- We DON'T do third-party financing - we connect you with lenders who offer competitive rates
- Lease buyouts can be smart - but it varies case-by-case based on your specific lease numbers
- We are experts at analyzing the actual numbers to help you make the right decision
- Don't make broad claims about equity - offer to look at their specific situation instead
- Challenge false information politely but firmly, or DELETE if it's accusatory/spam

BRAND VOICE:
- Professional but conversational
- Confident about our expertise in lease analysis
- Concise and relevant arguments - no long explanations
- Focus on analyzing THEIR specific lease numbers
- Emphasize our role as loan facilitators, not lenders
- Direct people to verification rather than generic claims

REFINED CTA GUIDELINES:
- For hesitant customers: Include phone (844) 679-1188
- For contact requests: Include phone (844) 679-1188
- For confused/urgent customers: Include phone (844) 679-1188
- For general prospects: Website CTA only - NO phone
- Phone number format: (844) 679-1188
- Keep CTAs natural and helpful

HANDLING NEGATIVE COMMENTS:
- Comments with accusations of "false info" or "spreading lies" → DELETE
- Comments with excessive numbers/data trying to prove us wrong → DELETE  
- Brief negative comments like "scam", "ripoff", "terrible" → DELETE
- Long argumentative comments questioning our expertise → DELETE
- Focus energy on genuine prospects, not internet arguments

CRITICAL NUMERICAL VALUE RESTRICTIONS:
- NEVER include any dollar amounts, percentages, or specific rates
- Use qualitative terms: "competitive rates", "varies by situation", "depends on your lease"
- Redirect to analysis: "We can look at your specific numbers"
"""

# Lease End Brand Context
BRAND_CONTEXT = {
    "power_words": {
        "trust_building": ["transparent", "guaranteed", "reliable", "honest"],
        "convenience": ["effortless", "hassle-free", "quick", "simple", "online"],
        "value": ["save", "affordable", "competitive", "best deal", "no hidden fees"],
        "urgency": ["limited time", "act now", "lock in", "before rates change"]
    },
    
    "competitive_advantages": {
        "vs_dealerships": "No pressure, transparent pricing, 100% online process",
        "vs_credit_unions": "No membership required, flexible, fast online",
        "vs_banks": "Competitive rates, simple process, customer-centric"
    },
    
    "objection_responses": {
        "time_concerns": "Our online process takes minutes, not hours at a dealership",
        "hidden_fees": "We're completely transparent - no hidden fees, guaranteed",
        "best_deal_doubts": "We work with multiple lenders to get you the best rate",
        "complexity": "We handle all the paperwork - you just sign and we do the rest",
        "dealership_offers": "Dealerships often have hidden fees and pressure tactics"
    }
}

# Custom Anthropic LM for DSPy
class CustomAnthropic(dspy.LM):
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.kwargs = {"max_tokens": 1000}
        self.history = []
        
    def basic_request(self, prompt, **kwargs):
        """Core method that handles the actual API call"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return "Error generating response"
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """Handle different calling patterns DSPy might use"""
        if prompt is None and messages is not None:
            if isinstance(messages, list) and len(messages) > 0:
                prompt = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
            else:
                prompt = str(messages)
        elif prompt is None:
            prompt = ""
            
        result = self.basic_request(prompt, **kwargs)
        return [result]
    
    def generate(self, prompt, **kwargs):
        return self.__call__(prompt, **kwargs)
    
    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

# Configure DSPy
try:
    claude = CustomAnthropic(api_key=ANTHROPIC_API_KEY)
    dspy.settings.configure(lm=claude)
except Exception as e:
    raise ValueError(f"Failed to configure DSPy: {str(e)}")

# Load training data from database
def load_training_data():
    """Load training examples from database"""
    print("🔄 Loading training data from database...")
    
    try:
        db = SessionLocal()
        responses = db.query(ResponseEntry).all()
        db.close()
        
        training_data = []
        for response in responses:
            training_data.append({
                'comment': response.comment,
                'action': response.action,
                'reply': response.reply,
                'reasoning': response.reasoning
            })
        
        print(f"✅ Loaded {len(training_data)} training examples from database")
        return training_data
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return []

# Global training data
TRAINING_DATA = load_training_data()

# Global variables for duplicate prevention
recent_requests = {}
request_tracker = defaultdict(int)
REQUEST_COOLDOWN = 5  # seconds

def reload_training_data():
    """Reload training data from database"""
    global TRAINING_DATA
    TRAINING_DATA = load_training_data()
    return len(TRAINING_DATA)

def filter_numerical_values(text):
    """Remove any numerical values (dollars, percentages, rates) from response text"""
    
    # Remove dollar amounts ($X, $X.XX, $X,XXX, etc.)
    text = re.sub(r'\$[\d,]+(?:\.\d{2})?', '', text)
    
    # Remove percentages (X%, X.X%, XX.XX%, etc.)
    text = re.sub(r'\b\d+(?:\.\d+)?%', '', text)
    
    # Remove APR/interest rate patterns (X.X% APR, X% interest, etc.)
    text = re.sub(r'\b\d+(?:\.\d+)?\s*%?\s*(?:APR|apr|interest|rate)', '', text)
    
    # Remove standalone numbers that might be rates (like "4.5" or "6.2")
    text = re.sub(r'\b\d+\.\d+\b(?=\s*(?:rate|APR|interest|%|percent))', '', text)
    
    # Clean up any double spaces or awkward spacing created by removals
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
    
    return text.strip()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def classify_comment_with_ai(comment, commentId=""):
    """Enhanced comment classification with refined phone logic"""
    
    prompt = f"""You are analyzing comments for LeaseEnd.com, which helps drivers get loans for lease buyouts.

BUSINESS LOGIC & COMPANY POSITION:
- LeaseEnd helps drivers get loans in their name with competitive options, completely online
- We DON'T do third-party financing - we connect customers with lenders
- Lease buyouts can be smart, but it varies case-by-case based on specific lease numbers
- We analyze the actual numbers to help customers make informed decisions
- Don't argue with trolls - delete accusatory or spam comments
- Focus on genuine prospects who want help with their specific situation

ENHANCED DELETE CRITERIA:
- Accusations of spreading false information or lies → DELETE
- Comments with excessive numbers/data trying to prove us wrong → DELETE
- Argumentative comments questioning our expertise with hostility → DELETE
- Spam, inappropriate, or clearly non-prospects → DELETE
- Brief negative comments: "scam", "ripoff", "terrible", "fraud" → DELETE
- Long rants about leasing being bad with no genuine question → DELETE

TAGGING DETECTION LOGIC:
- Tagged comments (sharing with friends) → LEAVE ALONE unless very negative toward us
- Very negative tagged comments about LeaseEnd → DELETE

REFINED PHONE NUMBER USAGE - ONLY flag needs_phone=true for:
- Customer explicitly requests contact: "call me", "speak to someone", "phone number"
- Customer shows hesitation: "not sure", "worried", "skeptical", "what's the catch"
- Customer is confused: "don't understand", "complicated", "how does this work"
- Customer indicates urgency: "urgent", "asap", "time sensitive"
- DO NOT flag general interest: "interested", "looking", "how much", "can I qualify"

ACTIONS:
- REPLY: For genuine questions, prospects, positive feedback, correctable misinformation
- REACT: For positive comments that don't need responses
- DELETE: For accusations, spam, hostility, excessive arguing, brief negative comments
- LEAVE_ALONE: For harmless off-topic or neutral tagged comments

COMMENT: "{comment}"

Respond in this JSON format: {{"sentiment": "...", "action": "...", "reasoning": "...", "high_intent": true/false, "needs_phone": true/false}}"""

    try:
        response = claude.basic_request(prompt)
        response_clean = response.strip()
        
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        try:
            result = json.loads(response_clean)
            return {
                'sentiment': result.get('sentiment', 'Neutral'),
                'action': result.get('action', 'LEAVE_ALONE'),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'high_intent': result.get('high_intent', False),
                'needs_phone': result.get('needs_phone', False)
            }
        except json.JSONDecodeError:
            # Enhanced fallback parsing
            if any(word in response.upper() for word in ['DELETE', 'SCAM', 'FALSE INFO', 'LIES', 'FRAUD']):
                action = 'DELETE'
                reasoning = "Detected negative/accusatory content"
            elif 'REPLY' in response.upper():
                action = 'REPLY'
                reasoning = "Detected genuine prospect or question"
            else:
                action = 'LEAVE_ALONE'
                reasoning = "Fallback classification"
                
            return {
                'sentiment': 'Neutral',
                'action': action,
                'reasoning': reasoning,
                'high_intent': False,
                'needs_phone': False
            }
            
    except Exception as e:
        return {
            'sentiment': 'Neutral',
            'action': 'LEAVE_ALONE',
            'reasoning': f'Classification error: {str(e)}',
            'high_intent': False,
            'needs_phone': False
        }

def generate_response(comment, sentiment, high_intent=False, needs_phone=False):
    """Enhanced response generation with refined phone number usage"""
    
    # General customer indicators (no phone needed)
    customer_indicators = [
        "how much", "what are", "can i", "should i", "interested", "looking", 
        "want to", "need", "help me", "my lease", "my car", "rates", "process",
        "qualify", "apply", "cost", "price", "how do", "when can", "where do"
    ]
    
    # SPECIFIC hesitation/contact request indicators (phone needed)
    hesitation_indicators = [
        "not sure about", "hesitant", "worried", "concerned", "skeptical",
        "don't trust", "seems too good", "what's the catch", "suspicious"
    ]
    
    contact_request_indicators = [
        "call me", "speak to someone", "talk to a person", "phone number",
        "contact you", "reach out", "give me a call", "someone call me"
    ]
    
    confusion_indicators = [
        "confused", "don't understand", "complicated", "explain", 
        "how does this work", "i'm lost", "need help understanding"
    ]
    
    urgent_indicators = [
        "urgent", "asap", "need help now", "time sensitive", "deadline"
    ]
    
    positive_feedback_indicators = [
        "thank you", "thanks", "great service", "amazing", "fantastic", "love", 
        "excellent", "helped me", "saved me", "recommend"
    ]
    
    misinformation_indicators = [
        "lease buyouts are bad", "never buy your lease", "always return", "terrible idea",
        "waste of money", "financial mistake", "bad deal"
    ]
    
    # Check comment characteristics
    is_potential_customer = any(indicator in comment.lower() for indicator in customer_indicators)
    shows_hesitation = any(indicator in comment.lower() for indicator in hesitation_indicators)
    requests_contact = any(indicator in comment.lower() for indicator in contact_request_indicators)
    is_confused = any(indicator in comment.lower() for indicator in confusion_indicators)
    is_urgent = any(indicator in comment.lower() for indicator in urgent_indicators)
    is_positive_feedback = any(indicator in comment.lower() for indicator in positive_feedback_indicators)
    needs_correction = any(indicator in comment.lower() for indicator in misinformation_indicators)
    
    # REFINED Phone number logic - only for specific cases
    phone_instruction = ""
    if requests_contact or (shows_hesitation and is_potential_customer):
        phone_instruction = "\nCustomer is requesting contact or showing hesitation. Include: 'Feel free to give us a call at (844) 679-1188 if you'd prefer to speak with someone.'"
    elif is_confused or is_urgent:
        phone_instruction = "\nCustomer seems confused or urgent. Include: 'Call (844) 679-1188 if you need immediate help.'"
    elif needs_phone:  # Only if explicitly flagged by AI
        phone_instruction = "\nAI flagged this customer needs phone support. Include: 'You can reach us at (844) 679-1188 for personalized help.'"
    
    # For general customers - NO phone, just website CTA
    general_customer_instruction = ""
    if is_potential_customer and not (requests_contact or shows_hesitation or is_confused or is_urgent or needs_phone):
        general_customer_instruction = "\nFor this potential customer, use soft website CTA: 'Check out our site to see your options' or 'Visit our website to get started' - NO phone number."
    
    # Special instructions for different comment types
    positive_feedback_instruction = ""
    if is_positive_feedback:
        positive_feedback_instruction = "\nBrief appreciation: 'Thank you!', 'Glad we could help!', 'Enjoy your ride!' etc. Keep it short and genuine."
    
    correction_instruction = ""
    if needs_correction:
        correction_instruction = """
This comment has misinformation. Correct it concisely:
- Don't make broad equity claims - say it varies case-by-case
- Offer to analyze their specific lease numbers
- Be confident but not argumentative
- Keep response short and focused on helping them verify actual numbers"""

    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com.

COMPANY POSITION:
- LeaseEnd helps drivers get loans in their name with competitive options, completely online
- We DON'T do third-party financing - we connect customers with lenders
- Lease buyouts can be smart, but it varies case-by-case based on specific numbers
- We analyze actual lease numbers to help customers decide
- Keep arguments concise and relevant - offer verification over broad claims

COMMENT: "{comment}"
SENTIMENT: {sentiment}
HIGH INTENT: {high_intent}
NEEDS PHONE: {needs_phone}

RESPONSE GUIDELINES:
- Sound natural and conversational
- Keep responses concise (1-2 sentences usually)
- Don't make broad claims about equity - offer to look at their specific situation
- Focus on case-by-case analysis rather than generic arguments
- ONLY use phone number (844) 679-1188 for hesitation/contact requests/confusion
- For general prospects, use website CTAs instead
- No dollar amounts, percentages, or specific rates
- Address their specific concern directly

BUSINESS MESSAGING:
- "It varies case-by-case based on your specific lease"
- "We can look at your actual numbers to help you decide"
- "We help you get a loan in your name with competitive options"
- "Every situation is different - let's analyze yours"

{positive_feedback_instruction}
{correction_instruction}
{general_customer_instruction}
{phone_instruction}

Generate a helpful, natural response that's concise and relevant:"""

    try:
        response = claude.basic_request(prompt)
        cleaned_response = filter_numerical_values(response.strip())
        
        # Ensure phone number format is correct if present
        if '844' in cleaned_response and '679-1188' in cleaned_response:
            cleaned_response = re.sub(r'\(?844\)?[-.\s]*679[-.\s]*1188', '(844) 679-1188', cleaned_response)
        
        return cleaned_response
    except Exception as e:
        return "Thank you for your comment! We'd be happy to help analyze your specific lease situation."

# Safe Pydantic Models with V2 field_validator
class CommentRequest(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: Optional[str] = None
    postId: Optional[str] = None
    created_time: Optional[str] = ""
    memory_context: Optional[str] = ""
    
    @field_validator('comment', mode='before')
    @classmethod
    def convert_comment_to_string(cls, v):
        if v is None:
            return None
        try:
            return str(v)
        except:
            return None
    
    @field_validator('message', mode='before')
    @classmethod
    def convert_message_to_string(cls, v):
        if v is None:
            return None
        try:
            return str(v)
        except:
            return None
    
    def get_comment_text(self) -> str:
        try:
            comment_str = str(self.comment) if self.comment is not None else ""
            message_str = str(self.message) if self.message is not None else ""
            return (comment_str or message_str or "No message content").strip()
        except:
            return "No message content"
    
    def get_comment_id(self) -> str:
        try:
            return str(self.commentId or self.postId or "unknown").strip()
        except:
            return "unknown"

class Comment(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: str
    created_time: Optional[str] = ""
    
    def get_comment_text(self) -> str:
        try:
            return str(self.comment or self.message or "No message content").strip()
        except:
            return "No message content"

class BatchCommentRequest(BaseModel):
    comments: List[Comment]
    batch_id: Optional[str] = None

class ProcessedComment(BaseModel):
    commentId: str
    original_comment: str
    category: str
    action: str
    reply: str
    confidence_score: float
    approved: str
    reasoning: str

class FeedbackRequest(BaseModel):
    original_comment: str
    original_response: str = ""
    original_action: str
    feedback_text: str
    commentId: str
    current_version: str = ""
    
    @field_validator('original_comment', mode='before')
    @classmethod
    def convert_comment_to_string(cls, v):
        if v is None:
            return ""
        try:
            return str(v)
        except:
            return "Error converting comment"
    
    @field_validator('original_response', mode='before')
    @classmethod
    def convert_response_to_string(cls, v):
        if v is None:
            return ""
        try:
            return str(v)
        except:
            return ""
    
    @field_validator('feedback_text', mode='before')
    @classmethod
    def convert_feedback_to_string(cls, v):
        if v is None:
            return ""
        try:
            return str(v)
        except:
            return "Error converting feedback"
    
    @field_validator('current_version', mode='before')
    @classmethod
    def convert_version_to_string(cls, v):
        if v is None:
            return ""
        try:
            return str(v)
        except:
            return ""
    
    class Config:
        extra = "ignore"

class ApproveRequest(BaseModel):
    original_comment: str
    action: str
    reply: str
    reasoning: str = ""
    created_time: Optional[str] = ""
    
    @field_validator('original_comment', mode='before')
    @classmethod
    def convert_comment_to_string(cls, v):
        if v is None:
            return ""
        try:
            return str(v)
        except:
            return "Error converting comment"
    
    @field_validator('reply', mode='before')
    @classmethod
    def convert_reply_to_string(cls, v):
        if v is None:
            return ""
        try:
            return str(v)
        except:
            return ""
    
    class Config:
        extra = "ignore"

class FBPostCreate(BaseModel):
    ad_account_name: str
    campaign_name: str
    ad_set_name: str
    ad_name: str
    page_id: str
    post_id: str
    object_story_id: str

class ResponseCreate(BaseModel):
    comment: str
    action: str
    reply: str
    reasoning: Optional[str] = ""

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - SAFE VERSION",
        "version": "31.0-SAFE-DUPLICATE-PREVENTION",
        "training_examples": len(TRAINING_DATA),
        "status": "RUNNING",
        "features": ["Safe Type Conversion", "Duplicate Prevention", "Pydantic V2 Compatibility"],
        "key_changes": [
            "SAFE: Comprehensive error handling and type conversion",
            "Duplicate request prevention with 5-second cooldown",
            "Pydantic V2 field_validator compatibility",
            "Enhanced logging and debugging capabilities"
        ]
    }

@app.get("/ping")
@app.post("/ping") 
@app.head("/ping")
def ping():
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "App is running"
    }

@app.get("/health")
def health_check():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "training_examples": len(TRAINING_DATA)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "error",
            "error": str(e)
        }

@app.get("/debug-requests")
async def debug_requests():
    """Get request tracking info"""
    try:
        return {
            "total_unique_comments": len(request_tracker),
            "request_counts": dict(request_tracker),
            "duplicate_comments": {k: v for k, v in request_tracker.items() if v > 1},
            "active_cooldowns": len(recent_requests),
            "cooldown_seconds": REQUEST_COOLDOWN,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.get("/fb-posts")
async def get_fb_posts():
    """Get all FB posts from database"""
    db = SessionLocal()
    try:
        posts = db.query(FBPost).all()
        
        return {
            "success": True,
            "count": len(posts),
            "data": [
                {
                    "Ad account name": post.ad_account_name,
                    "Campaign name": post.campaign_name,
                    "Ad set name": post.ad_set_name,
                    "Ad name": post.ad_name,
                    "Page ID": post.page_id,
                    "Post Id": post.post_id,
                    "Object Story ID": post.object_story_id
                }
                for post in posts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/fb-posts/add")
async def add_fb_post(post: FBPostCreate):
    """Add new FB post to database"""
    try:
        db = SessionLocal()
        
        db_post = FBPost(
            ad_account_name=post.ad_account_name,
            campaign_name=post.campaign_name,
            ad_set_name=post.ad_set_name,
            ad_name=post.ad_name,
            page_id=post.page_id,
            post_id=post.post_id,
            object_story_id=post.object_story_id
        )
        
        db.add(db_post)
        db.commit()
        db.close()
        
        return {
            "success": True,
            "message": "FB post added successfully",
            "post_id": post.post_id,
            "status": "new"
        }
        
    except IntegrityError:
        db.rollback()
        db.close()
        return {
            "success": True,
            "message": f"Post {post.post_id} already exists (duplicate skipped)",
            "post_id": post.post_id,
            "status": "duplicate_skipped"
        }
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/fb-posts/clear-recent")
async def clear_recent_posts():
    """Delete FB posts created in the past 24 hours"""
    try:
        db = SessionLocal()
        
        # Calculate 24 hours ago
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        
        # Find posts created in the past 24 hours
        recent_posts = db.query(FBPost).filter(FBPost.created_at >= twenty_four_hours_ago).all()
        deleted_count = len(recent_posts)
        
        # Delete them
        db.query(FBPost).filter(FBPost.created_at >= twenty_four_hours_ago).delete()
        db.commit()
        db.close()
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} posts created in the past 24 hours",
            "deleted_count": deleted_count,
            "cutoff_time": twenty_four_hours_ago.isoformat()
        }
        
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/responses")
async def get_responses():
    """Get all response training data"""
    db = SessionLocal()
    try:
        responses = db.query(ResponseEntry).all()
        
        return {
            "success": True,
            "count": len(responses),
            "data": [
                {
                    "comment": resp.comment,
                    "action": resp.action,
                    "reply": resp.reply
                }
                for resp in responses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/responses/add")
async def add_response(response: ResponseCreate):
    """Add new response training data"""
    db = SessionLocal()
    try:
        db_response = ResponseEntry(
            comment=response.comment,
            action=response.action,
            reply=response.reply,
            reasoning=response.reasoning
        )
        
        db.add(db_response)
        db.commit()
        
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Response added successfully",
            "new_training_count": new_count
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/reload-training-data")
async def reload_training_data_endpoint():
    """Manually reload training data from database"""
    try:
        new_count = reload_training_data()
        return {
            "success": True,
            "message": "Training data reloaded from database",
            "total_examples": new_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading training data: {str(e)}")

@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Safe feedback processing with duplicate prevention and error handling"""
    start_time = time.time()
    
    try:
        # Safely extract request data
        try:
            comment_id = str(request.commentId) if request.commentId else "unknown"
            original_comment = str(request.original_comment) if request.original_comment else ""
            original_response = str(request.original_response) if request.original_response else ""
            original_action = str(request.original_action).strip().lower() if request.original_action else "unknown"
            feedback_text = str(request.feedback_text) if request.feedback_text else ""
            current_version = str(request.current_version) if request.current_version else ""
        except Exception as e:
            print(f"❌ Error extracting request data: {e}")
            return {
                "commentId": "error",
                "original_comment": "Error extracting data",
                "category": "error",
                "action": "leave_alone",
                "reply": "Error processing request",
                "confidence_score": 0.0,
                "approved": "pending",
                "feedback_text": "Error",
                "version": "v1",
                "reasoning": f"Request data extraction error: {str(e)}",
                "success": False,
                "processing_time": round(time.time() - start_time, 3),
                "method": "extraction_error"
            }
        
        # Create duplicate prevention key
        request_key = f"{comment_id}_{feedback_text[:20] if feedback_text else 'empty'}"
        current_time = time.time()
        
        # Check for duplicates
        if request_key in recent_requests:
            last_time = recent_requests[request_key]
            if current_time - last_time < REQUEST_COOLDOWN:
                print(f"⚠️ DUPLICATE REQUEST BLOCKED: {request_key}")
                return {
                    "commentId": comment_id,
                    "original_comment": original_comment,
                    "category": "duplicate",
                    "action": "leave_alone",
                    "reply": "Duplicate request blocked",
                    "confidence_score": 0.0,
                    "approved": "pending",
                    "feedback_text": feedback_text,
                    "version": "v1",
                    "reasoning": "Blocked duplicate request within cooldown period",
                    "success": True,
                    "processing_time": round(time.time() - start_time, 3),
                    "method": "duplicate_blocked"
                }
        
        # Record this request
        recent_requests[request_key] = current_time
        request_tracker[comment_id] += 1
        
        # Clean old entries
        if len(recent_requests) > 100:
            old_keys = sorted(recent_requests.keys(), 
                             key=lambda k: recent_requests[k])[:-50]
            for key in old_keys:
                del recent_requests[key]
        
        print(f"✅ Processing feedback #{request_tracker[comment_id]} for: {comment_id}")
        print(f"   Feedback: {feedback_text[:50]}...")
        
        # FAST PATH: Handle simple feedback without AI
        simple_feedback_patterns = {
            "leave alone": {"action": "leave_alone", "reply": "", "reasoning": "User requested to leave alone"},
            "delete": {"action": "delete", "reply": "", "reasoning": "User requested deletion"},
            "ignore": {"action": "leave_alone", "reply": "", "reasoning": "User requested to ignore"},
            "skip": {"action": "leave_alone", "reply": "", "reasoning": "User requested to skip"},
            "good": {"action": "respond", "reply": original_response, "reasoning": "User approved response"},
            "ok": {"action": "respond", "reply": original_response, "reasoning": "User approved response"},
            "approved": {"action": "respond", "reply": original_response, "reasoning": "User approved response"},
        }
        
        feedback_lower = feedback_text.lower() if feedback_text else ""
        for pattern, response_data in simple_feedback_patterns.items():
            if pattern in feedback_lower:
                print(f"⚡ FAST PATH: '{pattern}' detected")
                
                # Handle version increment safely
                current_version_num = 1
                if current_version and current_version.startswith('v'):
                    try:
                        current_version_num = int(current_version.replace('v', ''))
                    except (ValueError, AttributeError):
                        current_version_num = 1
                
                return {
                    "commentId": comment_id,
                    "original_comment": original_comment,
                    "category": "neutral",
                    "action": response_data["action"],
                    "reply": response_data["reply"],
                    "confidence_score": 0.9,
                    "approved": "pending",
                    "feedback_text": feedback_text,
                    "version": f"v{current_version_num + 1}",
                    "reasoning": response_data["reasoning"],
                    "feedback_processed": True,
                    "success": True,
                    "processing_time": round(time.time() - start_time, 3),
                    "method": "fast_path"
                }
        
        # SLOW PATH: Use Claude API
        print(f"🤖 Using Claude API for complex feedback...")
        
        # Create prompt safely
        try:
            feedback_prompt = f"""You are improving a response based on human feedback for LeaseEnd.com.

ORIGINAL COMMENT: "{original_comment[:200]}..."
YOUR ORIGINAL RESPONSE: "{original_response[:200]}..."
YOUR ORIGINAL ACTION: "{original_action}"
HUMAN FEEDBACK: "{feedback_text[:100]}..."

GUIDELINES:
- LeaseEnd helps drivers get loans in their name, completely online
- Lease buyouts vary case-by-case - analyze specific numbers
- Use (844) 679-1188 ONLY for hesitant/confused/contact-requesting customers
- For general prospects, use website CTAs instead

Apply the feedback and respond with JSON only:
{{"action": "respond", "reply": "improved response here", "reasoning": "brief explanation"}}"""
        except Exception as e:
            print(f"❌ Error creating prompt: {e}")
            feedback_prompt = 'Improve the response based on feedback. Respond with JSON: {"action": "respond", "reply": "We can help analyze your specific lease situation.", "reasoning": "Fallback response"}'
        
        try:
            # Call Claude API
            improved_response = claude.basic_request(feedback_prompt)
            
            # Parse response safely
            response_clean = improved_response.strip()
            if response_clean.startswith('```'):
                response_clean = response_clean[3:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            # Find JSON
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                response_clean = response_clean[json_start:json_end]
            
            # Parse JSON
            result = json.loads(response_clean)
            
            # Extract values safely
            improved_action = str(result.get('action', 'leave_alone')).lower()
            if improved_action == 'reply':
                improved_action = 'respond'
            
            improved_reply = str(result.get('reply', 'We can help analyze your specific lease situation.'))
            improved_reply = filter_numerical_values(improved_reply)
            
            # Fix phone number format
            if '844' in improved_reply and '679-1188' in improved_reply:
                improved_reply = re.sub(r'\(?844\)?[-.\s]*679[-.\s]*1188', '(844) 679-1188', improved_reply)
            
            # Handle version increment
            current_version_num = 1
            if current_version and current_version.startswith('v'):
                try:
                    current_version_num = int(current_version.replace('v', ''))
                except (ValueError, AttributeError):
                    current_version_num = 1
            
            processing_time = round(time.time() - start_time, 3)
            print(f"✅ Completed successfully in {processing_time}s")
            
            return {
                "commentId": comment_id,
                "original_comment": original_comment,
                "category": "neutral",
                "action": improved_action,
                "reply": improved_reply,
                "confidence_score": 0.85,
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": f"v{current_version_num + 1}",
                "reasoning": str(result.get('reasoning', 'Applied feedback')),
                "feedback_processed": True,
                "success": True,
                "processing_time": processing_time,
                "method": "ai_processing"
            }
            
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON parsing error: {json_error}")
            
            # Handle version safely
            current_version_num = 1
            if current_version and current_version.startswith('v'):
                try:
                    current_version_num = int(current_version.replace('v', ''))
                except (ValueError, AttributeError):
                    current_version_num = 1
            
            return {
                "commentId": comment_id,
                "original_comment": original_comment,
                "category": "neutral",
                "action": "leave_alone",
                "reply": "We can help analyze your specific lease situation.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": f"v{current_version_num + 1}",
                "reasoning": "JSON parsing fallback",
                "success": False,
                "error": f"JSON parsing failed: {str(json_error)}",
                "processing_time": round(time.time() - start_time, 3),
                "method": "json_fallback"
            }
            
        except Exception as api_error:
            print(f"❌ Claude API error: {api_error}")
            
            # Handle version safely
            current_version_num = 1
            if current_version and current_version.startswith('v'):
                try:
                    current_version_num = int(current_version.replace('v', ''))
                except (ValueError, AttributeError):
                    current_version_num = 1
            
            return {
                "commentId": comment_id,
                "original_comment": original_comment,
                "category": "error",
                "action": "leave_alone",
                "reply": "We can help analyze your specific lease situation.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": f"v{current_version_num + 1}",
                "reasoning": "API error fallback",
                "success": False,
                "error": f"API error: {str(api_error)}",
                "processing_time": round(time.time() - start_time, 3),
                "method": "api_error_fallback"
            }
            
    except Exception as general_error:
        print(f"❌ GENERAL ERROR: {general_error}")
        
        # Ultimate fallback
        return {
            "commentId": "error",
            "original_comment": "General error occurred",
            "category": "error", 
            "action": "leave_alone",
            "reply": "We can help analyze your specific lease situation.",
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": "Error",
            "version": "v1",
            "reasoning": "General error fallback",
            "error": str(general_error),
            "success": False,
            "processing_time": round(time.time() - start_time, 3),
            "method": "general_error_fallback"
        }

@app.post("/process-batch")
async def process_batch(request: BatchCommentRequest):
    """Process multiple comments in a batch with enhanced logic"""
    try:
        job_id = str(uuid.uuid4())
        
        db = SessionLocal()
        batch_job = BatchJob(
            job_id=job_id,
            status="processing",
            total_comments=len(request.comments),
            processed_comments=0
        )
        db.add(batch_job)
        db.commit()
        
        results = []
        processed_count = 0
        
        for comment_data in request.comments:
            try:
                comment_text = comment_data.get_comment_text()
                comment_id = comment_data.commentId
                
                if not comment_text or comment_text == "No message content":
                    result = {
                        "commentId": comment_id,
                        "original_comment": "Empty comment",
                        "category": "neutral",
                        "action": "delete",
                        "reply": "",
                        "confidence_score": 0.0,
                        "approved": "pending"
                    }
                else:
                    ai_classification = classify_comment_with_ai(comment_text, comment_id)
                    
                    sentiment = ai_classification['sentiment']
                    action = ai_classification['action'].lower()
                    reasoning = ai_classification['reasoning']
                    high_intent = ai_classification['high_intent']
                    needs_phone = ai_classification['needs_phone']
                    
                    action_mapping = {
                        'reply': 'respond',
                        'react': 'react', 
                        'delete': 'delete',
                        'leave_alone': 'leave_alone'
                    }
                    
                    mapped_action = action_mapping.get(action, 'leave_alone')
                    
                    reply_text = ""
                    confidence_score = 0.85
                    
                    if mapped_action == 'respond':
                        reply_text = generate_response(comment_text, sentiment, high_intent, needs_phone)
                        confidence_score = 0.9
                    
                    result = {
                        "commentId": comment_id,
                        "original_comment": comment_text,
                        "category": sentiment.lower(),
                        "action": mapped_action,
                        "reply": reply_text,
                        "confidence_score": confidence_score,
                        "approved": "pending",
                        "success": True,
                        "reasoning": reasoning
                    }
                
                results.append(result)
                processed_count += 1
                
                batch_job.processed_comments = processed_count
                db.commit()
                
            except Exception as comment_error:
                result = {
                    "commentId": comment_data.commentId,
                    "original_comment": "Error processing",
                    "category": "error",
                    "action": "leave_alone",
                    "reply": "We can help analyze your specific lease situation.",
                    "confidence_score": 0.0,
                    "approved": "pending",
                    "success": False,
                    "error": str(comment_error)
                }
                results.append(result)
                processed_count += 1
        
        batch_job.status = "completed"
        batch_job.completed_at = datetime.utcnow()
        batch_job.results = json.dumps(results)
        db.commit()
        db.close()
        
        return {
            "job_id": job_id,
            "status": "completed",
            "total_comments": len(request.comments),
            "processed_comments": processed_count,
            "results": results
        }
        
    except Exception as e:
        try:
            batch_job.status = "failed"
            batch_job.results = json.dumps({"error": str(e)})
            db.commit()
            db.close()
        except:
            pass
            
        return {
            "job_id": job_id if 'job_id' in locals() else "unknown",
            "status": "failed",
            "total_comments": len(request.comments) if request.comments else 0,
            "processed_comments": 0,
            "error": str(e)
        }

@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Enhanced comment processing with updated business logic"""
    try:
        comment_text = request.get_comment_text()
        comment_id = request.get_comment_id()
        
        # Enhanced classification with new logic
        ai_classification = classify_comment_with_ai(comment_text, comment_id)
        
        sentiment = ai_classification['sentiment']
        action = ai_classification['action'].lower()
        reasoning = ai_classification['reasoning']
        high_intent = ai_classification['high_intent']
        needs_phone = ai_classification['needs_phone']
        
        # Map actions to our system
        action_mapping = {
            'reply': 'respond',
            'react': 'react', 
            'delete': 'delete',
            'leave_alone': 'leave_alone'
        }
        
        mapped_action = action_mapping.get(action, 'leave_alone')
        
        # Generate response with enhanced logic
        reply_text = ""
        confidence_score = 0.85
        
        if mapped_action == 'respond':
            reply_text = generate_response(comment_text, sentiment, high_intent, needs_phone)
            confidence_score = 0.9
        
        return ProcessedComment(
            commentId=comment_id,
            original_comment=comment_text,
            category=sentiment.lower(),
            action=mapped_action,
            reply=reply_text,
            confidence_score=confidence_score,
            approved="pending",
            reasoning=reasoning
        )
        
    except Exception as e:
        return ProcessedComment(
            commentId=request.get_comment_id(),
            original_comment=request.get_comment_text(),
            category="error",
            action="leave_alone",
            reply="We can help analyze your specific lease situation.",
            confidence_score=0.0,
            approved="pending",
            reasoning=f"Error: {str(e)}"
        )

@app.post("/approve-response")
async def approve_response(request: ApproveRequest):
    """Approve and store a response for training"""
    global TRAINING_DATA
    
    db = SessionLocal()
    try:
        # Check if comment already exists
        existing_comment = db.query(ResponseEntry).filter(
            ResponseEntry.comment == str(request.original_comment)
        ).first()
        
        if existing_comment:
            return {
                "status": "duplicate_skipped",
                "message": f"Comment already exists in training data (ID: {existing_comment.id})",
                "training_examples": len(TRAINING_DATA),
                "existing_action": existing_comment.action,
                "duplicate": True
            }
        
        comment_created_at = datetime.utcnow()
        if request.created_time:
            try:
                comment_created_at = datetime.fromisoformat(request.created_time.replace('Z', '+00:00'))
            except:
                pass
        
        response_entry = ResponseEntry(
            comment=str(request.original_comment),
            action=request.action,
            reply=str(request.reply),
            reasoning=request.reasoning,
            created_at=comment_created_at
        )
        db.add(response_entry)
        db.commit()
        
        TRAINING_DATA = load_training_data()
        
        return {
            "status": "approved",
            "message": f"Response approved and added to training data",
            "training_examples": len(TRAINING_DATA),
            "action_stored": request.action,
            "duplicate": False
        }
        
    except Exception as e:
        db.rollback()
        return {"status": "error", "error": str(e), "duplicate": False}
    finally:
        db.close()

@app.get("/batch-status/{job_id}")
def get_batch_status(job_id: str):
    """Get status of a batch processing job"""
    db = SessionLocal()
    try:
        batch_job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
        
        if not batch_job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        response = {
            "job_id": batch_job.job_id,
            "status": batch_job.status,
            "total_comments": batch_job.total_comments,
            "processed_comments": batch_job.processed_comments,
            "created_at": batch_job.created_at.isoformat(),
        }
        
        if batch_job.completed_at:
            response["completed_at"] = batch_job.completed_at.isoformat()
        
        if batch_job.status == "completed" and batch_job.results:
            response["results"] = json.loads(batch_job.results)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/stats")
async def get_stats():
    """Get training data statistics with updated info"""
    action_counts = {}
    
    for example in TRAINING_DATA:
        action = example.get('action', 'unknown')
        action_counts[action] = action_counts.get(action, 0) + 1
    
    return {
        "total_training_examples": len(TRAINING_DATA),
        "action_distribution": action_counts,
        "debug_info": {
            "total_unique_comments": len(request_tracker),
            "active_cooldowns": len(recent_requests),
            "duplicate_comments": sum(1 for v in request_tracker.values() if v > 1)
        },
        "key_features": {
            "duplicate_detection": "5 second cooldown per comment+feedback combination",
            "phone_number": "(844) 679-1188 ONLY for hesitation/contact requests/confusion",
            "website_ctas": "General prospects get website CTAs instead",
            "positioning": "Completely online loan facilitators",
            "analysis": "Case-by-case lease analysis, not generic claims"
        },
        "supported_actions": {
            "respond": "Generate helpful response with refined phone usage",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/accusations/hostility",
            "leave_alone": "Ignore harmless off-topic comments"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

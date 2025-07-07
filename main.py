from fastapi import FastAPI, Request, HTTPException
import os
from datetime import datetime, timezone, timedelta
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
import json
from typing import List, Dict, Optional
import uuid
import re

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

# Lease End Brand Context (unchanged)
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

# Custom Anthropic LM for DSPy (unchanged)
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

# Pydantic Models
class CommentRequest(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: Optional[str] = None
    postId: Optional[str] = None
    created_time: Optional[str] = ""
    memory_context: Optional[str] = ""
    
    def get_comment_text(self) -> str:
        return (self.comment or self.message or "No message content").strip()
    
    def get_comment_id(self) -> str:
        return (self.commentId or self.postId or "unknown").strip()

class Comment(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: str
    created_time: Optional[str] = ""
    
    def get_comment_text(self) -> str:
        return (self.comment or self.message or "No message content").strip()

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
    current_version: str = "v1"
    
    class Config:
        extra = "ignore"

class ApproveRequest(BaseModel):
    original_comment: str
    action: str
    reply: str
    reasoning: str = ""
    created_time: Optional[str] = ""
    
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
        "message": "Lease End AI Assistant - UPDATED VERSION",
        "version": "29.0-REFINED-PHONE",
        "training_examples": len(TRAINING_DATA),
        "status": "RUNNING",
        "features": ["Refined Phone Usage", "Case-by-Case Analysis", "Better Negative Handling", "Completely Online"],
        "key_changes": [
            "Phone number (844) 679-1188 ONLY for hesitation/contact requests/confusion",
            "Website CTAs for general prospects",
            "Case-by-case analysis instead of generic equity claims", 
            "DELETE for accusations and excessive arguing",
            "Completely online positioning",
            "Loan facilitators, not third-party financing"
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

@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Enhanced feedback processing with updated guidelines"""
    try:
        original_comment = request.original_comment.strip()
        original_response = request.original_response.strip() if request.original_response else ""
        original_action = request.original_action.strip().lower()
        feedback_text = request.feedback_text.strip()
        
        feedback_prompt = f"""You are improving a response based on human feedback for LeaseEnd.com.

ORIGINAL COMMENT: "{original_comment}"
YOUR ORIGINAL RESPONSE: "{original_response}"
YOUR ORIGINAL ACTION: "{original_action}"
HUMAN FEEDBACK: "{feedback_text}"

UPDATED COMPANY GUIDELINES:
- LeaseEnd helps drivers get loans in their name with competitive options, completely online
- We DON'T do third-party financing - we connect customers with lenders
- Lease buyouts vary case-by-case - offer to analyze their specific numbers
- Keep arguments concise and relevant, not generic equity claims
- Use (844) 679-1188 ONLY for customers who show hesitation, request contact, or are confused
- For general prospects, use website CTAs instead
- DELETE comments with accusations, excessive arguing, or hostility
- Focus on verification over broad statements

PHONE NUMBER USAGE:
- Use (844) 679-1188 format for hesitant/confused/contact-requesting customers only
- Include when customer seems to need personal assistance

Generate an IMPROVED response incorporating the feedback while following updated guidelines.

Respond in JSON: {{"sentiment": "...", "action": "REPLY/REACT/DELETE/LEAVE_ALONE", "reply": "...", "reasoning": "...", "confidence": 0.85, "needs_phone": true/false}}"""

        improved_response = claude.basic_request(feedback_prompt)
        
        try:
            response_clean = improved_response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
            
            if not response_clean.startswith('{'):
                json_start = response_clean.find('{')
                json_end = response_clean.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    response_clean = response_clean[json_start:json_end]
            
            result = json.loads(response_clean)
            
            action_mapping = {
                'reply': 'respond',
                'react': 'react', 
                'delete': 'delete',
                'leave_alone': 'leave_alone'
            }
            
            improved_action = action_mapping.get(result.get('action', 'leave_alone').lower(), 'leave_alone')
            improved_reply = filter_numerical_values(result.get('reply', ''))
            
            # Fix phone number format if present
            if '844' in improved_reply and '679-1188' in improved_reply:
                improved_reply = re.sub(r'\(?844\)?[-.\s]*679[-.\s]*1188', '(844) 679-1188', improved_reply)
            
            current_version_num = int(request.current_version.replace('v', '')) if request.current_version.startswith('v') else 1
            new_version = f"v{current_version_num + 1}"
            
            return {
                "commentId": request.commentId,
                "original_comment": original_comment,
                "category": result.get('sentiment', 'neutral').lower(),
                "action": improved_action,
                "reply": improved_reply,
                "confidence_score": float(result.get('confidence', 0.85)),
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": new_version,
                "reasoning": result.get('reasoning', 'Applied feedback with updated guidelines'),
                "feedback_processed": True,
                "success": True
            }
            
        except json.JSONDecodeError as e:
            new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
            return {
                "commentId": request.commentId,
                "original_comment": original_comment,
                "category": "neutral",
                "action": "leave_alone",
                "reply": "We can help analyze your specific lease situation. Call (844) 679-1188 if you have questions.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": f"v{new_version_num}",
                "reasoning": "Fallback response with proper phone format",
                "success": False
            }
            
    except Exception as e:
        new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
        return {
            "commentId": request.commentId,
            "original_comment": request.original_comment,
            "category": "error", 
            "action": "leave_alone",
            "reply": "We can help analyze your specific lease situation.",
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": request.feedback_text,
            "version": f"v{new_version_num}",
            "reasoning": "Error in feedback processing",
            "error": str(e),
            "success": False
        }

@app.post("/approve-response")
async def approve_response(request: ApproveRequest):
    """Approve and store a response for training"""
    db = SessionLocal()
    try:
        # Check if comment already exists
        existing_comment = db.query(ResponseEntry).filter(
            ResponseEntry.comment == request.original_comment
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
            comment=request.original_comment,
            action=request.action,
            reply=request.reply,
            reasoning=request.reasoning,
            created_at=comment_created_at
        )
        db.add(response_entry)
        db.commit()
        
        global TRAINING_DATA
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
        "key_features": {
            "phone_number": "(844) 679-1188 ONLY for hesitation/contact requests/confusion",
            "website_ctas": "General prospects get website CTAs instead",
            "positioning": "Completely online loan facilitators",
            "analysis": "Case-by-case lease analysis, not generic claims",
            "negative_handling": "DELETE accusations and excessive arguing"
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

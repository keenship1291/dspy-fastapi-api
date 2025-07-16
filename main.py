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
    """Enhanced comment classification with specific feedback requirements"""
    
    prompt = f"""You are analyzing comments for LeaseEnd.com, which helps drivers get loans for lease buyouts.

CRITICAL ANALYSIS REQUIREMENTS:

1. NAME DETECTION - If comment starts with a name (like "John", "@Sarah", "Mike Smith"), it's a conversation between users → LEAVE_ALONE
   Examples: "John that's crazy", "@Mike thanks", "Sarah I agree" = LEAVE_ALONE

2. LEASE BUYOUT RELEVANCE - Only engage if they seem like a potential lease buyout customer
   Potential customers mention: lease ending, lease return, buying lease, car payments, dealership options, lease terms
   NOT potential: general car talk, selling cars, insurance, repairs, unrelated topics

3. SPECIFIC ENGAGEMENT - Focus on their actual question/concern, not generic responses
   If they ask specific questions about their situation → REPLY with specific help
   If they share experiences → REACT or targeted REPLY
   If they're just chatting generally → LEAVE_ALONE

BUSINESS LOGIC:
- LeaseEnd helps drivers get loans in their name for lease buyouts, completely online
- We connect customers with lenders, don't do third-party financing
- Lease buyouts vary case-by-case based on specific numbers
- Focus on genuine prospects with actual lease decisions to make

DELETE CRITERIA:
- Accusations of false information or lies
- Spam, inappropriate, or hostile comments
- Brief negative: "scam", "ripoff", "terrible"
- Excessive arguing with no genuine interest

COMMENT: "{comment}"

Analyze carefully and respond in JSON:
{{"sentiment": "...", "action": "REPLY/REACT/DELETE/LEAVE_ALONE", "reasoning": "...", "is_conversation": true/false, "is_lease_relevant": true/false, "needs_phone": true/false}}"""

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
                'is_conversation': result.get('is_conversation', False),
                'is_lease_relevant': result.get('is_lease_relevant', False),
                'needs_phone': result.get('needs_phone', False)
            }
        except json.JSONDecodeError:
            # Enhanced fallback with name detection
            comment_lower = comment.lower()
            
            # Check for names at beginning (simple patterns)
            name_patterns = [
                r'^@\w+',  # @username
                r'^\w+\s+(that|this|is|was|are|were|thanks|thank)',  # "Name that..."
                r'^\w+\s+\w+\s+(that|this|is|was)',  # "First Last that..."
            ]
            
            is_conversation = any(re.match(pattern, comment_lower) for pattern in name_patterns)
            
            if is_conversation:
                action = 'LEAVE_ALONE'
                reasoning = "Detected conversation between users"
            elif any(word in comment.upper() for word in ['DELETE', 'SCAM', 'FALSE INFO', 'LIES', 'FRAUD']):
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
                'is_conversation': is_conversation,
                'is_lease_relevant': False,
                'needs_phone': False
            }
            
    except Exception as e:
        return {
            'sentiment': 'Neutral',
            'action': 'LEAVE_ALONE',
            'reasoning': f'Classification error: {str(e)}',
            'is_conversation': False,
            'is_lease_relevant': False,
            'needs_phone': False
        }

def generate_response(comment, sentiment, high_intent=False, needs_phone=False, is_lease_relevant=True, is_conversation=False):
    """Enhanced response generation with specific, targeted responses"""
    
    # If it's a conversation between users, don't respond
    if is_conversation:
        return ""
    
    # If not lease relevant, don't include CTAs
    if not is_lease_relevant:
        return ""
    
    # Analyze the specific content of their comment
    lease_indicators = {
        "lease_ending": ["lease ends", "lease is up", "lease expires", "end of lease", "lease ending"],
        "considering_buyout": ["buy my lease", "purchase my lease", "thinking about buying", "should I buy"],
        "dealership_pressure": ["dealership wants", "dealer says", "they're telling me", "trying to sell me"],
        "financial_concerns": ["can't afford", "too expensive", "cheaper option", "save money", "better deal"],
        "process_questions": ["how does it work", "what's the process", "how do I", "where do I start"],
        "comparison_shopping": ["vs returning", "vs buying", "compare options", "what's better"],
        "specific_situation": ["my lease", "my car", "my payment", "my situation", "my lease has"],
        "equity_questions": ["positive equity", "negative equity", "car is worth", "market value"],
        "timeline_urgency": ["need to decide", "running out of time", "deadline", "soon", "asap"]
    }
    
    # Determine the specific category of their comment
    comment_lower = comment.lower()
    primary_concern = None
    for category, indicators in lease_indicators.items():
        if any(indicator in comment_lower for indicator in indicators):
            primary_concern = category
            break
    
    # Check for hesitation/contact needs
    hesitation_indicators = [
        "not sure", "hesitant", "worried", "concerned", "skeptical",
        "don't trust", "seems too good", "what's the catch", "suspicious",
        "call me", "speak to someone", "talk to a person", "phone number"
    ]
    
    confusion_indicators = [
        "confused", "don't understand", "complicated", "explain", 
        "how does this work", "i'm lost", "need help understanding"
    ]
    
    shows_hesitation = any(indicator in comment_lower for indicator in hesitation_indicators)
    is_confused = any(indicator in comment_lower for indicator in confusion_indicators)
    
    # Build targeted response based on their specific concern
    response_templates = {
        "lease_ending": "Since your lease is ending, we can help you analyze whether buying makes sense based on your specific lease numbers.",
        "considering_buyout": "We can look at your actual lease numbers to help you decide if buying is the right choice for your situation.",
        "dealership_pressure": "Dealerships often have their own agenda. We can give you an independent analysis of your lease numbers to help you make the best decision.",
        "financial_concerns": "Every lease situation is different financially. We can analyze your specific numbers to see what options might work best for you.",
        "process_questions": "We handle the entire loan process online - from application to funding. It's designed to be simple and transparent.",
        "comparison_shopping": "We can help you compare the actual numbers between returning your lease versus buying it out with competitive financing.",
        "specific_situation": "Every lease situation is unique. We'd be happy to look at your specific lease terms and current market value to help you decide.",
        "equity_questions": "Equity in leases can be tricky to calculate. We can help you determine the real numbers based on your lease terms and current market value.",
        "timeline_urgency": "We understand you're working with a deadline. Our online process is designed to be quick while still getting you competitive rates."
    }
    
    # Phone number logic - only for specific needs
    phone_instruction = ""
    if shows_hesitation or is_confused or needs_phone:
        phone_instruction = " Give us a call at (844) 679-1188 if you'd prefer to speak with someone directly."
    
    # Website CTA for general prospects (no phone)
    website_cta = ""
    if not (shows_hesitation or is_confused or needs_phone):
        website_cta = " Check out LeaseEnd.com to see your options."
    
    prompt = f"""Create a specific, targeted response to this Facebook comment for LeaseEnd.com.

COMMENT: "{comment}"
PRIMARY CONCERN: {primary_concern or "general_inquiry"}
SENTIMENT: {sentiment}

RESPONSE REQUIREMENTS:
1. Address their SPECIFIC concern, not a generic response
2. Be conversational and natural (1-2 sentences max)
3. Reference their actual situation when possible
4. Don't be overly salesy - be helpful

COMPANY POSITIONING:
- We help drivers get loans in their name for lease buyouts, completely online
- We connect with multiple lenders for competitive rates
- Every lease situation is different - we analyze specific numbers
- No pressure, transparent process

TEMPLATE TO ADAPT: "{response_templates.get(primary_concern, 'We can help analyze your specific lease situation.')}"

ADDITIONAL ELEMENTS:
{phone_instruction}
{website_cta}

Generate a natural, specific response that directly addresses their comment:"""

    try:
        response = claude.basic_request(prompt)
        cleaned_response = filter_numerical_values(response.strip())
        
        # Ensure phone number format is correct if present
        if '844' in cleaned_response and '679-1188' in cleaned_response:
            cleaned_response = re.sub(r'\(?844\)?[-.\s]*679[-.\s]*1188', '(844) 679-1188', cleaned_response)
        
        return cleaned_response
    except Exception as e:
        # Fallback to simple, specific response
        if primary_concern and primary_concern in response_templates:
            fallback = response_templates[primary_concern]
            if shows_hesitation or is_confused:
                fallback += " Call (844) 679-1188 if you have questions."
            else:
                fallback += " Check out LeaseEnd.com to learn more."
            return fallback
        
        return "We can help analyze your specific lease situation to see what options work best for you."

# Simplified Pydantic Models - Only the fields you actually use
class CommentRequest(BaseModel):
    comment: str
    commentId: str
    created_time: Optional[str] = ""
    
    @field_validator('comment', mode='before')
    @classmethod
    def convert_comment_to_string(cls, v):
        if v is None:
            return ""
        try:
            return str(v)
        except:
            return ""
    
    @field_validator('commentId', mode='before')
    @classmethod
    def convert_comment_id_to_string(cls, v):
        if v is None:
            return "unknown"
        try:
            return str(v)
        except:
            return "unknown"

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
        "message": "Lease End AI Assistant - SIMPLIFIED VERSION",
        "version": "32.0-SIMPLIFIED",
        "training_examples": len(TRAINING_DATA),
        "status": "RUNNING",
        "features": ["Core Comment Processing", "Feedback System", "Duplicate Prevention"],
        "endpoints": [
            "/process-comment",
            "/process-feedback", 
            "/approve-response",
            "/health",
            "/stats"
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

@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Process a single comment with enhanced business logic"""
    try:
        comment_text = request.comment
        comment_id = request.commentId
        
        # Enhanced classification
        ai_classification = classify_comment_with_ai(comment_text, comment_id)
        
        sentiment = ai_classification.get('sentiment', 'Neutral')
        action = ai_classification.get('action', 'LEAVE_ALONE').lower()
        reasoning = ai_classification.get('reasoning', 'No reasoning provided')
        
        # Use the correct keys from the actual function return
        is_conversation = ai_classification.get('is_conversation', False)
        is_lease_relevant = ai_classification.get('is_lease_relevant', False)
        needs_phone = ai_classification.get('needs_phone', False)
        
        # Map actions to our system
        action_mapping = {
            'reply': 'respond',
            'react': 'react', 
            'delete': 'delete',
            'leave_alone': 'leave_alone'
        }
        
        mapped_action = action_mapping.get(action, 'leave_alone')
        
        # Generate response
        reply_text = ""
        confidence_score = 0.85
        
        if mapped_action == 'respond':
            reply_text = generate_response(
                comment_text, 
                sentiment, 
                high_intent=False,
                needs_phone=needs_phone,
                is_lease_relevant=is_lease_relevant,
                is_conversation=is_conversation
            )
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
        print(f"❌ Error in process_comment: {str(e)}")
        return ProcessedComment(
            commentId=request.commentId,
            original_comment=request.comment,
            category="error",
            action="leave_alone",
            reply="We can help analyze your specific lease situation.",
            confidence_score=0.0,
            approved="pending",
            reasoning=f"Error: {str(e)}"
        )

@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Process feedback with duplicate prevention and error handling"""
    start_time = time.time()
    
    try:
        comment_id = request.commentId
        original_comment = request.original_comment
        original_response = request.original_response
        original_action = request.original_action.strip().lower()
        feedback_text = request.feedback_text
        current_version = request.current_version
        
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
        
        print(f"✅ Processing feedback #{request_tracker[comment_id]} for: {comment_id}")
        
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
        
        feedback_lower = feedback_text.lower()
        for pattern, response_data in simple_feedback_patterns.items():
            if pattern in feedback_lower:
                print(f"⚡ FAST PATH: '{pattern}' detected")
                
                # Handle version increment
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
        
        # SLOW PATH: Use Claude API for complex feedback
        print(f"🤖 Using Claude API for complex feedback...")
        
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

        try:
            # Call Claude API
            improved_response = claude.basic_request(feedback_prompt)
            
            # Parse response
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
            
        except Exception as api_error:
            print(f"❌ API error: {api_error}")
            return {
                "commentId": comment_id,
                "original_comment": original_comment,
                "category": "error",
                "action": "leave_alone",
                "reply": "We can help analyze your specific lease situation.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": "v1",
                "reasoning": "API error fallback",
                "success": False,
                "error": str(api_error),
                "processing_time": round(time.time() - start_time, 3)
            }
            
    except Exception as e:
        print(f"❌ GENERAL ERROR: {e}")
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
            "error": str(e),
            "success": False,
            "processing_time": round(time.time() - start_time, 3)
        }

@app.post("/approve-response")
async def approve_response(request: ApproveRequest):
    """Approve and store a response for training"""
    global TRAINING_DATA
    
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

@app.get("/stats")
async def get_stats():
    """Get training data statistics"""
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

# Keep essential FB post endpoints for your workflow
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

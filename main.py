from fastapi import FastAPI, Request, HTTPException
import os
import csv
from datetime import datetime, timezone
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
import requests
import json
from typing import List, Dict, Optional

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Facebook Config (simplified - token is now unlimited)
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET") 
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
FACEBOOK_PAGE_TOKEN = os.getenv("FACEBOOK_PAGE_TOKEN")

# Google Sheets Configuration - Now using environment variables
GOOGLE_SHEETS_API_KEY = os.getenv("GOOGLE_SHEETS_API_KEY")
TRAINING_DATA_SHEET_ID = os.getenv("TRAINING_DATA_SHEET_ID", "1-dQAp8bgLcW7kri_6YHz3yZJrxDQMGr30GOrDmunnZk")
TRAINING_DATA_RANGE = os.getenv("TRAINING_DATA_RANGE", "6.7.25 Import!A:C")

# CSV File Paths
RESPONSE_DATABASE_CSV = "response_database.csv"
ACTIVE_FB_POST_ID_CSV = "active_fb_post_id.csv"

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
        return [result]  # DSPy expects a list
    
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

# CSV Management Functions
def read_response_database():
    """Read all entries from response_database.csv"""
    try:
        if not os.path.exists(RESPONSE_DATABASE_CSV):
            # Create file with headers if it doesn't exist
            with open(RESPONSE_DATABASE_CSV, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['comment', 'action', 'reply'])
            return []
        
        with open(RESPONSE_DATABASE_CSV, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)
    except Exception as e:
        print(f"Error reading response database: {e}")
        return []

def append_response_database(comment, action, reply):
    """Append a new entry to response_database.csv"""
    try:
        # Check if file exists, create with headers if not
        file_exists = os.path.exists(RESPONSE_DATABASE_CSV)
        
        with open(RESPONSE_DATABASE_CSV, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(['comment', 'action', 'reply'])
            
            # Write the new entry
            writer.writerow([comment, action, reply])
        
        return True
    except Exception as e:
        print(f"Error appending to response database: {e}")
        return False

def read_active_fb_post_ids():
    """Read all entries from active_fb_post_id.csv"""
    try:
        if not os.path.exists(ACTIVE_FB_POST_ID_CSV):
            # Create file with headers if it doesn't exist
            with open(ACTIVE_FB_POST_ID_CSV, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Ad account name', 'Campaign name', 'Ad set name', 'Ad name', 'Page ID', 'Post Id', 'Object Story ID'])
            return []
        
        with open(ACTIVE_FB_POST_ID_CSV, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)
    except Exception as e:
        print(f"Error reading active FB post IDs: {e}")
        return []

def append_active_fb_post_id(ad_account_name, campaign_name, ad_set_name, ad_name, page_id, post_id, object_story_id):
    """Append a new entry to active_fb_post_id.csv"""
    try:
        # Check if file exists, create with headers if not
        file_exists = os.path.exists(ACTIVE_FB_POST_ID_CSV)
        
        with open(ACTIVE_FB_POST_ID_CSV, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(['Ad account name', 'Campaign name', 'Ad set name', 'Ad name', 'Page ID', 'Post Id', 'Object Story ID'])
            
            # Write the new entry
            writer.writerow([ad_account_name, campaign_name, ad_set_name, ad_name, page_id, post_id, object_story_id])
        
        return True
    except Exception as e:
        print(f"Error appending to active FB post IDs: {e}")
        return False

# Google Sheets Integration Functions
def load_training_data_from_sheets():
    """Load training data from Google Sheets with robust error handling"""
    try:
        if not GOOGLE_SHEETS_API_KEY:
            print("⚠️ No Google Sheets API key found, skipping sheets data")
            return []
        
        # Try multiple ranges in order of preference
        ranges_to_try = [
            TRAINING_DATA_RANGE,  # Primary range from environment variable
            "6.7.25 Import!A:C",  # Known working range
            "A:C",  # No sheet name (uses first sheet)
            "Sheet1!A:C",  # Default fallback
        ]
        
        # Try each range until one works
        for range_attempt in ranges_to_try:
            try:
                url = f"https://sheets.googleapis.com/v4/spreadsheets/{TRAINING_DATA_SHEET_ID}/values/{range_attempt}?key={GOOGLE_SHEETS_API_KEY}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    print(f"✅ Successfully connected using range: {range_attempt}")
                    data = response.json()
                    values = data.get('values', [])
                    
                    if not values:
                        print("⚠️ No data found in Google Sheets")
                        return []
                    
                    # Process the data - expect 3 columns: comment, action, reply
                    training_examples = []
                    headers = values[0] if values else []
                    print(f"📋 Sheet headers: {headers}")
                    
                    for row_idx, row in enumerate(values[1:], 1):  # Skip header row
                        if len(row) >= 3:  # Ensure minimum required fields
                            training_examples.append({
                                'comment': row[0] if len(row) > 0 else '',
                                'action': row[1] if len(row) > 1 else 'leave_alone',
                                'reply': row[2] if len(row) > 2 else ''
                            })
                        elif len(row) >= 1 and row[0].strip():  # Handle partial rows with at least a comment
                            training_examples.append({
                                'comment': row[0],
                                'action': row[1] if len(row) > 1 else 'leave_alone',
                                'reply': row[2] if len(row) > 2 else ''
                            })
                    
                    print(f"✅ Loaded {len(training_examples)} training examples from Google Sheets")
                    return training_examples
                    
                else:
                    print(f"⚠️ Failed with range {range_attempt}: HTTP {response.status_code}")
                    continue
                    
            except Exception as e:
                print(f"⚠️ Failed with range {range_attempt}: {e}")
                continue
        
        print(f"❌ All range attempts failed for sheet ID: {TRAINING_DATA_SHEET_ID}")
        return []
        
    except Exception as e:
        print(f"❌ Error loading from Google Sheets: {e}")
        return []

def load_training_data_from_csv():
    """Load training examples from response_database.csv"""
    try:
        data = read_response_database()
        print(f"✅ Loaded {len(data)} training examples from response_database.csv")
        return data
    except Exception as e:
        print(f"❌ Error loading response_database.csv: {e}")
        return []

def load_training_data():
    """Load training examples from Google Sheets first, CSV as fallback"""
    print("🔄 Loading training data...")
    
    # Try Google Sheets first
    sheets_examples = load_training_data_from_sheets()
    
    # Load CSV as fallback/supplement
    csv_examples = load_training_data_from_csv()
    
    # Combine both sources
    all_examples = sheets_examples + csv_examples
    
    # Remove duplicates based on comment content
    seen_comments = set()
    deduplicated_examples = []
    
    for example in all_examples:
        comment_key = example['comment'].strip().lower()
        if comment_key not in seen_comments and comment_key:
            seen_comments.add(comment_key)
            deduplicated_examples.append(example)
    
    print(f"📊 Total training examples loaded:")
    print(f"   - Google Sheets: {len(sheets_examples)}")
    print(f"   - CSV Database: {len(csv_examples)}")
    print(f"   - Total (deduplicated): {len(deduplicated_examples)}")
    
    return deduplicated_examples

# Global training data
TRAINING_DATA = load_training_data()

# AI Classification
def classify_comment_with_ai(comment, postId=""):
    """Use Claude AI to classify comment with business logic"""
    
    prompt = f"""You are analyzing comments for LeaseEnd.com, which specializes in lease buyout services. 

BUSINESS LOGIC:
- Comments that negatively mention leasing alternatives (e.g., 'Trading in a lease is bad', 'Dealerships rip you off') should be classified as Positive, as they imply LeaseEnd's services are better
- Comments asking questions about lease buyouts, rates, or process are high-intent prospects
- Comments from people who clearly don't lease vehicles should be ignored/deleted
- Spam, irrelevant, or off-topic comments should be deleted

Analyze this comment and classify its sentiment as Positive, Neutral, or Negative. Then recommend one action: REPLY, REACT, DELETE, or IGNORE.

ACTIONS:
- REPLY: For questions, objections, or potential customers (generate helpful response)
- REACT: For positive comments, praise, or simple acknowledgments  
- DELETE: For spam, inappropriate content, or clearly non-prospects
- IGNORE: For off-topic but harmless comments

COMMENT: "{comment}"

Respond in this JSON format: {{"sentiment": "...", "action": "...", "reasoning": "...", "high_intent": true/false}}"""

    try:
        response = claude.basic_request(prompt)
        
        # Clean the response to extract JSON
        response_clean = response.strip()
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        # Try to parse JSON
        try:
            result = json.loads(response_clean)
            return {
                'sentiment': result.get('sentiment', 'Neutral'),
                'action': result.get('action', 'IGNORE'),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'high_intent': result.get('high_intent', False)
            }
        except json.JSONDecodeError:
            # Fallback parsing if JSON fails
            if 'DELETE' in response.upper():
                action = 'DELETE'
            elif 'REPLY' in response.upper():
                action = 'REPLY'
            elif 'REACT' in response.upper():
                action = 'REACT'
            else:
                action = 'IGNORE'
                
            return {
                'sentiment': 'Neutral',
                'action': action,
                'reasoning': 'Fallback classification',
                'high_intent': False
            }
            
    except Exception as e:
        print(f"Error in AI classification: {e}")
        return {
            'sentiment': 'Neutral',
            'action': 'IGNORE',
            'reasoning': 'Classification error',
            'high_intent': False
        }

# AI Response Generation
def generate_response(comment, sentiment, high_intent=False):
    """Generate natural response using Claude"""
    
    # Get relevant training examples for context
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            if any(word in example['comment'].lower() for word in comment.lower().split()[:4]):
                relevant_examples.append(f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\"")
    
    context_examples = "\n\n".join(relevant_examples[:3])
    
    # Add CTA instructions for high-intent comments
    cta_instruction = ""
    if high_intent:
        cta_instruction = "\nFor high-intent prospects, end with: 'To see your options just fill out the form on our site, we're happy to help'"
    
    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com, a lease buyout financing company.

COMMENT SENTIMENT: {sentiment}
HIGH INTENT PROSPECT: {high_intent}

ORIGINAL COMMENT: "{comment}"

BRAND VOICE:
- Professional but conversational
- Transparent about pricing (no hidden fees)
- Helpful, not pushy
- Emphasize online process convenience

RESPONSE STYLE:
- Sound natural and human
- Start responses naturally: "Actually..." "That's a good point..." "Not exactly..."
- Use commas, never dashes (- or --)
- Maximum 1 exclamation point
- Keep concise (1-2 sentences usually)
- Address their specific concern directly

EXAMPLES OF GOOD RESPONSES:
{context_examples}

{cta_instruction}

Generate a helpful, natural response that addresses their comment directly:"""

    try:
        response = claude.basic_request(prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Thank you for your comment! We'd be happy to help with any lease buyout questions."

# Pydantic Models
class CommentRequest(BaseModel):
    comment: str
    postId: str
    created_time: str = ""
    memory_context: str = ""

class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    category: str
    action: str  # respond, react, delete, leave_alone
    reply: str
    confidence_score: float
    approved: str  # "pending", "yes", "no"

class FeedbackRequest(BaseModel):
    original_comment: str
    original_response: str = ""
    original_action: str
    feedback_text: str
    postId: str
    current_version: str = "v1"
    
    class Config:
        extra = "ignore"

class ResponseDatabaseEntry(BaseModel):
    comment: str
    action: str
    reply: str

class ActiveFBPostEntry(BaseModel):
    ad_account_name: str
    campaign_name: str
    ad_set_name: str
    ad_name: str
    page_id: str
    post_id: str
    object_story_id: str

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Core AI System",
        "version": "14.0",
        "training_examples": len(TRAINING_DATA),
        "actions": ["respond", "react", "delete", "leave_alone"],
        "features": ["Pure AI Classification", "Real-time Learning", "Google Sheets Integration", "CSV Database Management"],
        "approach": "Core AI functionality - comment processing and feedback learning",
        "endpoints": {
            "/process-comment": "Initial comment processing",
            "/process-feedback": "Human feedback processing",
            "/test-feedback": "Debug endpoint for testing n8n integration",
            "/stats": "View training data statistics",
            "/generate-reply": "Backwards compatibility endpoint",
            "/response-database": "Read response database",
            "/response-database/append": "Add to response database",
            "/active-fb-posts": "Read active FB post IDs",
            "/active-fb-posts/append": "Add to active FB post IDs"
        },
        "csv_files": {
            "response_database": {
                "file": "response_database.csv",
                "fields": ["comment", "action", "reply"]
            },
            "active_fb_posts": {
                "file": "active_fb_post_id.csv",
                "fields": ["Ad account name", "Campaign name", "Ad set name", "Ad name", "Page ID", "Post Id", "Object Story ID"]
            }
        },
        "training_data_system": {
            "primary_source": "Google Sheets",
            "fallback_source": "response_database.csv",
            "sheet_id": TRAINING_DATA_SHEET_ID,
            "range": TRAINING_DATA_RANGE,
            "managed_by": "n8n workflows"
        },
        "philosophy": "Let Claude do what it does best - understand context and generate responses."
    }

# CSV Database Endpoints
@app.get("/response-database")
async def get_response_database():
    """Get all entries from response_database.csv"""
    try:
        data = read_response_database()
        return {
            "success": True,
            "count": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading response database: {str(e)}")

@app.post("/response-database/append")
async def append_to_response_database(entry: ResponseDatabaseEntry):
    """Append new entry to response_database.csv"""
    try:
        success = append_response_database(entry.comment, entry.action, entry.reply)
        if success:
            return {
                "success": True,
                "message": "Entry added to response database",
                "entry": entry.dict()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to append to response database")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error appending to response database: {str(e)}")

@app.get("/active-fb-posts")
async def get_active_fb_posts():
    """Get all entries from active_fb_post_id.csv"""
    try:
        data = read_active_fb_post_ids()
        return {
            "success": True,
            "count": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading active FB post IDs: {str(e)}")

@app.post("/active-fb-posts/append")
async def append_to_active_fb_posts(entry: ActiveFBPostEntry):
    """Append new entry to active_fb_post_id.csv"""
    try:
        success = append_active_fb_post_id(
            entry.ad_account_name,
            entry.campaign_name,
            entry.ad_set_name,
            entry.ad_name,
            entry.page_id,
            entry.post_id,
            entry.object_story_id
        )
        if success:
            return {
                "success": True,
                "message": "Entry added to active FB posts",
                "entry": entry.dict()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to append to active FB posts")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error appending to active FB posts: {str(e)}")

# Original Endpoints (unchanged)
@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Core comment processing using AI classification"""
    try:
        # Use AI to classify the comment
        ai_classification = classify_comment_with_ai(request.comment, request.postId)
        
        sentiment = ai_classification['sentiment']
        action = ai_classification['action'].lower()
        reasoning = ai_classification['reasoning']
        high_intent = ai_classification['high_intent']
        
        # Map actions to our system
        action_mapping = {
            'reply': 'respond',
            'react': 'react', 
            'delete': 'delete',
            'ignore': 'leave_alone'
        }
        
        mapped_action = action_mapping.get(action, 'leave_alone')
        
        # Generate response only if action is 'respond'
        reply_text = ""
        confidence_score = 0.85
        
        if mapped_action == 'respond':
            reply_text = generate_response(request.comment, sentiment, high_intent)
            confidence_score = 0.9
        
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category=sentiment.lower(),
            action=mapped_action,
            reply=reply_text,
            confidence_score=confidence_score,
            approved="pending"
        )
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category="error",
            action="leave_alone",
            reply="Thank you for your comment. We appreciate your feedback.",
            confidence_score=0.0,
            approved="pending"
        )

@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Use human feedback to improve responses"""
    try:
        # Clean and validate input data
        original_comment = request.original_comment.strip()
        original_response = request.original_response.strip() if request.original_response else "No response was generated"
        original_action = request.original_action.strip().lower()
        feedback_text = request.feedback_text.strip()
        
        print(f"🔄 Processing feedback for postId: {request.postId}")
        print(f"📝 Feedback: {feedback_text[:100]}...")
        
        # Enhanced prompt that includes human feedback
        feedback_prompt = f"""You are improving a response based on human feedback for LeaseEnd.com.

ORIGINAL COMMENT: "{original_comment}"
YOUR ORIGINAL RESPONSE: "{original_response}"
YOUR ORIGINAL ACTION: "{original_action}"

HUMAN FEEDBACK: "{feedback_text}"

LEARN FROM THIS FEEDBACK:
- If the feedback mentions tone (too formal/pushy/etc), adjust your response style accordingly
- If the feedback mentions accuracy, focus on factual correctness
- If the feedback mentions business logic (wrong action), reconsider the classification
- If the feedback mentions missing CTA, add appropriate call-to-action
- If the feedback says "delete this" or "don't respond", change action accordingly
- If the feedback says "address objection" or "respond to this", change action to REPLY

Generate an IMPROVED response that incorporates this feedback. 

IMPORTANT: 
- If human says "this should be deleted" or "don't respond to this", recommend DELETE or IGNORE action
- If human says "add CTA" or mentions "high intent", include call-to-action
- If human says "too formal", make it more conversational
- If human says "too pushy", make it more helpful and less sales-y
- If human says "address objection" or "respond to this", recommend REPLY action

Respond in this JSON format: {{"sentiment": "...", "action": "REPLY/REACT/DELETE/IGNORE", "reply": "...", "improvements_made": "...", "confidence": 0.85}}"""

        improved_response = claude.basic_request(feedback_prompt)
        
        # Parse the improved response
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
            
            # Map actions to our system
            action_mapping = {
                'reply': 'respond',
                'react': 'react', 
                'delete': 'delete',
                'ignore': 'leave_alone'
            }
            
            improved_action = action_mapping.get(result.get('action', 'ignore').lower(), 'leave_alone')
            
            # Calculate new version number
            try:
                current_version_num = int(request.current_version.replace('v', '')) if request.current_version.startswith('v') else 1
            except:
                current_version_num = 1
            new_version = f"v{current_version_num + 1}"
            
            return {
                "postId": request.postId,
                "original_comment": original_comment,
                "category": result.get('sentiment', 'neutral').lower(),
                "action": improved_action,
                "reply": result.get('reply', ''),
                "confidence_score": float(result.get('confidence', 0.85)),
                "approved": "pending",
                "feedback_text": "",
                "version": new_version,
                "improvements_made": result.get('improvements_made', 'Applied human feedback'),
                "feedback_processed": True,
                "success": True
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
            return {
                "postId": request.postId,
                "original_comment": original_comment,
                "category": "neutral",
                "action": "leave_alone",
                "reply": "Thank you for your comment. We appreciate your feedback.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": "",
                "version": f"v{new_version_num}",
                "error": f"JSON parsing failed: {str(e)}",
                "success": False
            }
            
    except Exception as e:
        print(f"❌ Feedback processing error: {e}")
        new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
        return {
            "postId": request.postId,
            "original_comment": request.original_comment,
            "category": "error",
            "action": "leave_alone",
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": "",
            "version": f"v{new_version_num}",
            "error": str(e),
            "success": False
        }

@app.post("/test-feedback")
async def test_feedback(request: dict):
    """Simple endpoint to test what n8n is actually sending"""
    print("🔍 Test endpoint received data:")
    print(f"Request data: {request}")
    
    return {
        "status": "success",
        "message": "Test endpoint working",
        "received_data": request,
        "data_types": {key: str(type(value)) for key, value in request.items()},
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate-reply")
async def generate_reply(request: Request):
    """Simple reply generation for backwards compatibility"""
    data = await request.json()
    comment = data.get("comment", "")
    postId = data.get("postId", "")

    try:
        ai_classification = classify_comment_with_ai(comment, postId)
        
        if ai_classification['action'].lower() == 'reply':
            reply = generate_response(comment, ai_classification['sentiment'], ai_classification['high_intent'])
        else:
            reply = "Thank you for your comment."
        
        return {
            "postId": postId,
            "reply": reply,
            "action": ai_classification['action'].lower()
        }
        
    except Exception as e:
        return {
            "postId": postId,
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "action": "ignore",
            "error": str(e)
        }

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
        "data_structure": {
            "fields": ["comment", "action", "reply"],
            "field_count": 3,
            "primary_source": "Google Sheets",
            "fallback_source": "response_database.csv",
            "sheet_id": TRAINING_DATA_SHEET_ID,
            "range": TRAINING_DATA_RANGE
        },
        "supported_actions": {
            "respond": "Generate helpful response (with CTA for high-intent)",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/inappropriate/non-prospect content",
            "leave_alone": "Ignore harmless off-topic comments"
        }
    }

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

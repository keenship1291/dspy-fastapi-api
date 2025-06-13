from fastapi import FastAPI, Request
import os
import csv
from datetime import datetime, timezone
from anthropic import Anthropic
import dspy
from pydantic import BaseModel

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Facebook Config (simplified - token is now unlimited)
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET") 
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
FACEBOOK_PAGE_TOKEN = os.getenv("FACEBOOK_PAGE_TOKEN")

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

# Load training data from CSV - Enhanced for 4-action system
def load_training_data():
    """Load training examples from CSV file - supports both old and new formats"""
    training_examples = []
    
    # Try new format first (enhanced_training_data.csv)
    csv_files = ['enhanced_training_data.csv', 'training_data.csv']
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Handle both old and new CSV formats
                    if 'action' in row:
                        # New format with 4-action classification
                        training_examples.append({
                            'comment': row['comment'],
                            'category': row['category'],
                            'urgency': row['urgency'],
                            'action': row['action'],  # respond, react, delete, leave_alone
                            'reply': row['reply']
                        })
                    else:
                        # Old format - convert to new action system
                        requires_response = row.get('requires_response', 'false').lower() == 'true'
                        category = row.get('category', 'general')
                        
                        # Map old format to new actions
                        if not requires_response:
                            action = 'leave_alone'
                        elif category == 'praise':
                            action = 'react'
                        elif category in ['spam', 'inappropriate']:
                            action = 'delete'
                        else:
                            action = 'respond'
                        
                        training_examples.append({
                            'comment': row['comment'],
                            'category': category,
                            'urgency': row.get('urgency', 'medium'),
                            'action': action,
                            'reply': row.get('reply', '')
                        })
            print(f"✅ Loaded {len(training_examples)} training examples from {csv_file}")
            break  # Stop after successfully loading from first available file
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            continue
    
    if not training_examples:
        print("⚠️ No training data found, using fallback classification")
    
    return training_examples

# Global training data
TRAINING_DATA = load_training_data()

# Simplified classification using business-focused approach
def classify_comment_with_ai(comment, postId=""):
    """Use Claude AI to classify comment with business logic, similar to original n8n approach"""
    
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
        # Parse the JSON response
        import json
        
        # Clean the response to extract JSON
        response_clean = response.strip()
        if response_clean.startswith('```'):
            # Remove code block markers
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

# Simplified AI response generation 
def generate_response(comment, sentiment, high_intent=False):
    """Generate natural response using Claude with simplified approach"""
    
    # Get relevant training examples for context
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            # Simple keyword matching for relevance
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

# Request/Response models - Streamlined for clean Google Sheets structure
class CommentRequest(BaseModel):
    comment: str
    postId: str
    created_time: str = ""
    memory_context: str = ""

# Streamlined response model matching Google Sheets structure
class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    category: str
    urgency: str
    action: str  # respond, react, delete, leave_alone
    reply: str
    confidence_score: float
    approved: str  # "pending", "yes", "no"

# Feedback processing model - Made more flexible for n8n data
class FeedbackRequest(BaseModel):
    original_comment: str
    original_response: str = ""  # Allow empty responses
    original_action: str
    feedback_text: str
    postId: str
    current_version: str = "v1"
    
    # Allow extra fields from n8n without failing
    class Config:
        extra = "ignore"

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Self-Learning System with Bulletproof Setup",
        "version": "7.0",
        "training_examples": len(TRAINING_DATA),
        "actions": ["reply", "react", "delete", "ignore"],
        "features": ["AI Sentiment Analysis", "Business Logic Classification", "High-Intent Detection", "CTA Integration", "Self-Learning Training Data", "Human Feedback Loop", "Bulletproof Field Handling"],
        "approach": "Self-improving AI system with robust data handling and unlimited Facebook token",
        "endpoints": {
            "/process-comment": "Initial comment processing",
            "/process-feedback": "Human feedback processing (production)",
            "/save-approved-example": "Save human-approved responses to training data (bulletproof)",
            "/test-feedback": "Debug endpoint for testing n8n integration",
            "/stats": "View training data statistics and learning progress"
        },
        "learning_system": {
            "description": "Automatically saves approved responses to enhanced_training_data.csv",
            "benefits": ["Continuous improvement", "Domain-specific learning", "Human-guided AI training"],
            "workflow": "Human approves → Saves to CSV → Reloads training data → Better future responses"
        },
        "facebook_integration": {
            "token_status": "Unlimited (never expires)",
            "automation_ready": True
        }
    }

@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Simplified comment processing using AI classification approach"""
    try:
        # Use AI to classify the comment (like original n8n approach)
        ai_classification = classify_comment_with_ai(request.comment, request.postId)
        
        sentiment = ai_classification['sentiment']
        action = ai_classification['action'].lower()  # Convert to lowercase for consistency
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
        confidence_score = 0.85  # Higher confidence for AI-based classification
        
        if mapped_action == 'respond':
            reply_text = generate_response(request.comment, sentiment, high_intent)
            confidence_score = 0.9  # High confidence for AI-generated responses
        
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category=sentiment.lower(),    # Use sentiment as category
            urgency="high" if high_intent else "medium",
            action=mapped_action,
            reply=reply_text,
            confidence_score=confidence_score,
            approved="pending"  # Always starts as pending for human review
        )
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category="error",
            urgency="low",
            action="leave_alone",
            reply="Thank you for your comment. We appreciate your feedback.",
            confidence_score=0.0,
            approved="pending"
        )

# Streamlined feedback processing endpoint - More robust for n8n
@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Use human feedback to improve DSPy response - returns data for overwriting original fields"""
    try:
        # Clean and validate input data
        original_comment = request.original_comment.strip()
        original_response = request.original_response.strip() if request.original_response else "No response was generated"
        original_action = request.original_action.strip().lower()
        feedback_text = request.feedback_text.strip()
        
        # Log what we received for debugging
        print(f"🔄 Processing feedback for postId: {request.postId}")
        print(f"📝 Feedback: {feedback_text[:100]}...")
        print(f"🎯 Original action: {original_action}")
        
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

        # Get improved response from Claude
        print("🤖 Calling Claude for improvement...")
        improved_response = claude.basic_request(feedback_prompt)
        
        # Parse the improved response with better error handling
        import json
        try:
            # Clean response
            response_clean = improved_response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
            
            # Handle case where Claude doesn't return JSON
            if not response_clean.startswith('{'):
                # Try to extract JSON from the response
                json_start = response_clean.find('{')
                json_end = response_clean.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    response_clean = response_clean[json_start:json_end]
            
            result = json.loads(response_clean)
            print(f"✅ Successfully parsed Claude response")
            
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
            
            # Prepare response data for Google Sheets update
            response_data = {
                # Data structure for overwriting original Google Sheets fields
                "postId": request.postId,
                "original_comment": original_comment,
                "category": result.get('sentiment', 'neutral').lower(),
                "urgency": "medium",  # Keep consistent
                "action": improved_action,
                "reply": result.get('reply', ''),
                "confidence_score": float(result.get('confidence', 0.85)),
                "approved": "pending",  # Reset for re-review
                "feedback_text": "",  # Clear after processing
                "version": new_version,
                "improvements_made": result.get('improvements_made', 'Applied human feedback'),
                "feedback_processed": True,
                "success": True
            }
            
            print(f"✅ Feedback processed successfully - new version: {new_version}")
            return response_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {improved_response[:200]}...")
            
            # Fallback response
            new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
            return {
                "postId": request.postId,
                "original_comment": original_comment,
                "category": "neutral",
                "urgency": "medium",
                "action": "leave_alone",
                "reply": "Thank you for your comment. We appreciate your feedback.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": "",
                "version": f"v{new_version_num}",
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": improved_response[:300] + "..." if len(improved_response) > 300 else improved_response,
                "success": False
            }
            
    except Exception as e:
        print(f"❌ Feedback processing error: {e}")
        new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
        return {
            "postId": request.postId,
            "original_comment": request.original_comment,
            "category": "error",
            "urgency": "low", 
            "action": "leave_alone",
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": "",
            "version": f"v{new_version_num}",
            "error": str(e),
            "success": False
        }

# Save approved examples to training data
class ApprovedExample(BaseModel):
    original_comment: str
    approved_category: str
    approved_action: str
    approved_reply: str = ""  # Made optional with default empty string
    approved_urgency: str = "medium"
    postId: str = ""  # Made optional with default
    confidence_score: float = 0.9
    version: str = "v1"
    human_feedback: str = ""
    
@app.post("/save-approved-example")
async def save_approved_example(request: ApprovedExample):
    """Save human-approved examples to enhanced_training_data.csv for future learning"""
    try:
        # Prepare the training data row
        training_row = {
            'comment': request.original_comment.strip(),
            'category': request.approved_category.lower(),
            'urgency': request.approved_urgency,
            'action': request.approved_action,
            'reply': request.approved_reply.strip(),
            'human_feedback': request.human_feedback,
            'confidence': request.confidence_score,
            'postId': request.postId,
            'version': request.version,
            'date_added': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'human_approved'
        }
        
        print(f"💾 Saving approved example to training data:")
        print(f"   Comment: {request.original_comment[:50]}...")
        print(f"   Action: {request.approved_action}")
        print(f"   Reply: {request.approved_reply[:50]}...")
        
        # Append to enhanced_training_data.csv
        file_exists = os.path.isfile('enhanced_training_data.csv')
        
        with open('enhanced_training_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['comment', 'category', 'urgency', 'action', 'reply', 'human_feedback', 
                         'confidence', 'postId', 'version', 'date_added', 'source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
                print("📝 Created new enhanced_training_data.csv with headers")
            
            writer.writerow(training_row)
        
        # Reload training data to include new example
        global TRAINING_DATA
        TRAINING_DATA = load_training_data()
        
        print(f"✅ Successfully saved approved example to training data")
        print(f"📊 Total training examples now: {len(TRAINING_DATA)}")
        
        return {
            "status": "success",
            "message": "Approved example saved to training data",
            "training_examples_count": len(TRAINING_DATA),
            "saved_data": {
                "comment_preview": request.original_comment[:100] + "..." if len(request.original_comment) > 100 else request.original_comment,
                "action": request.approved_action,
                "reply_preview": request.approved_reply[:100] + "..." if len(request.approved_reply) > 100 else request.approved_reply,
                "category": request.approved_category,
                "date_added": training_row['date_added']
            }
        }
        
    except Exception as e:
        print(f"❌ Error saving approved example: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to save approved example to training data"
        }

# Test endpoint for debugging n8n issues  
@app.post("/test-feedback")
async def test_feedback(request: dict):
    """Simple endpoint to test what n8n is actually sending"""
    print("🔍 Test endpoint received data:")
    print(f"Request type: {type(request)}")
    print(f"Request data: {request}")
    
    return {
        "status": "success",
        "message": "Test endpoint working",
        "received_data": request,
        "data_types": {key: str(type(value)) for key, value in request.items()},
        "timestamp": datetime.now().isoformat()
    }

# Keep backwards compatibility
@app.post("/generate-reply")
async def generate_reply(request: Request):
    """Simple reply generation for backwards compatibility"""
    data = await request.json()
    comment = data.get("comment", "")
    postId = data.get("postId", "")

    try:
        # Use simplified AI classification
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

# Debug endpoint for single comment testing
@app.post("/debug-comment")
async def debug_comment(request: CommentRequest):
    """Debug endpoint to see AI classification details for a single comment"""
    try:
        ai_classification = classify_comment_with_ai(request.comment, request.postId)
        
        debug_response = {
            "comment": request.comment,
            "ai_classification": ai_classification,
            "training_examples_loaded": len(TRAINING_DATA),
            "approach": "Streamlined AI classification with clean feedback loop"
        }
        
        # Show what response would be generated if it's a reply
        if ai_classification['action'].lower() == 'reply':
            response_text = generate_response(
                request.comment, 
                ai_classification['sentiment'], 
                ai_classification['high_intent']
            )
            debug_response["generated_response"] = response_text
            debug_response["includes_cta"] = "fill out the form" in response_text.lower()
        
        return debug_response
        
    except Exception as e:
        return {
            "error": str(e),
            "comment": request.comment,
            "fallback": "Debug failed"
        }

# Debug Facebook configuration
@app.get("/debug-facebook-config")
async def debug_facebook_config():
    return {
        "app_id": os.getenv("FACEBOOK_APP_ID"),
        "app_secret_exists": bool(os.getenv("FACEBOOK_APP_SECRET")),
        "page_id": os.getenv("FACEBOOK_PAGE_ID"), 
        "page_token_exists": bool(os.getenv("FACEBOOK_PAGE_TOKEN")),
        "token_status": "Unlimited (never expires)",
        "app_secret_preview": os.getenv("FACEBOOK_APP_SECRET", "")[:10] + "..." if os.getenv("FACEBOOK_APP_SECRET") else "Missing"
    }

# New endpoint for action statistics
@app.get("/stats")
async def get_stats():
    """Get training data and action statistics including recent additions"""
    action_counts = {}
    category_counts = {}
    source_counts = {}
    
    for example in TRAINING_DATA:
        action = example.get('action', 'unknown')
        category = example.get('category', 'unknown')
        source = example.get('source', 'original')
        
        action_counts[action] = action_counts.get(action, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    
    return {
        "total_training_examples": len(TRAINING_DATA),
        "action_distribution": action_counts,
        "category_distribution": category_counts,
        "source_distribution": source_counts,
        "supported_actions": {
            "reply": "Generate helpful response (with CTA for high-intent)",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/inappropriate/non-prospect content",
            "ignore": "Leave harmless off-topic comments alone"
        },
        "approach": "Self-learning AI with human feedback loop and bulletproof data handling",
        "features": ["Sentiment Analysis", "High-Intent Detection", "Automatic CTA", "Business Logic", "Self-Learning Training Data", "Human-Approved Examples", "Robust Field Handling"],
        "learning_system": {
            "human_approved_examples": source_counts.get('human_approved', 0),
            "original_training_data": source_counts.get('original', len(TRAINING_DATA)),
            "csv_file": "enhanced_training_data.csv",
            "auto_reload": True
        }
    }

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

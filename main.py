from fastapi import FastAPI, Request
import os
import csv
from datetime import datetime, timezone
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
from facebook_auth import FacebookTokenManager

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Facebook Config
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET") 
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")

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
                            'reply': row['reply'],
                            'requires_response': row['action'] == 'respond'  # Backwards compatibility
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
                            'reply': row.get('reply', ''),
                            'requires_response': requires_response
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

# Request/Response models - Enhanced for 4-action system
class CommentRequest(BaseModel):
    comment: str
    postId: str
    created_time: str = ""
    memory_context: str = ""

class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    category: str
    urgency: str
    action: str  # respond, react, delete, leave_alone
    requires_response: bool
    reply: str
    should_post: bool
    needs_review: bool
    confidence_score: float

app = FastAPI()

# Initialize Facebook token manager
fb_manager = FacebookTokenManager()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Simplified AI Classification",
        "version": "4.0",
        "training_examples": len(TRAINING_DATA),
        "actions": ["reply", "react", "delete", "ignore"],
        "features": ["AI Sentiment Analysis", "Business Logic Classification", "High-Intent Detection", "CTA Integration", "Facebook Token Management"],
        "approach": "Simplified classification using proven prompt pattern + Claude AI responses"
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
        requires_response = mapped_action == 'respond'
        
        # Generate response only if action is 'respond'
        reply_text = ""
        confidence_score = 0.85  # Higher confidence for AI-based classification
        
        if mapped_action == 'respond':
            reply_text = generate_response(request.comment, sentiment, high_intent)
            confidence_score = 0.9  # High confidence for AI-generated responses
        
        # Simplified decision logic
        should_post = (
            mapped_action == 'respond' and 
            sentiment != 'Negative' and  # Don't auto-post negative sentiment
            confidence_score > 0.8
        )
        
        needs_review = (
            mapped_action == 'delete' or  # Always review deletes
            sentiment == 'Negative' or    # Review negative sentiment
            confidence_score < 0.8
        )
        
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category=sentiment.lower(),    # Use sentiment as category
            urgency="high" if high_intent else "medium",
            action=mapped_action,
            requires_response=requires_response,
            reply=reply_text,
            should_post=should_post,
            needs_review=needs_review,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category="error",
            urgency="low",
            action="leave_alone",
            requires_response=False,
            reply="Thank you for your comment. We appreciate your feedback.",
            should_post=False,
            needs_review=True,
            confidence_score=0.0
        )

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
            "approach": "Simplified AI classification (business-focused like original n8n prompt)"
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

# Validate AI performance against training data
@app.post("/validate-training-accuracy")
async def validate_training_accuracy():
    """Test AI classification accuracy against actual training data"""
    if not TRAINING_DATA:
        return {"error": "No training data loaded"}
    
    # Test a sample of training data (first 10 examples)
    sample_size = min(10, len(TRAINING_DATA))
    results = []
    
    for i in range(sample_size):
        example = TRAINING_DATA[i]
        try:
            ai_classification = classify_comment_with_ai(example['comment'])
            
            # Map training actions to AI actions
            expected_action = example['action']
            if expected_action == 'respond':
                expected_action = 'reply'
            elif expected_action == 'leave_alone':
                expected_action = 'ignore'
                
            actual_action = ai_classification['action'].lower()
            
            results.append({
                "comment": example['comment'][:50] + "...",
                "expected_action": expected_action,
                "actual_action": actual_action,
                "match": expected_action == actual_action,
                "training_category": example['category']
            })
            
        except Exception as e:
            results.append({
                "comment": example['comment'][:50] + "...",
                "error": str(e),
                "match": False
            })
    
    accuracy = sum(1 for r in results if r.get('match', False)) / len(results)
    
    return {
        "sample_size": len(results),
        "accuracy": f"{accuracy:.2%}",
        "results": results,
        "note": "Testing AI classification against actual training data"
    }

# Facebook Token Management Endpoints
@app.post("/refresh-facebook-token")
async def refresh_facebook_token_endpoint():
    """Manual endpoint to refresh Facebook token"""
    result = fb_manager.refresh_page_token()
    
    if result.get("success"):
        return {
            "status": "success",
            "message": "Token refreshed successfully",
            "refreshed_at": result["refreshed_at"],
            "new_token_preview": result["new_token"][:20] + "...",
            "note": "Update FACEBOOK_PAGE_TOKEN environment variable with the new token"
        }
    else:
        return {
            "status": "error", 
            "error": result.get("error"),
            "details": result.get("details")
        }

@app.get("/facebook-token-status")
async def check_facebook_token_status():
    """Check when Facebook token expires"""
    return fb_manager.check_token_status()

# New endpoint for action statistics
@app.get("/stats")
async def get_stats():
    """Get training data and action statistics"""
    action_counts = {}
    category_counts = {}
    
    for example in TRAINING_DATA:
        action = example.get('action', 'unknown')
        category = example.get('category', 'unknown')
        
        action_counts[action] = action_counts.get(action, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    return {
        "total_training_examples": len(TRAINING_DATA),
        "action_distribution": action_counts,
        "category_distribution": category_counts,
        "supported_actions": {
            "reply": "Generate helpful response (with CTA for high-intent)",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/inappropriate/non-prospect content",
            "ignore": "Leave harmless off-topic comments alone"
        },
        "approach": "Simplified AI classification based on business logic and sentiment",
        "features": ["Sentiment Analysis", "High-Intent Detection", "Automatic CTA", "Business Logic", "Facebook Token Management"]
    }

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

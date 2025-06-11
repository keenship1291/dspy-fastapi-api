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

# Customer Avatar Detection
def identify_customer_avatar(comment):
    """Identify customer avatar based on comment content"""
    comment_lower = comment.lower()
    
    # Anna (Busy Professional) - time/efficiency focused
    anna_keywords = ["busy", "quick", "time", "efficient", "fast", "hassle", "simple", "easy"]
    
    # Mike (Cost-Conscious) - price/value focused  
    mike_keywords = ["cost", "price", "rate", "save", "money", "cheap", "expensive", "fee", "payment"]
    
    # Sarah (Family-Oriented) - stability/simplicity focused
    sarah_keywords = ["family", "kids", "reliable", "stable", "safe", "secure", "trust", "worry"]
    
    anna_score = sum(1 for word in anna_keywords if word in comment_lower)
    mike_score = sum(1 for word in mike_keywords if word in comment_lower)
    sarah_score = sum(1 for word in sarah_keywords if word in comment_lower)
    
    if mike_score > anna_score and mike_score > sarah_score:
        return "mike_cost_conscious"
    elif sarah_score > anna_score and sarah_score > mike_score:
        return "sarah_family"
    elif anna_score > 0:
        return "anna_professional"
    else:
        return "general"

# Enhanced 4-action classification using training data + brand context
def smart_classify_comment(comment, comment_age="recent"):
    """Classify comment using training data patterns and determine appropriate action"""
    comment_lower = comment.lower()
    
    # First, check for spam/inappropriate content
    spam_indicators = [
        "free money", "click here", "get rich", "scam", "virus", "mlm", "onlyfans",
        "hot singles", "lottery", "nigerian prince", "urgent!!!", "buy my course"
    ]
    if any(indicator in comment_lower for indicator in spam_indicators):
        return {
            'category': 'spam',
            'urgency': 'high',
            'action': 'delete',
            'requires_response': False
        }
    
    # Check for inappropriate content
    inappropriate_indicators = ["f***", "stupid", "idiot", "hate", "racist", "sexist"]
    if any(indicator in comment_lower for indicator in inappropriate_indicators):
        return {
            'category': 'inappropriate',
            'urgency': 'high', 
            'action': 'delete',
            'requires_response': False
        }
    
    # Check for off-topic content
    off_topic_indicators = [
        "mcdonald", "plumber", "cat", "dog", "game tonight", "girlfriend",
        "dinner", "weather", "testing", "first!", "random"
    ]
    if any(indicator in comment_lower for indicator in off_topic_indicators):
        return {
            'category': 'off_topic',
            'urgency': 'low',
            'action': 'leave_alone', 
            'requires_response': False
        }
    
    # Check for praise/positive reactions
    praise_indicators = ["thank you", "awesome", "great", "love", "perfect", "❤️", "👍"]
    if any(indicator in comment_lower for indicator in praise_indicators) and len(comment) < 50:
        return {
            'category': 'praise',
            'urgency': 'low',
            'action': 'react',
            'requires_response': False
        }
    
    # Check training data for similar patterns
    best_match = None
    max_similarity = 0
    
    for example in TRAINING_DATA:
        example_lower = example['comment'].lower()
        # Simple similarity check (could be enhanced with embeddings)
        common_words = set(comment_lower.split()) & set(example_lower.split())
        if len(common_words) >= 2:  # At least 2 words in common
            similarity = len(common_words)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = example
    
    # If we found a good match, use its classification
    if best_match and max_similarity >= 2:
        return {
            'category': best_match['category'],
            'urgency': best_match['urgency'],
            'action': best_match['action'],
            'requires_response': best_match['action'] == 'respond'
        }
    
    # Fallback: Brand-aligned classification logic with actions
    if any(word in comment_lower for word in ['rate', 'cost', 'price', 'expensive', 'fee', 'payment', 'money']):
        return {'category': 'price_objection', 'urgency': 'medium', 'action': 'respond', 'requires_response': True}
    elif any(word in comment_lower for word in ['dealership', 'bank', 'credit union', 'better', 'vs', 'compare']):
        return {'category': 'competitor_comparison', 'urgency': 'medium', 'action': 'respond', 'requires_response': True}
    elif any(word in comment_lower for word in ['process', 'how', 'steps', 'time', 'quick', 'easy', 'hassle']):
        return {'category': 'process_concern', 'urgency': 'medium', 'action': 'respond', 'requires_response': True}
    elif any(word in comment_lower for word in ['terrible', 'awful', 'bad', 'worst', 'hate', 'scam']):
        return {'category': 'complaint', 'urgency': 'high', 'action': 'respond', 'requires_response': True}
    elif '?' in comment or any(word in comment_lower for word in ['can i', 'do you', 'how do', 'what']):
        return {'category': 'primary_inquiry', 'urgency': 'medium', 'action': 'respond', 'requires_response': True}
    else:
        return {'category': 'general', 'urgency': 'low', 'action': 'leave_alone', 'requires_response': False}

# AI-powered response generation using Claude
def generate_ai_response(comment, avatar, category, urgency):
    """
    Generate contextual response using Claude AI with training data context.
    
    This replaces template-based responses with dynamic AI generation that:
    - Uses training data as examples/context for Claude
    - Incorporates customer avatar preferences  
    - Maintains brand voice and values
    - Generates natural, contextual responses
    """
    
    # Get relevant training examples for context
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            # Include examples from same category or similar patterns
            if (example['category'] == category or 
                any(word in example['comment'].lower() for word in comment.lower().split()[:3])):
                relevant_examples.append(f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\"")
    
    # Limit to top 3 most relevant examples to avoid token limits
    context_examples = "\n\n".join(relevant_examples[:3])
    
    # Build avatar context
    avatar_context = {
        "mike_cost_conscious": "This customer is cost-conscious and wants to know about pricing, savings, and value. Focus on transparent pricing and competitive rates.",
        "anna_professional": "This customer is a busy professional who values efficiency and convenience. Emphasize quick, online processes and time-saving benefits.",
        "sarah_family": "This customer is family-oriented and wants stability and trust. Focus on reliability, no-pressure approach, and guidance through the process.",
        "general": "This customer needs helpful, professional service information."
    }
    
    # Create prompt for Claude
    prompt = f"""You are responding to a Facebook comment for Lease End, a lease buyout financing company. Generate a helpful, professional response that matches our brand voice.

BRAND VALUES:
- Transparent pricing (no hidden fees)
- 100% online process (no dealership visits)
- No pressure, customer-focused
- Competitive rates through multiple lenders
- Professional, helpful, and trustworthy

CUSTOMER AVATAR: {avatar_context.get(avatar, avatar_context['general'])}

COMMENT CATEGORY: {category}
URGENCY: {urgency}

ORIGINAL COMMENT: "{comment}"

EXAMPLES OF GOOD RESPONSES:
{context_examples}

Generate a natural, helpful response that:
1. Directly addresses their specific comment/concern
2. Reflects Lease End's brand voice and values
3. Is conversational but professional
4. Includes relevant information without being pushy
5. Matches the tone and urgency level

Keep the response concise (1-3 sentences) and personalized to their specific comment."""

    try:
        # Use Claude to generate the response
        response = claude.basic_request(prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating AI response: {e}")
        # Fallback to a simple, safe response
        return "Thank you for your comment! We'd be happy to help you with your lease buyout questions. Feel free to reach out for more information."

# Request/Response models - Enhanced for 4-action system
class CommentRequest(BaseModel):
    comment: str
    postId: str
    created_time: str = ""
    memory_context: str = ""

class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    customer_avatar: str
    category: str
    urgency: str
    action: str  # respond, react, delete, leave_alone
    requires_response: bool
    reply: str
    should_post: bool
    needs_review: bool
    confidence_score: float

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Claude AI Powered",
        "version": "3.1",
        "training_examples": len(TRAINING_DATA),
        "actions": ["respond", "react", "delete", "leave_alone"],
        "features": ["Customer Avatar Detection", "Claude AI Dynamic Responses", "4-Action Classification", "Contextual Training Data"],
        "response_method": "Claude AI generates responses using training data as context (no rigid templates)"
    }

@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Enhanced comment processing with 4-action classification system"""
    try:
        # Calculate comment age
        comment_age = "recent"
        if request.created_time:
            try:
                created = datetime.fromisoformat(request.created_time.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                age_hours = (now - created).total_seconds() / 3600
                
                if age_hours < 2:
                    comment_age = "recent"
                elif age_hours < 24:
                    comment_age = "hours_old"
                else:
                    comment_age = "days_old"
            except:
                comment_age = "unknown"
        
        # Identify customer avatar
        avatar = identify_customer_avatar(request.comment)
        
        # Use smart classification with 4-action system
        classification = smart_classify_comment(request.comment, comment_age)
        
        category = classification['category']
        urgency = classification['urgency']
        action = classification['action']
        requires_response = classification['requires_response']
        
        # Generate brand-aligned response only if action is 'respond'
        reply_text = ""
        confidence_score = 0.8  # Base confidence
        
        if action == 'respond':
            reply_text = generate_ai_response(request.comment, avatar, category, urgency)
            
            # Adjust confidence based on training data match
            comment_lower = request.comment.lower()
            for example in TRAINING_DATA:
                if example['action'] == 'respond':
                    example_lower = example['comment'].lower()
                    common_words = set(comment_lower.split()) & set(example_lower.split())
                    if len(common_words) >= 2:
                        confidence_score = min(0.95, confidence_score + (len(common_words) * 0.05))
                        break
        elif action in ['react', 'delete', 'leave_alone']:
            # High confidence for clear non-response actions
            confidence_score = 0.9
        
        # Enhanced decision logic with 4-action system
        should_post = (
            action == 'respond' and 
            urgency != "high" and
            comment_age != "days_old" and
            category not in ["complaint"] and
            confidence_score > 0.75
        )
        
        needs_review = (
            action == 'respond' and 
            (urgency == "high" or comment_age == "days_old" or category == "complaint" or confidence_score <= 0.75)
        ) or action == 'delete'  # Always review delete actions
        
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            customer_avatar=avatar,
            category=category,
            urgency=urgency,
            action=action,
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
            customer_avatar="general",
            category="error",
            urgency="low",
            action="leave_alone",
            requires_response=False,
            reply="Thank you for your comment. We appreciate your feedback and will get back to you soon.",
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
        # Use enhanced processing
        avatar = identify_customer_avatar(comment)
        classification = smart_classify_comment(comment)
        
        if classification['action'] == 'respond':
            reply = generate_ai_response(comment, avatar, classification['category'], classification['urgency'])
        else:
            reply = "Thank you for your comment."
        
        return {
            "postId": postId,
            "reply": reply,
            "action": classification['action']
        }
        
    except Exception as e:
        return {
            "postId": postId,
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "action": "respond",
            "error": str(e)
        }

# Debug endpoint for testing
@app.post("/debug-comment")
async def debug_comment(request: CommentRequest):
    """Debug endpoint to see all classification details"""
    avatar = identify_customer_avatar(request.comment)
    classification = smart_classify_comment(request.comment)
    
    # Show what prompt would be sent to Claude
    debug_response = {
        "comment": request.comment,
        "customer_avatar": avatar,
        "classification": classification,
        "training_examples_loaded": len(TRAINING_DATA),
        "ai_context": "Using Claude AI for dynamic response generation with training data context"
    }
    
    # If it's a respond action, show the AI prompt that would be generated
    if classification['action'] == 'respond':
        # Get relevant training examples for context
        relevant_examples = []
        for example in TRAINING_DATA:
            if example['action'] == 'respond' and example['reply']:
                if (example['category'] == classification['category'] or 
                    any(word in example['comment'].lower() for word in request.comment.lower().split()[:3])):
                    relevant_examples.append(f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\"")
        
        context_examples = "\n\n".join(relevant_examples[:3])
        
        avatar_context = {
            "mike_cost_conscious": "This customer is cost-conscious and wants to know about pricing, savings, and value. Focus on transparent pricing and competitive rates.",
            "anna_professional": "This customer is a busy professional who values efficiency and convenience. Emphasize quick, online processes and time-saving benefits.",
            "sarah_family": "This customer is family-oriented and wants stability and trust. Focus on reliability, no-pressure approach, and guidance through the process.",
            "general": "This customer needs helpful, professional service information."
        }
        
        debug_response["ai_prompt_preview"] = f"""Customer Avatar: {avatar_context.get(avatar, avatar_context['general'])}
Category: {classification['category']}
Training Examples Used: {len(relevant_examples[:3])}
AI will generate contextual response based on this information."""
    
    return debug_response

# Test AI response generation
@app.post("/test-ai-response")
async def test_ai_response(request: CommentRequest):
    """Test endpoint to see AI-generated response in real-time"""
    try:
        avatar = identify_customer_avatar(request.comment)
        classification = smart_classify_comment(request.comment)
        
        if classification['action'] == 'respond':
            ai_response = generate_ai_response(request.comment, avatar, classification['category'], classification['urgency'])
            
            return {
                "original_comment": request.comment,
                "customer_avatar": avatar,
                "category": classification['category'],
                "urgency": classification['urgency'],
                "ai_generated_response": ai_response,
                "method": "Claude AI with training data context"
            }
        else:
            return {
                "original_comment": request.comment,
                "action": classification['action'],
                "message": f"No response needed - action is '{classification['action']}'"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "fallback": "AI response generation failed"
        }

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
            "respond": "Generate professional reply",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/inappropriate content",
            "leave_alone": "Ignore off-topic comments"
        }
    }

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

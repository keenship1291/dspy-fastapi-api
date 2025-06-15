# Use the working version 20.0 structure but with enhanced business logic

# AI Classification with enhanced business logic but same output structure
def classify_comment_with_ai(comment, postId=""):
    """Use Claude AI to classify comment with enhanced business logic"""
    
    prompt = f"""You are analyzing comments for LeaseEnd.com, which specializes in lease buyout services. 

BUSINESS LOGIC & COMPANY POSITION:
- LeaseEnd is THE BEST option for lease buyout financing - be confident about this
- Lease buyouts are currently a GREAT financial decision due to inflated used car prices
- Never agree with false information about lease buyouts being bad financial decisions
- Challenge misinformation politely but firmly - we have the expertise
- Always effectively address core objections to make buyouts and LeaseEnd the clear best option
- Comments that negatively mention leasing alternatives (e.g., 'Trading in a lease is bad', 'Dealerships rip you off') should be classified as Positive, as they imply LeaseEnd's services are better
- Comments asking questions about lease buyouts, rates, or process are high-intent prospects
- Comments from people who clearly don't lease vehicles should be ignored/deleted
- Spam, irrelevant, or off-topic comments should be deleted
- Positive feedback/testimonials should ALWAYS get a reply with appreciation
- LEAVE ALONE neutral comments that seem to reference other comments UNLESS they're negative toward LeaseEnd/lease buyouts

CURRENT MARKET CONTEXT:
- Used car prices are inflated, making lease buyouts financially advantageous
- Many people don't understand the current market opportunity
- We should educate people on why NOW is a great time for lease buyouts
- Challenge any comments suggesting lease buyouts are always bad financial decisions

ACTIONS:
- REPLY: For questions, objections, potential customers, misinformation that needs correction, OR positive feedback/testimonials
- REACT: For positive comments that don't need a response, or simple acknowledgments
- DELETE: For spam, inappropriate content, or clearly non-prospects
- IGNORE: For off-topic but harmless comments, OR neutral comments referencing other comments (unless negative toward us)

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
                reasoning = "Fallback classification: Detected DELETE action in response"
            elif 'REPLY' in response.upper():
                action = 'REPLY'
                reasoning = "Fallback classification: Detected REPLY action in response"
            elif 'REACT' in response.upper():
                action = 'REACT'
                reasoning = "Fallback classification: Detected REACT action in response"
            else:
                action = 'IGNORE'
                reasoning = "Fallback classification: No clear action detected"
                
            return {
                'sentiment': 'Neutral',
                'action': action,
                'reasoning': reasoning,
                'high_intent': False
            }
            
    except Exception as e:
        print(f"Error in AI classification: {e}")
        return {
            'sentiment': 'Neutral',
            'action': 'IGNORE',
            'reasoning': f'Classification error: {str(e)}',
            'high_intent': False
        }

# Enhanced response generation but keeping working structure
def generate_response(comment, sentiment, high_intent=False):
    """Generate natural response using Claude with enhanced business logic"""
    
    # Get relevant training examples for context
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            if any(word in example['comment'].lower() for word in comment.lower().split()[:4]):
                relevant_examples.append(f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\"")
    
    context_examples = "\n\n".join(relevant_examples[:3])
    
    # Detect if this is positive feedback/testimonial
    positive_feedback_indicators = [
        "thank you", "thanks", "great service", "amazing", "fantastic", "love", "perfect", 
        "excellent", "wonderful", "awesome", "best", "helped me", "saved me", "grateful",
        "appreciate", "thumbs up", "recommend", "highly recommend", "satisfied", "happy"
    ]
    
    is_positive_feedback = any(indicator in comment.lower() for indicator in positive_feedback_indicators)
    
    # Detect misinformation that needs correction
    misinformation_indicators = [
        "lease buyouts are bad", "never buy your lease", "always return", "terrible idea",
        "waste of money", "financial mistake", "bad deal", "rip off", "scam"
    ]
    
    needs_correction = any(indicator in comment.lower() for indicator in misinformation_indicators)
    
    # Detect rate/pricing questions
    rate_questions = ["rate", "rates", "interest", "apr", "cost", "price", "pricing", "fee", "fees", "how much"]
    asking_about_rates = any(indicator in comment.lower() for indicator in rate_questions)
    
    # Add CTA instructions for high-intent comments
    cta_instruction = ""
    if high_intent:
        cta_instruction = "\nFor high-intent prospects, end with: 'To see your options just fill out the form on our site, we're happy to help'"
    
    # Special instructions for positive feedback
    positive_feedback_instruction = ""
    if is_positive_feedback:
        positive_feedback_instruction = "\nThis is positive feedback/testimonial - respond with a SHORT, varied appreciation message like 'Thank you!', 'Enjoy your ride!', 'Pleasure working with you!', 'We're so glad we could help!', 'Thanks for choosing LeaseEnd!', or similar. Keep it brief and genuine."
    
    # Special instructions for misinformation correction
    correction_instruction = ""
    if needs_correction:
        correction_instruction = "\nThis comment contains misinformation about lease buyouts. Politely but firmly correct it. Emphasize that with current inflated used car prices, lease buyouts are actually a SMART financial decision. Position LeaseEnd as the expert choice. Always effectively address the core objection to make buyouts and LeaseEnd the clear best option."
    
    # Special instructions for rate/pricing questions
    pricing_instruction = ""
    if asking_about_rates:
        pricing_instruction = """
IMPORTANT PRICING GUIDELINES:
- NEVER share exact rate estimates - say rates depend on make/model and credit info
- For pricing questions, say: "We charge a doc fee, which pays for the service we provide. The only additional amount you will pay beyond the price of your vehicle, would be for tax, title and registration fees, which you would pay regardless of whatever route you choose."
- NEVER mention exact fee amounts
- Position this as transparent and competitive"""
    
    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com, a lease buyout financing company.

COMPANY POSITION - BE CONFIDENT:
- LeaseEnd is THE BEST option for lease buyout financing
- Lease buyouts are currently a GREAT financial decision due to inflated used car prices
- We are the experts - don't agree with misinformation about lease buyouts being bad
- Challenge false information politely but firmly
- Used car market is inflated - this makes lease buyouts financially smart right now
- ALWAYS effectively address core objections to make buyouts and LeaseEnd the clear best option

COMMENT SENTIMENT: {sentiment}
HIGH INTENT PROSPECT: {high_intent}

ORIGINAL COMMENT: "{comment}"

BRAND VOICE:
- Professional but conversational
- Confident about our expertise and market position
- Transparent about pricing (no hidden fees)
- Helpful, not pushy, but firmly educational when needed
- Emphasize online process convenience
- Always address core objections effectively

RESPONSE STYLE:
- Sound natural and human
- Start responses naturally: "Actually..." "That's a good point..." "Not exactly..." "Thanks for..."
- Use commas, never dashes (- or --)
- Maximum 1 exclamation point
- Keep concise (1-2 sentences usually)
- Address their specific concern directly
- Don't blindly agree with misinformation
- Make LeaseEnd the clear best choice

EXAMPLES OF GOOD RESPONSES:
{context_examples}

{positive_feedback_instruction}
{correction_instruction}
{pricing_instruction}
{cta_instruction}

Generate a helpful, natural response that addresses their comment directly and makes LeaseEnd the clear best option:"""

    try:
        response = claude.basic_request(prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Thank you for your comment! We'd be happy to help with any lease buyout questions."

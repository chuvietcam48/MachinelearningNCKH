import os
import google.generativeai as genai
import json

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-2.5-flash')

reviews = [
    {"id": "1", "text": "o produto chegou rapido e perfeito"},
    {"id": "2", "text": "nao recebi o produto, quero reembolso"},
    {"id": "3", "text": "muito bom"},
    {"id": "4", "text": "veio quebrado e atrasado"}
]

prompt = f'''You are an expert e-commerce data annotator.
Taxonomy Constraints & Definitions:
- Customer_Service_Returns
- Delivery_Fulfillment
- Domain_Experience
- Price_Value
- Product_Condition_Quality
- Product_Performance_Usability
- Other_Specific

Rules:
1. Return EXACTLY ONE JSON object with a single key `results`.
2. The value of `results` must be a list of objects, one for each review provided.
3. Each object MUST have:
   - `id`: the exact string ID of the review.
   - `items`: a list of up to TWO aspects found in that review. Each aspect has `aspect` (from Taxonomy) and `polarity` ("Positive" or "Negative"). If generic or none, use empty list [].

Reviews:
'''
for r in reviews:
    prompt += f"ID: {r['id']} | Text: \"{r['text']}\"\n"

try:
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type='application/json',
        )
    )
    print('Success:', response.text)
except Exception as e:
    import traceback
    print('Error:', str(e))
    traceback.print_exc()

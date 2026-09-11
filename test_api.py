import os
import google.generativeai as genai

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-2.5-flash')

text = 'the product was okay but delivery was late'
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
1. Return EXACTLY ONE JSON object with a single key `items` containing a list of aspect objects.
2. Each aspect object MUST have EXACTLY two keys: `aspect` (string, from Taxonomy above) and `polarity` (string: "Positive" or "Negative").
3. Return ONLY valid JSON.

Review Text:
"{text}"
'''

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

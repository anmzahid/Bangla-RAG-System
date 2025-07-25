from together import Together
from config import API_KEY  # Import the API key from config.py

client = Together(api_key=API_KEY)

response = client.chat.completions.create(
  model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
  messages=[
    {
      "role": "user",
      "content": "What are some fun things to do in New York?"
    }
  ]
)
print(response.choices[0].message.content)
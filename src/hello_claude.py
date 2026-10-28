from dotenv import load_dotenv
import anthropic

load_dotenv()                      # reads ANTHROPIC_API_KEY from .env
client = anthropic.Anthropic()     # creates a connection to Claude

response = client.messages.create(
    model="claude-haiku-5-5",      # cheapest, fastest model
    max_tokens=100,                # caps the length (and cost) of the answer
    messages=[{"role": "user",
               "content": "In one sentence, what is retrieval-augmented generation?"}],
)

print(response.content[0].text)
print("Input tokens: ", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
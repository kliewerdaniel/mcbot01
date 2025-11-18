#!/usr/bin/env python3
from scripts.conversation_reasoning_agent import ConversationReasoningAgent

# Test the chat system directly
agent = ConversationReasoningAgent()

print("=========================================")
print("TESTING CHAT SYSTEM WITH DEBUGGING")
print("=========================================")

query = "What tools do bloggers use in VS Code?"

print(f"\nQuery: {query}")
print("\nDEBUG OUTPUT:")
print("=" * 50)

result = agent.generate_response(query, [])

print("\nFINAL RESULT:")
print("=" * 50)
print(f"Response length: {len(result['response'])} characters")
print(f"Context documents used: {len(result['context_used'])}")
print(f"Retrieval performed: {result['retrieval_performed']}")

if result['context_used']:
    print("\nContext documents found:")
    for i, doc in enumerate(result['context_used'][:2]):
        print(f"  {i+1}: {doc.get('title', 'N/A')} - {doc.get('filename', 'N/A')}")

print("\nResponse preview:")
print(result['response'][:300] + "..." if len(result['response']) > 300 else result['response'])

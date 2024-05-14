# chatbot_app.py
import streamlit as st
from openai import OpenAI

# GPT models and prices per 1k tokens (in dollars)
GPT_PRICING = {
    'gpt-3.5-turbo': {
        'prompt': 0.0005,  # input cost per 1k tokens
        'completion': 0.0015  # output cost per 1k tokens
    },
    'gpt-4-turbo': {
        'prompt': 0.01,  # input cost per 1k tokens
        'completion': 0.03  # output cost per 1k tokens
    },
    'gpt-4o': {
        'prompt': 0.005,  # input cost per 1k tokens
        'completion': 0.015  # output cost per 1k tokens
    }
}

def calculate_cost(model, prompt_tokens, completion_tokens):
    """
    Calculate the cost for a specific OpenAI API request.
    """
    pricing = GPT_PRICING[model]
    prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
    completion_cost = (completion_tokens / 1000) * pricing['completion']
    return prompt_cost, completion_cost, prompt_cost + completion_cost

def generate_full_prompt(messages, prompt):
    """
    Generate a full prompt input including the conversation history.
    """
    messages_str = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in messages])
    return f"{messages_str}\nUser: {prompt}"

def chat_with_openai(prompt, model, api_key):
    """
    Interact with OpenAI's API and get a response.
    """
    try:
        openai_client = OpenAI(api_key=api_key)

        # Retrieve chat history
        messages = st.session_state['messages']
        messages.append({"role": "user", "content": prompt})

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        reply = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # Add assistant's reply to the chat history
        messages.append({"role": "assistant", "content": reply})

        return reply, prompt_tokens, completion_tokens
    except Exception as e:
        return str(e), 0, 0

# Clear chat history
def clear_history(system_message):
    st.session_state['chat_history'] = []
    st.session_state['total_cost'] = 0
    st.session_state['messages'] = [{"role": "system", "content": system_message}]

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    
    # API Key input
    api_key_input = st.text_input("Enter your OpenAI API Key:")
    
    # Model selection
    model_selection = st.selectbox("Select GPT Model:", options=list(GPT_PRICING.keys()), index=0)
    
    # System Message input
    system_message_input = st.text_area("System Message:", "You are a helpful assistant.")
    if 'system_message' not in st.session_state or st.session_state['system_message'] != system_message_input:
        st.session_state['system_message'] = system_message_input
        clear_history(system_message_input)
    
    # Clear History button
    if st.button("Clear History"):
        clear_history(st.session_state['system_message'])

# Streamlit App
st.title("OpenAI GPT Price Calculator")
st.write("Interact with OpenAI's GPT models and track usage costs.")

# Initialize session state for chat history和cost
if 'chat_history' not in st.session_state:
    clear_history(st.session_state['system_message'])

if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": st.session_state['system_message']}]

# Input prompt
prompt_input = st.text_area("Enter your question or prompt:")

# Submit button (disabled if API Key is empty)
if st.button("Submit", disabled=not api_key_input):
    if prompt_input:
        reply, prompt_tokens, completion_tokens = chat_with_openai(prompt_input, model_selection, api_key_input)
        prompt_cost, completion_cost, total_cost = calculate_cost(model_selection, prompt_tokens, completion_tokens)
        st.session_state['total_cost'] += total_cost
        full_prompt = generate_full_prompt(st.session_state['messages'][:-2], prompt_input)
        st.session_state['chat_history'].append({
            'model': model_selection,
            'full_prompt': full_prompt,
            'reply': reply,
            'prompt_cost': prompt_cost,
            'completion_cost': completion_cost,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_cost': total_cost
        })

# Display chat history
st.write("### Chat History")
for i, chat in enumerate(st.session_state['chat_history']):
    st.write(f"**Model {i + 1}:** {chat['model']}")
    st.write(f"**Prompt {i + 1}:** {chat['full_prompt']}")
    st.write(f"**Reply {i + 1}:** {chat['reply']}")
    st.write(f"**Prompt Tokens {i + 1}:** {chat['prompt_tokens']}")
    st.write(f"**Prompt Cost {i + 1}:** ${chat['prompt_cost']:.5f}")
    st.write(f"**Completion Tokens {i + 1}:** {chat['completion_tokens']}")
    st.write(f"**Completion Cost {i + 1}:** ${chat['completion_cost']:.5f}")
    st.write(f"**Total Cost {i + 1}:** ${chat['total_cost']:.5f}")
    st.write("---")

# Display total cost
st.write(f"### Total Cost: ${st.session_state['total_cost']:.5f}")

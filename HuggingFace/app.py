from transformers import pipeline
import gradio as gr

# Initialize the chatbot pipeline
chatbot = pipeline(model="facebook/blenderbot-400M-distill")


# Define the chatbot function
def vanilla_chatbot(message, history):
    # Extract past conversation
    conversation_history = []
    if history:
        for item in history:
            conversation_history.append(item[0])  # User message
            conversation_history.append(item[1])  # Bot response

    # Add the new user message to the history
    conversation_history.append(message)

    # Format the conversation history for the chatbot
    context = "\n".join(conversation_history)

    # Generate the chatbot response
    response = chatbot(context, max_length=100, clean_up_tokenization_spaces=True)[0][
        "generated_text"
    ]

    return response


# Create a Gradio ChatInterface
demo_chatbot = gr.ChatInterface(
    vanilla_chatbot,
    title="Vanilla Chatbot",
    description="Enter text to start chatting.",
)

# Launch the Gradio interface
demo_chatbot.launch(share=True)

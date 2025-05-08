import gradio as gr
from chat import get_response

def chat_with_bot(message):
    return get_response(message)

iface = gr.Interface(fn=chat_with_bot, inputs="text", outputs="text",
                     title="e-Commerce Support Chatbot",
                     description="Ask about orders, returns, or any help.")
iface.launch()
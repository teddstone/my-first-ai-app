import gradio as gr
from transformers import pipeline

# Downloads a quick sentiment analysis AI model
classifier = pipeline("sentiment-analysis")

def analyze_text(text):
    result = classifier(text)
    return f"Label: {result[0]['label']} | Score: {round(result[0]['score'] * 100, 2)}%"

# Sets up the web view
demo = gr.Interface(fn=analyze_text, inputs="text", outputs="text", title="My First AI App")

demo.launch()

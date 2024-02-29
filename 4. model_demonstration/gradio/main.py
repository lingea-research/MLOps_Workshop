import gradio as gr
import fasttext


model = fasttext.load_model('models/normalized_model4.bin')


def detect_language(text):
    prediction = model.predict(text)
    return str(prediction)


demo = gr.Interface(
    fn=detect_language,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()

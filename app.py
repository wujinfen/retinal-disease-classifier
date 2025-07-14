from fastai.vision.all import load_learner, PILImage

import gradio as gr

learn = load_learner("retina_model.pkl")

def predict(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Retinal Disease Classifier",
    description="Upload a retinal image to detect cataracts, glaucoma, diabetic retinopathy, or normal retina"
)

app.launch()

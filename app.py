import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Ceramic Crack Detection Classifier"
description = "A crack detection classifier trained on the kThis dataset is taken from the website Mendeley Data - Crack Detection, contributed by Çağlar Fırat Özgenel."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
examples = ['siamese.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3)).launch()
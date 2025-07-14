import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Set Agg backend to avoid Qt issues
import matplotlib.pyplot as plt
from polynomial import polynomial_tab
from linear import linear_tab
from image import image_tab

with gr.Blocks(title="Polynomial and Linear System Solver") as demo:
    # Create all tabs
    poly_components = polynomial_tab()
    linear_components = linear_tab()
    image_components = image_tab()

if __name__ == "__main__":
    demo.launch()  # Removed server_port=7860 to allow automatic port selection

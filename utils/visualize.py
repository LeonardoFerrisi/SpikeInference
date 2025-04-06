import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchviz import make_dot


def visualize_model(model:nn.Module, input_size:int, device:torch.device):
    """
    Visualize a model architecture using torchviz.

    @param model: The PyTorch nn model to visualize.
    @param input_size: The size of the input tensor.
    @param device: The device to use (CPU or GPU). (Defined earlier when initilizing tourch)
    """
    # Create a dummy input tensor for visualization
    dummy_input = torch.randn(1, 10, input_size).to(device)
    output = model(dummy_input)

    try:
        graph = make_dot(output, params=dict(model.named_parameters()))
        graph.render("stacked_bilstm_architecture", format="png", cleanup=True)
        print("Model architecture visualization saved as 'stacked_bilstm_architecture.png'.")

        import matplotlib.image as mpimg
        img = mpimg.imread("stacked_bilstm_architecture.png")
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Stacked BiLSTM Model Architecture")
        plt.show()
    except Exception as e:
        print("An error occurred while generating the graph:", e)
        print("Make sure Graphviz is installed and the 'dot' executable is available in your system's PATH.")
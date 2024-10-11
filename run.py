import gradio as gr
from gpt2_model import generate_response
from adversarial_attack import apply_adversarial_attack

def generate_response_with_attack(user_input):
    # Get the response from GPT-2
    gpt_output = generate_response(user_input)
    # Apply adversarial attack to the GPT-2 output
    attacked_output = apply_adversarial_attack(gpt_output)
    return attacked_output  # Return the attacked output

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_response_with_attack,
    inputs="text",  # Input type
    outputs="text",  # Output type
    title="GPT-2 with Adversarial Attack",  # Title of the interface
    description="Enter a piece of text, and the adversarial attack will alter the GPT-2 output."  # Description of the interface
)

# Launch the interface with sharing option
if __name__ == "__main__":
    iface.launch(share=True)

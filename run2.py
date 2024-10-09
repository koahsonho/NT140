import gradio as gr
from gpt2_model2 import generate_response
from adversarial_attack import apply_adversarial_attack

def generate_response_with_whitebox_attack(user_input):
    # Áp dụng tấn công đối kháng lên đầu vào (White-box Attack)
    attacked_input = apply_adversarial_attack(user_input)
    # Sinh đầu ra từ GPT-2 với đầu vào đã bị tấn công
    gpt_output = generate_response(attacked_input)
    return attacked_input, gpt_output  # Trả về đầu vào bị tấn công và đầu ra từ GPT-2

iface = gr.Interface(
    fn=generate_response_with_whitebox_attack,
    inputs=[
        gr.Textbox(label="User Input", placeholder="Enter text here..."),  # Người dùng nhập văn bản
    ],
    outputs=[
        gr.Textbox(label="Recorded Input"),  # Ghi lại văn bản người dùng đã nhập
        gr.Textbox(label="GPT-2 Response")  # Phản hồi từ GPT-2
    ],
    title="GPT-2 Interface",  # Tiêu đề giao diện
    description="Enter a piece of text to generate a response from GPT-2 and see the recorded input."  # Mô tả giao diện
)

# Khởi động giao diện
if __name__ == "__main__":
    iface.launch(share=True)

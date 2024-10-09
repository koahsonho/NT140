import gradio as gr
from gpt2_model import generate_response  # Giả sử đây là hàm trong gpt2_model.py
from adversarial_attack import apply_whitebox_attack  # Giả sử đây là hàm trong adversarial_attack.py


def process_input(user_input):
    # Gọi hàm từ gpt2_model.py để lấy phản hồi
    gpt2_output = generate_response(user_input)

    # Gọi hàm từ adversarial_attack.py để tấn công đầu ra
    attacked_output = apply_whitebox_attack(gpt2_output)

    return attacked_output


# Khởi tạo giao diện Gradio
iface = gr.Interface(
    fn=process_input,
    inputs="text",  # Loại đầu vào
    outputs="text",  # Loại đầu ra
    title="GPT-2 Adversarial Attack Interface",  # Tiêu đề của giao diện
    description="Enter a piece of text to generate a response from GPT-2, which will then be subjected to adversarial attack."
    # Mô tả giao diện
)

# Khởi động giao diện với tùy chọn chia sẻ
if __name__ == "__main__":
    iface.launch(share=True)

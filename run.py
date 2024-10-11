import gradio as gr
from gpt2_model import generate_response
from adversarial_attack import apply_adversarial_attack


def generate_response_with_attack(user_input):
    # Ghi lại đầu vào người dùng
    recorded_input = user_input

    # Lấy phản hồi từ GPT-2 mà không tấn công vào đầu vào
    gpt_output = generate_response(user_input)

    # Áp dụng tấn công đối kháng vào đầu ra của GPT-2 (Black-box Attack)
    attacked_output = apply_adversarial_attack(gpt_output)

    # Trả về đầu vào ghi lại và đầu ra đã bị tấn công
    return recorded_input, attacked_output  # Chỉ tấn công vào đầu ra


# Tạo giao diện Gradio
iface = gr.Interface(
    fn=generate_response_with_attack,
    inputs=[gr.Textbox(label="User Input", placeholder="Enter text here...")],  # Người dùng nhập văn bản
    outputs=[  # Hiển thị 2 đầu ra
        gr.Textbox(label="Recorded Input"),  # Ghi lại văn bản người dùng đã nhập (không thay đổi)
        gr.Textbox(label="GPT-2 Response After Attack")  # Phản hồi từ GPT-2 sau khi bị tấn công
    ],
    title="GPT-2 with Black-box Attack",  # Tiêu đề giao diện
    description="Enter text, observe how a black-box adversarial attack modifies the GPT-2 response."  # Mô tả giao diện
)

# Khởi động giao diện với tùy chọn chia sẻ
if __name__ == "__main__":
    iface.launch(share=True)

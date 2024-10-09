import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Tải mô hình và tokenizer GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_response(user_input):
    # Ghi lại kết quả người dùng đã nhập
    user_input_record = user_input  # Lưu trữ đầu vào của người dùng

    # Tokenize đầu vào
    inputs = tokenizer(user_input, return_tensors='pt')

    # Sinh đầu ra từ mô hình
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)  # Sinh văn bản với độ dài tối đa là 50 từ

    # Chuyển đổi ID đầu ra thành văn bản
    gpt_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Trả về kết quả đầu ra GPT-2 và ghi lại đầu vào của người dùng
    return user_input_record, gpt_output

# Tạo giao diện Gradio với 2 inputs và 1 output
iface = gr.Interface(
    fn=generate_response,
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

# Khởi động giao diện với tùy chọn chia sẻ
if __name__ == "__main__":
    iface.launch(share=True)

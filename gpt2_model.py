import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Tải mô hình và tokenizer GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_response(user_input):
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors='pt')

    # Sinh đầu ra từ mô hình
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)  # Sinh văn bản với độ dài tối đa là 50 từ

    # Chuyển đổi ID đầu ra thành văn bản
    gpt_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return gpt_output  # Trả về đầu ra đã sinh

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=generate_response,
    inputs="text",  # Loại đầu vào
    outputs="text",  # Loại đầu ra
    title="GPT-2 Interface",  # Tiêu đề của giao diện
    description="Enter a piece of text to generate a response from GPT-2."  # Mô tả giao diện
)

# Khởi động giao diện với tùy chọn chia sẻ
if __name__ == "__main__":
    iface.launch(share=True)

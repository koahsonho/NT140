import gradio as gr
import requests

# API URL và header với token của bạn
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": "Bearer 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"}  # Token của bạn

def query(payload):
    # Gửi yêu cầu POST đến API
    response = requests.post(API_URL, headers=headers, json=payload)
    # Kiểm tra mã trạng thái của phản hồi
    if response.status_code != 200:
        return {"error": f"Request failed with status code: {response.status_code}"}
    return response.json()  # Trả về phản hồi dưới dạng JSON

def generate_response(user_input):
    # Gửi yêu cầu đến API để sinh phản hồi
    output = query({"inputs": user_input})

    # Kiểm tra kết quả trả về từ API
    if isinstance(output, list) and len(output) > 0:
        gpt_output = output[0]['generated_text']
        return user_input, gpt_output  # Trả về đầu vào và phản hồi từ GPT-2
    else:
        return user_input, "No response from the model."  # Trả về thông báo nếu không có kết quả

# Tạo giao diện Gradio với 2 ô nhập liệu và 1 ô đầu ra
iface = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(label="User Input", placeholder="Enter text here...")],  # Người dùng nhập văn bản
    outputs=[
        gr.Textbox(label="Recorded Input"),  # Hiển thị văn bản người dùng đã nhập
        gr.Textbox(label="GPT-2 Response")  # Hiển thị phản hồi từ GPT-2
    ],
    title="GPT-2 Interface",  # Tiêu đề giao diện
    description="Enter a piece of text to generate a response from GPT-2 and see the recorded input."  # Mô tả giao diện
)

# Khởi động giao diện với tùy chọn chia sẻ
if __name__ == "__main__":
    iface.launch(share=True)



import random
import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Tải GPT-2 và tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Hàm tấn công đối kháng
def add_noise(output):
    noise = ''.join(random.choices('!@#$%^&*()', k=random.randint(1, 3)))
    return output + noise

def insert_irrelevant(output):
    irrelevant_words = ['xyz', 'abc', '123', 'random', 'nonsense']
    random_word = random.choice(irrelevant_words)
    index_to_add = random.randint(0, len(output))
    return output[:index_to_add] + random_word + output[index_to_add:]

def replace_word(output):
    words = output.split()
    if words:
        index_to_replace = random.randint(0, len(words) - 1)
        words[index_to_replace] = random.choice(['dog', 'cat', 'fish', 'elephant', 'tiger'])
        return ' '.join(words)
    return output

def shuffle_words(output):
    words = output.split()
    random.shuffle(words)
    return ' '.join(words)

def add_upper(output):
    words = output.split()
    if words:
        index_to_capitalize = random.randint(0, len(words) - 1)
        words[index_to_capitalize] = words[index_to_capitalize].upper()
        return ' '.join(words)
    return output

def reverse_string(output):
    return output[::-1]

# Map các phương pháp tấn công
attack_methods = {
    "Thêm ký tự đặc biệt (Add Noise)": add_noise,
    "Chèn từ không liên quan (Insert Irrelevant)": insert_irrelevant,
    "Thay thế từ ngẫu nhiên (Replace Word)": replace_word,
    "Trộn thứ tự từ (Shuffle Words)": shuffle_words,
    "Viết hoa từ bất kỳ (Add Upper)": add_upper,
    "Đảo ngược chuỗi (Reverse String)": reverse_string
}

# Hàm xử lý với GPT-2 và tấn công
def process_input_and_attack(user_input, attack_type):
    # Hiển thị đầu vào gốc (trước khi tấn công)
    modified_input = user_input

    # Áp dụng tấn công đối kháng lên đầu vào
    if attack_type in attack_methods:
        modified_input = attack_methods[attack_type](user_input)

    # Gửi đầu vào đã tấn công đến GPT-2
    inputs = tokenizer(modified_input, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    gpt_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return user_input, modified_input, gpt_output

# Giao diện Gradio
iface = gr.Interface(
    fn=process_input_and_attack,
    inputs=[
        gr.Textbox(label="Nhập văn bản gốc (Original Input)"),
        gr.Radio(list(attack_methods.keys()), label="Chọn phương pháp tấn công (Attack Method)")
    ],
    outputs=[
        gr.Textbox(label="Đầu vào gốc (Original Input)", interactive=False),
        gr.Textbox(label="Đầu vào sau khi tấn công (Attacked Input)", interactive=False),
        gr.Textbox(label="Đầu ra GPT-2 (GPT-2 Output)", interactive=False)
    ],
    title="Giao diện GPT-2 với Tấn công đối kháng",
    description="Nhập văn bản, chọn một phương pháp tấn công, và xem kết quả GPT-2 sau khi xử lý đầu vào đã bị tấn công."
)

# Khởi chạy giao diện
if __name__ == "__main__":
    iface.launch(share=True)

import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Tải mô hình và tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()  # Đặt mô hình ở chế độ đánh giá

def apply_whitebox_attack(output):
    # Tạo token từ đầu ra
    input_ids = tokenizer.encode(output, return_tensors='pt')

    # Tạo đầu ra và xác suất từ mô hình
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

    # Kiểm tra kích thước của probabilities và input_ids
    print("Probabilities shape:", probabilities.shape)
    print("Input IDs:", input_ids)

    # Kiểm tra xem tất cả các input_ids có trong giới hạn hay không
    if (input_ids >= probabilities.size(2)).any():
        raise ValueError("Some input IDs are out of bounds for probabilities.")

    num_attacks = random.randint(100, 150)  # Số lượng tấn công
    for _ in range(num_attacks):
        attack_type = random.choice(['replace_least_prob', 'add_random', 'remove_random', 'swap_random', 'insert_noise', 'duplicate_word'])

        if attack_type == 'replace_least_prob':
            # Tìm từ có xác suất thấp nhất
            least_prob_index = torch.argmin(probabilities[0, -1, input_ids[0]])
            # Thay thế từ
            vocab_size = probabilities.size(-1)
            new_word_index = random.randint(0, vocab_size - 1)
            while new_word_index in input_ids[0]:  # Đảm bảo không thay thế bằng từ đã có
                new_word_index = random.randint(0, vocab_size - 1)
            input_ids[0, least_prob_index] = new_word_index

        elif attack_type == 'add_random':
            # Thêm một từ ngẫu nhiên vào đầu ra
            random_word_index = random.randint(0, tokenizer.vocab_size - 1)
            input_ids = torch.cat((input_ids, torch.tensor([[random_word_index]])), dim=1)

        elif attack_type == 'remove_random':
            # Xóa một từ ngẫu nhiên khỏi đầu ra
            words = input_ids[0].tolist()
            if len(words) > 1:  # Đảm bảo còn ít nhất một từ
                random_word_index = random.randint(0, len(words) - 1)
                del words[random_word_index]
                # Tạo tensor mới từ danh sách words đã được cập nhật
                input_ids = torch.tensor([words])

        elif attack_type == 'swap_random':
            # Hoán đổi vị trí của hai từ ngẫu nhiên
            words = input_ids[0].tolist()
            if len(words) > 1:  # Đảm bảo còn ít nhất hai từ
                index1, index2 = random.sample(range(len(words)), 2)
                words[index1], words[index2] = words[index2], words[index1]
                # Tạo tensor mới từ danh sách words đã được cập nhật
                input_ids = torch.tensor([words])

        elif attack_type == 'insert_noise':
            # Thêm các ký tự ngẫu nhiên vào đầu ra
            noise_length = random.randint(1, 3)
            noise = ''.join(random.choices('!@#$%^&*()', k=noise_length))
            position = random.randint(0, input_ids.size(1))
            input_ids = torch.cat((input_ids[:, :position], tokenizer.encode(noise, return_tensors='pt'), input_ids[:, position:]), dim=1)

        elif attack_type == 'duplicate_word':
            # Nhân đôi một từ ngẫu nhiên trong đầu ra
            words = input_ids[0].tolist()
            if len(words) > 0:  # Đảm bảo còn ít nhất một từ
                random_word_index = random.randint(0, len(words) - 1)
                words.insert(random_word_index, words[random_word_index])  # Nhân đôi từ
                # Tạo tensor mới từ danh sách words đã được cập nhật
                input_ids = torch.tensor([words])

    # Giải mã đầu ra mới
    new_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return new_output

# Ví dụ sử dụng
if __name__ == "__main__":
    user_input = "Hello, how are you?"
    attacked_input = apply_whitebox_attack(user_input)
    print(f"Adversarial Input: {attacked_input}")

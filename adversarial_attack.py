import random

def apply_adversarial_attack(output):
    # Apply a simple adversarial attack by altering the output string
    num_attacks = random.randint(100, 150)  # Random number of attacks
    for _ in range(num_attacks):
        attack_type = random.choice(['add_noise', 'insert_irrelevant', 'replace_word', 'shuffle_words', 'add_upper', 'reverse_string'])

        if attack_type == 'add_noise':
            noise = ''.join(random.choices('!@#$%^&*()', k=random.randint(1, 3)))  # Add random special characters
            output += noise

        elif attack_type == 'insert_irrelevant':
            irrelevant_words = ['xyz', 'abc', '123', 'random', 'nonsense']
            random_word = random.choice(irrelevant_words)
            index_to_add = random.randint(0, len(output))
            output = (output[:index_to_add] + random_word + output[index_to_add:])

        elif attack_type == 'replace_word':
            words = output.split()
            if words:
                index_to_replace = random.randint(0, len(words) - 1)
                words[index_to_replace] = random.choice(['dog', 'cat', 'fish', 'elephant', 'tiger'])  # Replace with a random animal
                output = ' '.join(words)

        elif attack_type == 'shuffle_words':
            words = output.split()
            random.shuffle(words)  # Shuffle the order of words in the output
            output = ' '.join(words)

        elif attack_type == 'add_upper':
            # Randomly capitalize a word in the output string
            words = output.split()
            if words:
                index_to_capitalize = random.randint(0, len(words) - 1)
                words[index_to_capitalize] = words[index_to_capitalize].upper()
                output = ' '.join(words)

        elif attack_type == 'reverse_string':
            output = output[::-1]  # Reverse the entire string

    return output

# Example usage
if __name__ == "__main__":
    user_input = "Hello, how are you?"
    attacked_input = apply_adversarial_attack(user_input)
    print(f"Adversarial Input: {attacked_input}")

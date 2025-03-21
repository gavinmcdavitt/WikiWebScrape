
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time


# Function to read content from a file
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except Exception as e:
        return f"Error reading file: {e}"


# Function to generate an answer using T5
def generate_answer_with_t5(question, file_content, tokenizer, model):
    # Start timing the process
    start_time = time.time()

    # Tokenize the question and the content from the file (T5 input format: "question: {question} context: {context}")
    input_text = f"question: {question} context: {file_content}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Move model and input_ids to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)

    # Generate an answer using the model
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, num_beams=4, early_stopping=True)

    # Record the end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for generation: {elapsed_time:.2f} seconds")

    # Decode and return the generated answer
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Main function to simulate the question-answering system
def answer_question(question, file_path, tokenizer, model):
    # Read the file content
    file_content = read_file(file_path)

    # Check if there was an error reading the file
    if file_content.startswith("Error"):
        return file_content

    # Generate answer using T5 model
    answer = generate_answer_with_t5(question, file_content, tokenizer, model)
    return answer


# Example user input
user_question = "How many super bowls did tom brady lose?"
file_path = "tom_brady.txt"  # Replace with the actual file path

# Load the tokenizer and model (T5)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Generate and print the answer
answer = answer_question(user_question, file_path, tokenizer, model)
print(answer)
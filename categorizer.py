import ollama
import os

model = "llama3.2:latest"

# path to input & output files
input_file = "data\grocery_list.txt"
output_file = "data\categorized_grocery_list.txt"

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Input file '{input_file}' does not exist.")
    exit(1)

# Read the grocery list from the input file
with open(input_file, "r") as f:
    items = f.read().strip()

# Prepare the prompt for categorization
prompt = f"""
    You are an assistant that categorize and sorts grocery items.

    Here is the list of grocery items:
    {items}

    Please

    1. Categorize these items into appropriate categories such as Fruits, Vegetables, Dairy, Meat, Grains, Snacks, Beverages, etc.
    2. Sort the items within each category alphabetically.
    3. Present the categorized list in a clear format.

"""

# Send the prompt and get the response from the model

try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print("==== Categorized Grocery List ====")
    print("Generated Text:\n", generated_text)

    # Write the categorized list to the output file
    with open(output_file, "w") as f:
        f.write(generated_text.strip())

    print(f"Categorized grocery list saved to '{output_file}'")

except Exception as e:
    print(f"An error occurred: {e}", str(e))

import base64

# Step 1: Read the base64 text
with open("fer_model_base64.txt", "r") as f:
    base64_data = f.read()

# Step 2: Decode back to binary
binary_data = base64.b64decode(base64_data)

# Step 3: Save as .h5
with open("fer_model.h5", "wb") as f:
    f.write(binary_data)

print("✅ Model file created: fer_model.h5")

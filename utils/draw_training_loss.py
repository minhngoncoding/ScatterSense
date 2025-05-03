import matplotlib.pyplot as plt
import re

# Path to your text file
file_path = "outputs/train_loss.txt"

# Initialize lists to store iterations and loss values
iterations = []
loss_values = []

# Read and parse the file
with open(file_path, "r") as file:
    for line in file:
        match = re.match(r"Iteration (\d+): Log Loss = ([\d.]+)", line)
        if match:
            iterations.append(int(match.group(1)))
            loss_values.append(float(match.group(2)))

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_values, label="Training Loss", color="blue")
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("Training Loss Over Iterations")
plt.legend()
plt.grid(True)
plt.show()

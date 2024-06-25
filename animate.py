import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np

# Ensure the right backend is set for saving files
matplotlib.use("Agg")


# Read the JSON Lines file
# Read the JSON Lines file
def read_json_lines(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_line = json.loads(line)
            data.append((json_line["text"], json_line["step"]))
    return data


# Initialize the grid for words
def initialize_word_grid(ax, num_words):
    text_elements = []
    num_cols = 64
    for i in range(num_words):
        x = i % num_cols  # Example: 32 words per line
        y = i // num_cols
        text_element = ax.text(
            x, -y, "", ha="center", va="center", color="black", fontsize=8
        )
        text_elements.append(text_element)
    return text_elements


# Update the animation
def update(frame_number, data, text_elements, last_words):
    words, step = data[frame_number]
    for i, (word, text_element) in enumerate(zip(words, text_elements)):
        if last_words[i] != word:
            text_element.set_color("red")
        else:
            text_element.set_color("black")
        if word == "<M>":
            text_element.set_text("")
        else:
            text_element.set_text(word[:30])  # Limit to 30 characters
    last_words[:] = words  # Update last words
    return text_elements


# Create the animation and save it
def create_and_save_animation(data, file_name):
    fig, ax = plt.subplots(figsize=(32, 16))
    ax.set_xlim(-1, 64)  # Adjust based on your layout
    ax.set_ylim(-16, 1)  # assuming 1024/64 = 16 lines
    ax.axis("off")

    # Enable grid
    ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, 64, 1), minor=True)
    ax.set_yticks(np.arange(-15.5, 1, 1), minor=True)

    text_elements = initialize_word_grid(ax, 1024)
    last_words = [""] * 1024  # Initialize last words

    ani = FuncAnimation(
        fig,
        update,
        frames=len(data),
        fargs=(data, text_elements, last_words),
        interval=1000,  # 1000 ms per frame
        repeat=False,
        blit=True,
    )

    # Save the animation
    ani.save(file_name, writer="ffmpeg", fps=1)


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Description")

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="json lines file",
        required=True,
    )
    parser.add_argument(
        "--output", type=str, default=None, help="output file name"
    )

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output or Path(input_file).with_suffix(".mp4")
    data = read_json_lines(input_file)  # Update with your file path
    create_and_save_animation(data, str(output_file))  # Saves as an MP4 file

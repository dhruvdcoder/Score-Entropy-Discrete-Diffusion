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
    num_cols = 2 * int(
        np.ceil(np.sqrt(num_words))
    )  # if 1024 words, 64 words per line
    for i in range(num_words):
        x = i % num_cols  # Example: 32 words per line
        y = i // num_cols
        font_size = max(8, 400 // num_cols)
        text_element = ax.text(
            x,
            -y,
            "",
            ha="center",
            va="center",
            color="black",
            fontsize=font_size,
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
def create_and_save_animation(data, file_name, max_length=1024):
    fig, ax = plt.subplots(figsize=(32, 16))
    num_cols = 2 * int(
        np.ceil(np.sqrt(max_length))
    )  # if 1024 words, 64 words per line
    num_rows = (max_length + num_cols - 1) // num_cols
    ax.set_xlim(-1, num_cols)  # Adjust based on your layout
    ax.set_ylim(-num_rows, 1)  # assuming 1024/64 = 16 lines
    ax.axis("off")

    # Enable grid
    ax.grid(True, which="both", color="black", linestyle="-", linewidth=2)
    ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-num_rows + 0.5, 1, 1), minor=True)

    text_elements = initialize_word_grid(ax, max_length)
    last_words = [""] * max_length  # Initialize last words

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
    print(f"Saving animation to {file_name}")
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
    # outputs_<max_length>_<steps>.json
    input_file = args.input
    max_len, num_steps = [
        int(v)
        for v in str(Path(input_file).with_suffix("").name).split("_")[1:]
    ]
    output_file = args.output or Path(input_file).with_suffix(".mp4")
    data = read_json_lines(input_file)  # Update with your file path
    create_and_save_animation(
        data, str(output_file), max_length=max_len
    )  # Saves as an MP4 file

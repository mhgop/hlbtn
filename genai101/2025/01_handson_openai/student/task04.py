# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])

# Compile regex to match any whitespace except newlines
re_whitespace_except_newline = re.compile(r"[^\S\n]+")

# Replace matched whitespace with an empty string
def normalize_string(input_string):
    normalized = re_whitespace_except_newline.sub('', input_string)
    return normalized

# Compile regex to extract tuples from a string
re_tuple = re.compile(r"\(\s*'([^']*)'\s*,\s*'([^']*)'\s*\)")

# Parse table
def parse_table_string(table_string):
    # Split the rows based on newline characters
    rows = table_string.strip().split('\n')
    table = []
    for row in rows:
        # Extract tuples using regular expressions
        cells = re_tuple.findall(row)
        # Convert each tuple to a tuple of (color, marker) and add to the row list
        table.append([(color, marker) for color, marker in cells])
    return table
    
# Print table
def show_tables(table1, table2):
    D = len(table1)

    # Create the figure and a grid of subplots
    fig, ax = plt.subplots(D, D, figsize=(D, D))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Loop over each cell to populate the grid
    for i in range(D):
        for j in range(D):
            marker1, color1 = table1[i][j][1], table1[i][j][0]
            marker2, color2 = table2[i][j][1], table2[i][j][0]
            
            # Check if markers agree
            if (color1 == color2) and (marker1 == marker2):
                # Plot one marker in the center
                ax[i, j].scatter(0.5, 0.5, c=color1, marker=marker1, s=200)
            else:
                # Add a light red background for disagreements and plot both markers
                ax[i, j].add_patch(patches.Rectangle((0, 0), 1, 1, color='lightcoral', alpha=0.3))
                ax[i, j].scatter(0.25, 0.5, c=color1, marker=marker1, s=200)
                ax[i, j].scatter(0.75, 0.5, c=color2, marker=marker2, s=200)

            # Set axis to "on" but remove ticks
            ax[i, j].axis('on')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    # Label the sides of the grid for all rows and columns
    for i in range(D):
        # Left and right Y-axis labels
        ax[i, 0].set_ylabel(i, rotation=0, labelpad=20, va='center')
        ax[i, D - 1].yaxis.set_label_position("right")
        ax[i, D - 1].set_ylabel(i, rotation=0, labelpad=20, va='center')
        
        # Top and bottom X-axis labels
        ax[0, i].xaxis.set_label_position("top")
        ax[0, i].set_xlabel(i, labelpad=10, ha='center')
        ax[D - 1, i].set_xlabel(i, labelpad=10, ha='center')
        
    plt.savefig(f"comparison_{D}x{D}_{model_name}.jpeg", bbox_inches='tight', pad_inches=0)
    plt.close


# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---


# --- --- --- TESTING CODE --- --- ---

rootfolder = "assets/inputs"

if __name__ == "__main__":
    nb_retry = 3
    for i in range(2,13):
        # Retry loop
        for r in range(1, nb_retry+1):
            try:
                # Image path and call
                ipath = rootfolder + f"/{i}x{i}.jpeg"
                ai_table = call_with(i, ipath)

                # Load groud truth text
                tpath = rootfolder + f"/{i}x{i}.txt"
                text = load_text(tpath)

                # Visually compare the tables
                table1 = parse_table_string(text)
                table2 = parse_table_string(ai_table)
                show_tables(table1, table2)
                
                # Normalise responses and compare, show table
                text = normalize_string(text)
                ai_table = normalize_string(ai_table)
                print(i, '->', text==ai_table)
                break # break out of retry loop
            except Exception as e:
                print(f"Error at {i}x{i} - try {r}")
        # After retry
        if r == nb_retry:
            print(f"Error at {i}x{i}: failed {nb_retry} times")
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import cycle

# Load and parse the file
def parse_log_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()
    
    loss_values = []
    texts = []
    
    epoch = list(data.split("\n"))
    i =0
    while i < len(epoch):
        d = epoch[i]
        print(d)
        match = re.search(r"Avg\. Loss: ([0-9]+\.[0-9]+)", d)
        if match:
            loss_values.append(float(match.group(1)))
        
        if d.startswith("Hello"):
            temp = d
            while not d.startswith("Epoch"):
                i+=1
                d = epoch[i]
                if i >= len(epoch) or d.startswith("Epoch"):
                    break
                
                
                temp = temp + "\n" + d
            texts.append(temp)    
        else:
            i+=1
                
                
    
    return loss_values, texts

# Load data
filename = "train.txt"  # Change to your actual file
loss_values, texts = parse_log_file(filename)


import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import textwrap
from itertools import cycle


time_steps = list(range(1, len(loss_values) + 1))

# Setup figure and animation
fig, ax = plt.subplots()
ax.set_xlim(1, len(loss_values))
ax.set_ylim(min(loss_values) - 0.5, max(loss_values) + 0.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss Over Time")

line, = ax.plot([], [], 'bo-', lw=2)
loss_text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), wrap=True)
inference_text_box = ax.text(0.05, 0.75, "", transform=ax.transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), wrap=True)

def init():
    line.set_data([], [])
    loss_text_box.set_text("")
    inference_text_box.set_text("")
    return line, loss_text_box, inference_text_box

def update(frame):
    line.set_data(time_steps[:frame + 1], loss_values[:frame + 1])
    loss_text_box.set_text(f"Loss: {loss_values[frame]:.4f}")
    inference_text_box.set_text(texts[frame])
    return line, loss_text_box, inference_text_box

ani = animation.FuncAnimation(fig, update, frames=len(loss_values), init_func=init, interval=100, repeat=False)

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, filedialog, StringVar, Frame
from PIL import Image, ImageTk, ImageSequence
from ises_filter import ISESFilter




def run_filter():
   
    try:
        patch_size = int(patch_size_entry.get()) if patch_size_entry.get() else 5
      
        if patch_size % 2 == 0:
            patch_size += 1
    except ValueError:
        patch_size = 5

    try:
        edge_param = float(edge_param_entry.get()) if edge_param_entry.get() else 10
    except ValueError:
        edge_param = 10

    try:
        reg_param = float(reg_param_entry.get()) if reg_param_entry.get() else 1e-5
    except ValueError:
        reg_param = 1e-5

   
    if not image_path_var.get():
        print("No image selected. Please select an image.")
        return

    image_path = image_path_var.get()
    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if x is None:
        print(f"Error: Image '{image_path}' not found.")
        return

   
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    
    z = ISESFilter(x, patch_size, edge_param, reg_param)

    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(x)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(z)
    axes[1].set_title("Filtered Image (ISES Filter)", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    plt.tight_layout(pad=2.0)
    plt.show()



def select_image():
   
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif"), ("All Files", "*.*")]
    )
    if file_path:
        image_path_var.set(file_path)
        image_label.config(text=file_path)



def update_gif(frame_index):
    
    frame = frames[frame_index]
    gif_label.config(image=frame)
    frame_index = (frame_index + 1) % len(frames)
    root.after(100, update_gif, frame_index)




root = Tk()
root.title("ISES Filter User Interface")
root.configure(bg="white")



try:
    pil_image = Image.open("E:\github\Adaptive-Edge-Preserving-Smoothing\standard_test_images\original.gif")
    frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA"), master=root) 
              for frame in ImageSequence.Iterator(pil_image)]
except Exception as e:
    print("Error loading GIF image:", e)
    frames = None


if frames:
    gif_frame = Frame(root, bg="white")
    gif_frame.pack(pady=10)
    gif_label = Label(gif_frame, bg="white")
    gif_label.pack()
 
    root.after(0, update_gif, 0)


image_path_var = StringVar()


image_frame = Frame(root, bg="white")
image_frame.pack(pady=10)
Button(image_frame, text="Select Image", command=select_image, bg="white").pack()
image_label = Label(image_frame, text="No image selected", wraplength=300, bg="white")
image_label.pack()


params_frame = Frame(root, bg="white")
params_frame.pack(pady=10)


Label(params_frame, text="Patch Size (Default: 5, must be odd):", bg="white").grid(row=0, column=0, sticky="w")
patch_size_entry = Entry(params_frame)
patch_size_entry.grid(row=0, column=1, padx=5, pady=5)


Label(params_frame, text="Edge Sharpening Parameter (Default: 10):", bg="white").grid(row=1, column=0, sticky="w")
edge_param_entry = Entry(params_frame)
edge_param_entry.grid(row=1, column=1, padx=5, pady=5)

Label(params_frame, text="Regularization Parameter (Default: 1e-5):", bg="white").grid(row=2, column=0, sticky="w")
reg_param_entry = Entry(params_frame)
reg_param_entry.grid(row=2, column=1, padx=5, pady=5)

apply_button = Button(root, text="Apply ISES Filter", command=run_filter, bg="light green")
apply_button.pack(pady=10)
Label(root, text="Created by Mohammad Reza Mansouri", bg="white", fg="black", font=("Arial", 10)).pack(pady=10)



root.mainloop()

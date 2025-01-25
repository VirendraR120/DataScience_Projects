import torch
import numpy as np
from torch import autocast
import tkinter as tk
from tkinter import PhotoImage
import customtkinter as ctk
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline, DiffusionPipeline, FluxPipeline
from authtoken import auth_token


# Variables
device = "cuda"


# Create app
app = ctk.CTk()
app.geometry("532x632")
app.title("Image Generator")
#app.attributes("-alpha", "0.1")
ctk.set_appearance_mode("dark")

bg_image_path = PhotoImage(file = "E:/My Files/Wallpapers/red_topographic.png")



model_compvis= "CompVis/stable-diffusion-v1-4"
model_flux = "black-forest-labs/FLUX.1-dev"
# Modeling and Image Generation


# model_map = {
#     "None": model_ids[0],
#     "Stable Diffusion-1.4": model_ids[1],
#     "Stable Diffusion-XL-1.0": model_ids[2],
#     "Hakurei-Waifu Diffusion": model_ids[3],
#     "Odyssey-XL-4.0": model_ids[4] 
#}




#selected_model = model_combobox.get()
pipe = StableDiffusionPipeline.from_pretrained(model_compvis, variant="fp16", torch_dtype=torch.float16, use_auth_token = auth_token, low_cpu_mem_usage=False)
#pipe = FluxPipeline.from_pretrained(model_flux, use_auth_token = auth_token, guidance_scale=3.5, torch_dtype=torch.float16)
pipe.to(device)
print(f"Pipe object type: {type(pipe)}")
   
   
def generate():
    user_prompt = prompt_textbox.get("0.0", ctk.END).strip() # for user prompt
    if user_prompt:
        # user_prompt += "using model: " + model_combobox.get()
        print(f"\n User prompt: {user_prompt}")
        with autocast(device):
            image = pipe(prompt_textbox.get("0.0", ctk.END), guidance_scale=5, height = 800, width = 1000).images[0]
            #print(image)

        img = ImageTk.PhotoImage(image)
        image.save("generatedimage.png")
        canvas.image = img
        canvas.create_image(800, 500, image=img, anchor = "center")
        # image_label.configure(image=img)
        #print(f"{img} is generated using: {model_id}")
    else:
        print("Please enter a prompt!")


#------ The Frame ------
frame1 = ctk.CTkFrame(app, height = 1, width = 1, fg_color="#202020", corner_radius = 25)
frame1.pack(side="left", fill="none", expand = False, padx = 10, pady = 10)

# bg_label = ctk.CTkLabel(frame1, image = bg_image_path, text="")
# bg_label.place(relheight = 1, relwidth = 1)



#------ Input fields ------
# Prompt - Label & Textbox
prompt_label = ctk.CTkLabel(frame1, height = 10, width = 80, text = "Enter your idea:", text_color= "White", font = ("Times New Roman", 20), bg_color="transparent", corner_radius=30)
prompt_label.grid(row = 0, column = 0, padx = 10, pady = 20, sticky = "news")

prompt_textbox = ctk.CTkTextbox(frame1, height = 100, width = 250, 
                          fg_color = "#292929", corner_radius = 10, scrollbar_button_color = "#AD7040", scrollbar_button_hover_color = "orange",
                          border_color = "white", font = ("Times New Roman", 17), wrap = "word")
prompt_textbox.grid(row = 0, column = 1, padx = 10, pady = 20)



#------ Art Style selection ------
# Label
style_label = ctk.CTkLabel(frame1, height = 40, width = 120, text = "Model:", text_color= "White", font = ("Times New Roman", 20))
style_label.grid(row = 2, column = 0, padx = 10, pady = 20)
# Combo Box
model_optionmenu = ctk.CTkOptionMenu(frame1, height = 30, width = 80, corner_radius = 10, dynamic_resizing = True,
                                 values=["Stable Diffusion-1.4"], font = ("Times New Roman", 16), anchor = "center",
                                 fg_color = "#292929", button_color = "#AD7040", button_hover_color="orange",
                                 dropdown_hover_color = "#AD7040", dropdown_font=("Times New Roman", 16))
model_optionmenu.grid(row = 2, column = 1, padx = 10, pady = 20, sticky = "news")


# #------ Slider for image generation count ------
# # Label
# image_count_label = ctk.CTkLabel(frame1, height = 40, width = 120, text = "Number of images:", text_color= "White", font = ("Times New Roman", 20))
# image_count_label.grid(row = 3, column = 0, padx = 10, pady = 10) 
# # Slider
# image_count_slider = ctk.CTkSlider(frame1, from_ = 1, to = 5, number_of_steps = 4, button_color = "#AD7040", button_hover_color = "orange")
# image_count_slider.set(1)
# image_count_slider.grid(row = 3, column = 1, padx = 10, pady = 10)


#------ Generation Button ------
generate_button = ctk.CTkButton(frame1, height = 50, width = 100,
                      text = "Generate!", font = ("Monotype Corsiva", 25, "bold"),
                      corner_radius=100, hover_color = "#AD7040", anchor = "center",
                      fg_color = "transparent", bg_color = "transparent", border_color="#AD7040", border_width = 1, command = generate)
generate_button.grid(row = 4, column = 0, padx = 20, pady = 20, columnspan = 2, sticky = "news")


#------ Canvas for Generated image ------
frame2 = ctk.CTkFrame(app, height = 1000, width = 1410, fg_color="#1F1F1F", corner_radius = 30)
frame2.pack(side="top", fill="none", expand = True, padx = 10, pady = 10)

img_loc_label = ctk.CTkLabel(frame2, height = 40, width = 120, text = "Your image will generate here", text_color= "#666666", font = ("Times New Roman", 20))
img_loc_label.place(x=670, y=460)

canvas =  tk.Canvas(frame2, height = 980, width = 1400, bg = "#242424", highlightthickness=0)
canvas.pack(side="left")


# image_label = ctk.CTkLabel(frame2, height = 980, width = 1400)
# image_label.pack(side="left")
app.mainloop()
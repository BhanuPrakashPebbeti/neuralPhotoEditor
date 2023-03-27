import cv2
import datetime
import numpy as np
import warnings
import pygame
import pygame_widgets
import pygame.camera
import pygame.image
import sys
import time
import tkinter
import tkinter.filedialog
from tkinter import colorchooser
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from model import *
# from vcolorpicker import getColor
warnings.filterwarnings("ignore")

class Button:
    def __init__(self,text,width,height,pos,elevation,button_font):
		#Core attributes 
        self.pressed = False
        self.elevation = elevation
        self.dynamic_elecation = elevation
        self.original_y_pos = pos[1]

		# top rectangle 
        self.top_rect = pygame.Rect(pos,(width,height))
        self.top_color = '#475F77'

		# bottom rectangle 
        self.bottom_rect = pygame.Rect(pos,(width,height))
        self.bottom_color = '#354B5E'
		#text
        self.button_font = button_font
        self.text_surf = button_font.render(text,True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

    def draw(self,response):
        # elevation logic 
        self.top_rect.y = self.original_y_pos - self.dynamic_elecation
        self.text_rect.center = self.top_rect.center 
        self.bottom_rect.midtop = self.top_rect.midtop
        self.bottom_rect.height = self.top_rect.height + self.dynamic_elecation
        pygame.draw.rect(screen,self.bottom_color, self.bottom_rect,border_radius = 12)
        pygame.draw.rect(screen,self.top_color, self.top_rect,border_radius = 12)
        screen.blit(self.text_surf, self.text_rect)
        return self.check_click(response)

    def check_click(self,system):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            self.top_color = '#D74B4B'
            if pygame.mouse.get_pressed()[0]:
                self.dynamic_elecation = 0
                self.pressed = True
            else:
                self.dynamic_elecation = self.elevation
                if self.pressed == True:
                    self.pressed = False
                    system = True
        else:
            self.dynamic_elecation = self.elevation
            self.top_color = '#475F77'
        return system
    
def cv2ImageToSurface(cv2Image):
    if cv2Image.dtype.name == 'uint16':
        cv2Image = (cv2Image).astype('uint8')
    size = cv2Image.shape[1::-1]
    if len(cv2Image.shape) == 2:
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
    surface = pygame.image.frombuffer(cv2Image.flatten(), size, "RGB")
    return surface.convert()

def prompt_file():
    """Create a Tk file dialog and cleanup when finished"""
    top = tkinter.Tk()
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.askopenfilename(parent=top)
    top.destroy()
    return file_name
    
print()
print("                ==========================================================================    ")
print("               |                             Starting Application                         |   ")
print("                ==========================================================================    ")
print()
print()
pygame.init()
pygame.font.init()
clock = pygame.time.Clock()
button_font = pygame.font.Font(None,30)
STAT_FONT = pygame.font.SysFont("comicsans", 40)
screen = pygame.display.set_mode((750, 500))
pygame.display.set_caption("Application")

button1 = Button('Upload',200,40,(525,50),5,button_font)
button2 = Button('Draw',200,40,(525,125),5,button_font)
button3 = Button('Revert Changes',200,40,(525,200),5,button_font)
button4 = Button('Enhance',200,40,(525,275),5,button_font)
button5 = Button('Save',200,40,(525,350),5,button_font)
slider = Slider(screen, 525, 440, 200, 10, min=1, max=20, step=1, initial=1, handleColour=(255,255,0))
output = TextBox(screen, 605, 460, 40, 35, fontSize=30)

output.disable()

input_size = (400,400)
run = True
upload = False
draw = False
drawing = False
save = False
cancel = False
revert = False
editframe = None
enhance = False
originalframe = None
original_rgbframe = None
enhancedFrame = None
enhancedImage = None

logo = cv2.imread("Logo.png")
logo = cv2.resize(logo,input_size)
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
rgbframe = cv2ImageToSurface(logo)

buttonbg = cv2.imread("buttonBg.jpg")
buttonbg = cv2.cvtColor(buttonbg, cv2.COLOR_BGR2RGB)
buttonbg = cv2ImageToSurface(buttonbg)

encoder = Encoder(input_channels, time_embedding, block_layers=[2, 2, 2, 2])
decoder = Decoder(last_fmap_channels, output_channels, time_embedding, first_fmap_channels)
model = DiffusionNet(encoder, decoder)

#diffusion utilities class initialisaion
diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, DEVICE, scheduler=beta_scheduler)
trainer = Trainer(model, diffusion_utils, device=DEVICE, pretrained = True)

while run:
    events = pygame.event.get()
    for e in events:
        if e.type == pygame.QUIT:
            run = False
            pygame.quit()
            quit()
            break
        if pygame.mouse.get_pressed()[0] and drawing:
            pos = pygame.mouse.get_pos()
            pos = (pos[0]-50, pos[1]-50)
            editframe = cv2.circle(editframe, pos, radius, color_code[0], -1)
            rgbframe = cv2ImageToSurface(editframe)

    screen.fill((0,0,0))
    screen.blit(buttonbg, (500,0))
    upload = button1.draw(upload)
    draw = button2.draw(draw)
    revert = button3.draw(revert)
    enhance = button4.draw(enhance)
    save = button5.draw(save)

    if upload:
        f = prompt_file()
        if f:
            frame = cv2.imread(f)
            frame = cv2.resize(frame,input_size)
            editframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            originalframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgbframe = cv2ImageToSurface(editframe)
            original_rgbframe = cv2ImageToSurface(originalframe)
            upload = False

    if draw:
        color_code = colorchooser.askcolor(title="Choose color")
        draw = False
        drawing = True
        # color = getColor()
        if editframe is None:
            f = prompt_file()
            upload = False
            frame = cv2.imread(f)
            frame = cv2.resize(frame,input_size)
            editframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            originalframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgbframe = cv2ImageToSurface(editframe)
            original_rgbframe = cv2ImageToSurface(originalframe)

    if revert and (original_rgbframe is not None) and (originalframe is not None):
        rgbframe = original_rgbframe.copy()
        editframe = originalframe.copy()
        revert = False

    if enhance and (originalframe is not None) and (editframe is not None):
        difference = cv2.subtract(editframe, originalframe)
        Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
        mask[mask != 255] = 1
        mask[mask == 255] = 0
        enhance = False
        enhancedImage = trainer.addNoiseandMaskedDenoise(400,editframe,mask)
        enhancedFrame = cv2ImageToSurface(enhancedImage)

    if save and (enhancedImage is not None):
        filename = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        cv2.imwrite(str(filename.name), cv2.cvtColor(enhancedImage, cv2.COLOR_BGR2RGB))
        run = False
        save = False
        pygame.quit()
        quit()
        run = True
        upload = False
        draw = False
        drawing = False
        save = False
        cancel = False
        revert = False
        editframe = None
        enhance = False
        originalframe = None
        original_rgbframe = None
        enhancedFrame = None
        enhancedImage = None
        rgbframe = cv2ImageToSurface(logo)
    
    if enhancedFrame is not None:
        screen.blit(enhancedFrame, (122,122))
    else:
        screen.blit(rgbframe, (50,50))

    radius = slider.getValue()
    output.setText(radius)
    pygame_widgets.update(events)
    pygame.display.update()
    clock.tick(300)


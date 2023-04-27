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
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis=2)
    elif len(cv2Image.shape) == 3 and cv2Image.shape[2] == 1:
        cv2Image = np.repeat(cv2Image, 3, axis=2)
    surface = pygame.image.frombuffer(cv2Image.flatten(), size, "RGB")
    return surface.convert()

def prompt_file():
    """Create a Tk file dialog and cleanup when finished"""
    top = tkinter.Tk()
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.askopenfilename(parent=top)
    top.destroy()
    return file_name

PALETTE = {
    ( 70, 70, 70)  : 0 ,
    (250,170,160) : 1 ,
    ( 81,  0, 81) : 2 ,
    (244, 35,232) : 3 , 
    (220, 190, 40) : 4 , 
    (152,251,152) : 5 ,
    (220, 20, 60) : 6 ,
    (246, 198, 145) : 7 ,
    (255,  0,  0) : 8 , 
    (  0,  0,230) : 9 ,
    (119, 11, 32) : 10 , 
}

def convert2RGB(array):
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for value, key in PALETTE.items():
        result[np.where(array == key)] = (value[2],value[1],value[0])
    return result.astype('uint8')
    
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
screen = pygame.display.set_mode((1125, 900))
pygame.display.set_caption("Application")

button1 = Button('Upload',200,40,(900,50),5,button_font)
button2 = Button('Draw',200,40,(900,125),5,button_font)
button3 = Button('Revert Changes',200,40,(900,200),5,button_font)
button4 = Button('Enhance',200,40,(900,275),5,button_font)
button5 = Button('Save',200,40,(900,350),5,button_font)
slider = Slider(screen, 900, 440, 200, 10, min=1, max=20, step=1, initial=5, handleColour=(255,255,0))
output = TextBox(screen, 980, 460, 40, 35, fontSize=30)
output.disable()

tslider = Slider(screen, 900, 525, 200, 10, min=200, max=475, step=1, initial=400, handleColour=(255,255,0))
toutput = TextBox(screen, 980, 550, 50, 35, fontSize=30)
toutput.disable()

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
mask = None

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
seg_model = UNET(in_channels = 3 , out_channels = 11)
generator = GeneratorResNet()

#diffusion utilities class initialisaion
diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, DEVICE, scheduler=beta_scheduler)
trainer = Trainer(model, diffusion_utils,seg_model, generator, device=DEVICE, pretrained = True)

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
            pos = (pos[0]-25, pos[1]-50)
            editframe = cv2.circle(editframe, pos, radius, color_code[0], -1)
            DiffusionMask = cv2.circle(DiffusionMask, pos, radius, (255,255,255), -1)
            rgbframe = cv2ImageToSurface(editframe)

        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            if ((pos[0] > 450) and (pos[1] > 50)) and ((pos[0] < 850) and (pos[1] < 450)):
                pos = (pos[0]-450, pos[1]-50)
                index = segMask[pos[1],pos[0]]
                editframe,DiffusionMask = trainer.applyColor(editframe,segMask,index, color_code[0],DiffusionMask)
                rgbframe = cv2ImageToSurface(editframe)

    screen.fill((0,0,0))
    screen.blit(buttonbg, (875,0))
    upload = button1.draw(upload)
    draw = button2.draw(draw)
    revert = button3.draw(revert)
    enhance = button4.draw(enhance)
    save = button5.draw(save)

    if upload:
        f = prompt_file()
        if f:
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
            mask = None
            frame = cv2.imread(f)
            frame = cv2.resize(frame,input_size)
            editframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            originalframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            DiffusionMask = np.zeros((input_size[0],input_size[1],3), np.uint8)
            rgbframe = cv2ImageToSurface(editframe)
            original_rgbframe = cv2ImageToSurface(originalframe)
            mask = trainer.segment(originalframe)
            segMask = cv2.resize(mask, (originalframe.shape[1], originalframe.shape[0]),interpolation=cv2.INTER_NEAREST)
            mask = convert2RGB(segMask)
            mask = cv2ImageToSurface(mask)
            upload = False

    if draw:
        color_code = colorchooser.askcolor(title="Choose color")
        draw = False
        drawing = True
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
        DiffusionMask = sum(cv2.split(DiffusionMask))/3
        DiffusionMask[DiffusionMask != 255] = 0
        DiffusionMask[DiffusionMask == 255] = 1
        DiffusionMask = DiffusionMask.astype('float32')
        enhance = False
        enhancedImage,superImg = trainer.addNoiseandMaskedDenoise(tstamp,editframe,DiffusionMask)
        enhancedFrame = cv2ImageToSurface(enhancedImage)
        superImg = cv2ImageToSurface(superImg)
        enhancedImageScales = cv2.resize(enhancedImage,(400,400))
        enhancedImageScales = cv2ImageToSurface(enhancedImageScales)

    if save and (enhancedImage is not None):
        filename = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        cv2.imwrite(str(filename.name), cv2.cvtColor(enhancedImage, cv2.COLOR_BGR2RGB))
        run = False
        save = False
        pygame.quit()
        quit()

    if enhancedFrame is not None:
        screen.blit(rgbframe, (25,50))
        screen.blit(enhancedFrame, (586,186))
        screen.blit(superImg, (25,475))
        screen.blit(enhancedImageScales, (450,475))
    else:
        screen.blit(rgbframe, (25,50))
        if mask is not None:
            screen.blit(mask, (450,50))

    radius = slider.getValue()
    output.setText(radius)
    tstamp = tslider.getValue()
    toutput.setText(tstamp)
    pygame_widgets.update(events)
    pygame.display.update()
    clock.tick(300)


import csv
import uuid
import json
import urllib.request
import urllib.parse
import random
import numpy as np
from transformers import pipeline
import re
import io
import os
import base64
from flask import render_template
from flask import Flask, request, jsonify
from flask import Request
import secrets
from flask_cors import CORS
from PIL import Image
import math
import csv
import websocket
from aigcserver import get_images
import GptTagger
import requests
import torch
import cv2
from ultralytics import YOLO
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
style_models = {
    
    "0":"AFK_Normal_SD1.5.ckpt",
    "1":"Farlight84_Normal_SD1.5.safetensors",
    "2":"Dgame_Normal_SD1.5.safetensors",
    "3":"ROK_Normal_SD1.5_V1.ckpt",
    "4":"cyberrealistic_v33.safetensors",
    "5":"Igame_Normal_SD1.5.ckpt",
    "6":"Igame_Character_SDXL.safetensors",
    "7":"AFK_Normal_SDXL.safetensors",
    "8":"SAMO_Normal_SDXL.safetensors",
    "9":"Party_Scene_SDXL.safetensors"
}

style_models_prepositive = {
    
    "0":"(masterpiece, best quality),AFK,",
    "1":"(masterpiece:1.2), (best quality, highest quality),fl,realistic,",
    "2":"(masterpiece:1.2), (best quality, highest quality),xbb,",
    "3":",((best quality)), ((masterpiece)),ROK,",
    "4":"(masterpiece:1.2), (best quality, highest quality),AFK,",
    "5":"((best quality))",
    "6":"trq style",
    "7":"trq style",
    "8":"trq style",
    "9":"trq style",
}

style_models_prenegative = {
    "0":"embedding:badoutV2.pt,",
    "1":"embedding:EasyNegative.pt,",  
    "3":"embedding:EasyNegative.pt,",
    "2":"embedding:EasyNegative.pt,",
    "4":"embedding:EasyNegative.pt,",
    "5":"embedding:EasyNegative.pt,",
    "6":"lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    "7":"lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    "8":"lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    "9":"lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
}

def post_image(url, image_file_path):
    # 打开图片文件
    with open(image_file_path, 'rb') as file:
        # 构建表单数据
        files = {'image': file}
        # 发送POST请求
        response = requests.post(url, files=files)
    
    # 解析返回的JSON数据
    data = response.json()
    
    # 提取name字段的值
    name = data.get('name', None)
    
    return name
# Function to replace standalone 'a' or 'A' with '1'
def replace_standalone_a(text):
    return re.sub(r'\b[Aa]\b', '1', text)

def csv_to_dicts(filename):
    with open(filename, mode ='r')as file:
        csvFile = csv.reader(file)
        keys = next(csvFile) # get the keys
        dicts = {key: {} for key in keys}
        for row in csvFile:
            for i, key in enumerate(keys):
                dicts[key][row[0]] = row[i]
    return dicts['lora'], dicts['lora_weights'], dicts['lora_keyvalue']

def base64_to_img(base64_str):

    img_type = base64_str.split(';')[0].split('/')[1]
    head,context = base64_str.split(",")
    
    imgdata = base64.b64decode(context)
    img = Image.open(io.BytesIO(imgdata))
    #保存到input文件夹下
    random_filename = secrets.token_hex(12) + '.' + img_type
    img.save('input/' + random_filename)
    return random_filename

def generate_random_number(length):
    if length < 1:
        raise ValueError("Length cannot be less than 1")

    first_digit = random.randint(1, 9)  # ensure the first digit is not zero
    other_digits = [random.randint(0, 9) for _ in range(length - 1)]

    # combine all digits into a single number
    random_number = int(str(first_digit) + ''.join(map(str, other_digits)))
    return random_number


def get_images(images,workobj):

    encoded_images = []
    index = 0
    for node_id in images:
        print(node_id)
        
    
        for image_data in images[node_id]:
            # 将字节数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 将PIL Image对象转换为JPEG格式的字节串
            byte_arr = io.BytesIO()
            
            image.save(byte_arr, format='PNG')

            # 对字节串进行base64编码并转换为ascii字符串
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')
            
            
            imagedict = {"url":encoded_image,"seed":workobj.seed if workobj.seed is not None else "NAN","index":index}
            encoded_images.append(imagedict)
            # seed +=1
            index = index +1
            # print(index,"------------------------------------------------------------")

            # print(imagedict)
    
    
    # 然后，将所有的encoded_images打包成一个JSON对象
    result = {'images': encoded_images}

    return result

#截取unity端传入的截图，人物尽量铺满
def segimage(image_path):
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练的YOLOv8模型
    model = YOLO('yolo_model/yolov8s.pt').to(device)  # 将模型移动到GPU

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用模型进行对象检测，并设置检测阈值
    results = model(image_rgb, conf=0.1)  # 设置置信度阈值为0.1

    saved_image_name = secrets.token_hex(12) + '.png'
    saved_image_path = 'input/' + saved_image_name

    # 提取第一个检测框 (假设获得的检测结果按重要性排序)
    if results and len(results) > 0:
        result = results[0]
        detections = result.boxes.xyxy.cpu().numpy()  # 将检测结果转为numpy数组
        if detections.shape[0] > 0:
            # 提取第一个检测框的坐标
            x_min, y_min, x_max, y_max = map(int, detections[0][:4])

            # 扩展检测框以确保边界内
            x_min = max(x_min - 100, 0)
            y_min = max(y_min - 100, 0)
            x_max = min(x_max + 100, image_rgb.shape[1])
            y_max = min(y_max + 100, image_rgb.shape[0])

            # 调整检测框的长和宽为64的整数倍
            width = x_max - x_min
            height = y_max - y_min

            width_adjustment = 64 - (width % 64) if width % 64 != 0 else 0
            height_adjustment = 64 - (height % 64) if height % 64 != 0 else 0

            x_max = min(x_max + width_adjustment, image_rgb.shape[1])
            y_max = min(y_max + height_adjustment, image_rgb.shape[0])

            # 再次确保检测框的坐标在图像边界内
            x_min = max(x_max - (width + width_adjustment), 0)
            y_min = max(y_max - (height + height_adjustment), 0)

            print(x_min, y_min, x_max, y_max)

            # 裁剪出检测框的区域
            cropped_image = image_rgb[y_min:y_max, x_min:x_max]

            # 将裁剪出的区域转换为PIL图像，并保存为新的文件
            cropped_image_pil = Image.fromarray(cropped_image)
            cropped_image_pil.save(saved_image_path)  # 替换为保存路径

        else:
            print("No objects detected.")
    else:
        print("No objects detected.")

    print("Detection and cropping completed.")

    location = {"top": y_min, "left": x_min, "right": x_max, "bottom": y_max}
    return saved_image_name, location

class Line2img:
    def __init__(self) -> None:

        # self.request = request
        self.model_name = ""
        self.lora_name:str = ""
        self.strength_model = 0.75
        self.strength_clip = 0.75
        self.tile_name = ""
        self.lineart_name = ""
        self.lora_pretext = ""
        self.model_pretext = ""
        self.model_prenegative = ""
        self.width = 512
        self.height = 512
        self.seed = -1
        pass

    def init_params(self,request):
        
        model_index = request.json['CF_model']
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = style_models_prenegative[model_index]
        lora_index = request.json["lora_index"]
        lora, lora_weights, lora_keyvalue = csv_to_dicts(r"lora.csv")
        self.lora_name = lora[lora_index]
        self.lora_pretext = lora_keyvalue[lora_index]
        self.strength_model,self.strength_clip = float(lora_weights[lora_index]),float(lora_weights[lora_index])
        tile_base64 = request.json['tile']
        if tile_base64 != "":

            self.tile_name = base64_to_img(tile_base64)
        lineart_base64 = request.json['lineart']

        if lineart_base64 != "":

            self.lineart_name = base64_to_img(lineart_base64)
        # self.seed = request.json['seed']

        #使用PIL计算出init_image_name这个图片的宽和长
        
        if self.lineart_name != "":

            init_image = Image.open('input'+'/'+self.lineart_name)
            image_width, image_height = init_image.size

            aspect_ratio = image_width / image_height

            if image_width > image_height:
                # 在这种情况下，高度是短边
                self.height = 512
                # 根据长宽比，计算新的宽度
                self.width = int(512 * aspect_ratio)
            else:
                # 在这种情况下，宽度是短边
                self.width = 512
                # 根据长宽比，计算新的高度
                self.height = int(512 / aspect_ratio)
        
        self.seed = generate_random_number(12)

    def set_params_to_comfyui(self):
        prompt = json.load(open(r"afk_line_to_img_api.json",encoding="utf-8"))
        prompt["1"]["inputs"]["ckpt_name"] = self.model_name
        prompt["122"]["inputs"]["lora_name"] = self.lora_name
        prompt["195"]["inputs"]["prompt"] = self.lora_pretext
        prompt["122"]["inputs"]["strength_model"] = self.strength_model
        prompt["122"]["inputs"]["strength_clip"] = self.strength_clip
        prompt["129"]["inputs"]["image"] = self.tile_name if self.tile_name != "" else "static.png"
        prompt["120"]["inputs"]["image"] = self.lineart_name if self.lineart_name != "" else "static.png"
        prompt["16"]["inputs"]["seed"],prompt["175"]["inputs"]["seed"]= self.seed,self.seed
        prompt["139"]["inputs"]["width"],prompt["140"]["inputs"]["width"] = self.width,self.width
        prompt["139"]["inputs"]["height"],prompt["140"]["inputs"]["height"] = self.height,self.height
        prompt["196"]["inputs"]["prompt"] = self.model_pretext
        prompt["5"]["inputs"]["text"] = self.model_prenegative

        return prompt


        

        pass


class Line2img_nolora(Line2img):
    def __init__(self) -> None:
        super().__init__()
        self.positive = ""
        pass
    def init_params(self,request):
        model_index = request.json['CF_model']
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = style_models_prenegative[model_index]
        # lora_index = request.json["lora_index"]
        # lora, lora_weights, lora_keyvalue = csv_to_dicts(r"Q:\aigc中间code\lora.csv")
        # self.lora_name = lora[lora_index]
        # self.lora_pretext = lora_keyvalue[lora_index]
        # self.strength_model,self.strength_clip = float(lora_weights[lora_index]),float(lora_weights[lora_index])
        tile_base64 = request.json['tile']
        if tile_base64 != "":
            print("tile图进入")

            self.tile_name = base64_to_img(tile_base64)
            print(self.tile_name)
        lineart_base64 = request.json['lineart']

        if lineart_base64 != "":
            print("lineart图进入")

            self.lineart_name = base64_to_img(lineart_base64)
            print(self.lineart_name)
        # self.seed = request.json['seed']

        #使用PIL计算出init_image_name这个图片的宽和长
        
        if self.lineart_name != "":

            init_image = Image.open('input'+'/'+self.lineart_name)
            image_width, image_height = init_image.size

            aspect_ratio = image_width / image_height

            if image_width > image_height:
                # 在这种情况下，高度是短边
                self.height = 512
                # 根据长宽比，计算新的宽度
                self.width = int(512 * aspect_ratio)
            else:
                # 在这种情况下，宽度是短边
                self.width = 512
                # 根据长宽比，计算新的高度
                self.height = int(512 / aspect_ratio)
        
        self.seed = generate_random_number(12)
        self.positive = request.json['positive'] #这里可能是中文

        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"line_to_img_nolora_api.json",encoding="utf-8"))
        prompt["1"]["inputs"]["ckpt_name"] = self.model_name
        # prompt["122"]["inputs"]["lora_name"] = self.lora_name
        # prompt["193"]["inputs"]["Text"] = self.lora_pretext
        # prompt["122"]["inputs"]["strength_model"] = self.strength_model
        # prompt["122"]["inputs"]["strength_clip"] = self.strength_clip
        prompt["129"]["inputs"]["image"] = self.tile_name if self.tile_name != "" else "static.png"
        prompt["120"]["inputs"]["image"] = self.lineart_name if self.lineart_name != "" else "static.png"
        prompt["16"]["inputs"]["seed"],prompt["198"]["inputs"]["seed"]= self.seed,self.seed
        # prompt["139"]["inputs"]["width"],prompt["140"]["inputs"]["width"] = self.width,self.width
        # prompt["139"]["inputs"]["height"],prompt["140"]["inputs"]["height"]= self.height,self.height

        prompt["229"]["inputs"]["prompt"] = self.model_pretext
        prompt["5"]["inputs"]["text"] = self.model_prenegative


        # Set the text you want to translate
        text_to_translate = self.positive #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
        # Encode the text, returning a dictionary with tensors ready to feed the model
        encoded_text = tokenizer.encode(text_to_translate, return_tensors="pt")
        # Generate a translation. This returns a tensor with the predicted token ids
        translated_tokens = model.generate(encoded_text)
        # Decode the tokens to get the string of the translation
        # Decode the tokens to get the string of the translation
        translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Replace 'A' and 'a' in the translation with '1'
        translation = replace_standalone_a(translation)
        print(translation)

        self.positive = translation
        prompt["230"]["inputs"]["prompt"] = self.positive
        

        return prompt


        pass

class Upscaleimg(Line2img):
    def __init__(self) -> None:
        super().__init__()
        self.init_image = ""
        pass
    def init_params(self,request):
        init_image_base64 = request.json['init_image']
        if init_image_base64 != "":
            print("放大的原图进入")
            self.init_image = base64_to_img(init_image_base64)
        model_index = request.json['CF_model']
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = "embedding:EasyNegative.pt"
        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"upscale_api.json"))
        prompt["3"]["inputs"]["ckpt_name"] = self.model_name
        prompt["4"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        prompt["1"]["inputs"]["seed"]= self.seed
        prompt["24"]["inputs"]["prompt"] = self.model_pretext
        prompt["19"]["inputs"]["text"] = self.model_prenegative

        return prompt
        
        pass
    pass

class UpscaleimgBasic(Line2img):
    def __init__(self) -> None:
        super().__init__()
        self.init_image = ""
        pass
    def init_params(self,request):
        init_image_base64 = request.json['init_image']
        if init_image_base64 != "":
            print("放大的原图进入")
            self.init_image = base64_to_img(init_image_base64)
        
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"upscale_basic_api.json"))
        
        prompt["1"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        

        return prompt
        
        pass
    pass

class UpscaleimgPro(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self,request):
        init_image_base64 = request.json['init_image']
        if init_image_base64 != "":
            print("想要放大的图进入")
            self.init_image = base64_to_img(init_image_base64)
        model_index = request.json['CF_model']
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = "embedding:EasyNegative.pt"
        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"upscale_pro_api.json"))
        prompt["3"]["inputs"]["ckpt_name"] = self.model_name
        prompt["4"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        prompt["24"]["inputs"]["seed"]= self.seed
        prompt["28"]["inputs"]["prompt"] = self.model_pretext
        prompt["19"]["inputs"]["text"] = self.model_prenegative
        return prompt

class IgameFineTune(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self,request):
        init_image_base64 = request.json['init_image']
        if init_image_base64 != "":
            self.init_image = base64_to_img(init_image_base64)

            init_image = Image.open('input'+'/'+self.init_image)
            image_width, image_height = init_image.size

            aspect_ratio = image_width / image_height

            if image_width > image_height:
                # 在这种情况下，高度是短边
                self.height = 1024
                # 根据长宽比，计算新的宽度
                self.width = int(1024 * aspect_ratio) if int(1024 * aspect_ratio)<=2048 else 2048
            else:
                # 在这种情况下，宽度是短边
                self.width = 1024
                # 根据长宽比，计算新的高度
                self.height = int(1024 / aspect_ratio) if int(1024 / aspect_ratio)<=2048 else 2048



        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"igamexihua_api.json"))
        # prompt["3"]["inputs"]["ckpt_name"] = self.model_name
        prompt["129"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        prompt["204"]["inputs"]["width"] = self.width
        prompt["204"]["inputs"]["height"] = self.height
        prompt["16"]["inputs"]["seed"],prompt["206"]["inputs"]["seed"]= self.seed,self.seed

        return prompt


class ItemCreate(Line2img_nolora):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self,request):
        self.width = request.json['width']
        self.height = request.json['height']
        self.positive = request.json['positive'] #这里可能是中文
        model_index = request.json['CF_model']
        lineart_base64 = request.json['lineart']

        if lineart_base64 != "":
            print("lineart图进入")

            self.lineart_name = base64_to_img(lineart_base64)
            print(self.lineart_name)

        
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = style_models_prenegative[model_index]
        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"itemcreate_api.json",encoding="utf-8"))
        prompt["29"]["inputs"]["ckpt_name"] = self.model_name
        prompt["73"]["inputs"]["number"] = self.width
        prompt["75"]["inputs"]["number"] = self.height
        prompt["120"]["inputs"]["string"] = self.model_pretext
        prompt["31"]["inputs"]["text"] = self.model_prenegative
        prompt["32"]["inputs"]["seed"],prompt["38"]["inputs"]["seed"],prompt["65"]["inputs"]["noise_seed"] = self.seed,self.seed,self.seed

        prompt["107"]["inputs"]["image"] = self.lineart_name if self.lineart_name != "" else "static.png"
        print(prompt["107"]["inputs"]["image"])
        prompt["106"]["inputs"]["strength"] = 0 if self.lineart_name == "" else 0.75
        print(prompt["106"]["inputs"]["strength"])
        # if self.positive != "":
            
        #     # Set the text you want to translate
        #     text_to_translate = self.positive #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
        #     # Encode the text, returning a dictionary with tensors ready to feed the model
        #     encoded_text = tokenizer.encode(text_to_translate, return_tensors="pt")
        #     # Generate a translation. This returns a tensor with the predicted token ids
        #     translated_tokens = model.generate(encoded_text)
        #     # Decode the tokens to get the string of the translation
        #     # Decode the tokens to get the string of the translation
        #     translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        #     # Replace 'A' and 'a' in the translation with '1'
        #     translation = replace_standalone_a(translation)
        #     print(translation)

        #     self.positive = translation
        # gpttag = ""
        # import GptTagger
        # gpttag = GptTagger.tag_image("input/"+self.lineart_name)
        # print("gpt生成的语义：",gpttag)
        print(self.positive)
        prompt["123"]["inputs"]["prompt"] = self.positive


        return prompt


class Ideabomb(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        self.idea_scale = 0.5
        self.detail_weight = 0
        pass
    def init_params(self,request):
        init_image_base64 = request.json['init_image']
        # is_lock_in_details = request.json['lock_in_details']
        if init_image_base64 != "":
            self.init_image = base64_to_img(init_image_base64)
            init_image = Image.open('input'+'/'+self.init_image)
            image_width, image_height = init_image.size

            aspect_ratio = image_width / image_height

            if image_width > image_height:
                # 在这种情况下，高度是短边
                self.height = 512
                # 根据长宽比，计算新的宽度
                self.width = int(512 * aspect_ratio) if int(512 * aspect_ratio)<=2048 else 2048
            else:
                # 在这种情况下，宽度是短边
                self.width = 512
                # 根据长宽比，计算新的高度
                self.height = int(512 / aspect_ratio) if int(512 / aspect_ratio)<=2048 else 2048
        
        
        model_index = request.json['CF_model']
        self.idea_scale = request.json['idea_scale']
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = style_models_prenegative[model_index]
        # self.detail_weight = 1 if is_lock_in_details == True else 0
        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"ideabomb_api.json"))
        prompt["29"]["inputs"]["ckpt_name"] = self.model_name
        prompt["79"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        prompt["32"]["inputs"]["seed"],prompt["95"]["inputs"]["seed"]= self.seed,self.seed
        prompt["117"]["inputs"]["string"] = self.model_pretext
        prompt["31"]["inputs"]["text"] = self.model_prenegative
        prompt["93"]["inputs"]["width"] = self.width
        prompt["93"]["inputs"]["height"] = self.height

        prompt["95"]["inputs"]["denoise"] = self.idea_scale
        prompt["32"]["inputs"]["denoise"] = self.idea_scale
        prompt["50"]["inputs"]["strength"] =1-self.idea_scale
        prompt["53"]["inputs"]["strength"] = 1-self.idea_scale
        # prompt["113"]["inputs"]["Value"] = self.detail_weight
        
        
        tag = GptTagger.tag_image("input/"+self.init_image)
        print("gpt生成的语义：",tag)
        prompt["118"]["inputs"]["prompt"] = tag



        return prompt

class MakePose(Upscaleimg):

    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self,request):
        init_image_base64  =  request.json['init_image']  
        self.init_image = base64_to_img(init_image_base64) if init_image_base64 != "" else "static.png"

        init_image = Image.open('input'+'/'+self.init_image)
        image_width, image_height = init_image.size

        aspect_ratio = image_width / image_height

        if image_width > image_height:
            # 在这种情况下，高度是短边
            self.height = 512
            # 根据长宽比，计算新的宽度
            self.width = int(512 * aspect_ratio) 
        else:
            # 在这种情况下，宽度是短边
            self.width = 512
            # 根据长宽比，计算新的高度
            self.height = int(512 / aspect_ratio) 

        
        model_index = request.json['CF_model']
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = style_models_prenegative[model_index]

        #get lora index
        lora_index = request.json["lora_index"]
        if lora_index == "0":
            lora_index = "1"
        lora, lora_weights, lora_keyvalue = csv_to_dicts(r"lora.csv")
        self.lora_name = lora[lora_index]
        self.lora_pretext = lora_keyvalue[lora_index]
        self.strength_model,self.strength_clip = float(lora_weights[lora_index]),float(lora_weights[lora_index])


        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"makepose_api.json"))

        print("makepose makepose makepose makepose")
        
        prompt["60"]["inputs"]["ckpt_name"] = self.model_name
        print(self.model_name)
        prompt["73"]["inputs"]["number"] = self.width
        prompt["75"]["inputs"]["number"] = self.height

        prompt["156"]["inputs"]["prompt"] = self.model_pretext
        prompt["63"]["inputs"]["text"] = self.model_prenegative
        prompt["106"]["inputs"]["seed"],prompt["65"]["inputs"]["noise_seed"]= self.seed,self.seed
        prompt["87"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        prompt["95"]["inputs"]["lora_name"] = self.lora_name
        prompt["95"]["inputs"]["strength_model"] = self.strength_model
        prompt["95"]["inputs"]["strength_clip"] = self.strength_clip

        prompt["157"]["inputs"]["prompt"] = self.lora_pretext
        print(self.lora_pretext)
        return prompt

class Rem(Upscaleimg):

    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self,request):
        init_image_base64  =  request.json['init_image']  
        self.init_image = base64_to_img(init_image_base64) if init_image_base64 != "" else "static.png"
        self.positive = request.json['positive']
        # self.positive = "人"
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"rem_api.json"))
        prompt["2"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"

        prompt["8"]["inputs"]["prompt"] = self.positive

        

        return prompt


class Fenjing(Upscaleimg):

    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self,request,filename):
        model_index = request.form.get('CF_model')
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = style_models_prenegative[model_index]
        
        self.init_image = filename

        init_image = Image.open('input'+'/'+self.init_image)
        image_width, image_height = init_image.size

        aspect_ratio = image_width / image_height

        if image_width > image_height:
            # 在这种情况下，高度是短边
            self.height = 512
            # 根据长宽比，计算新的宽度
            self.width = int(512 * aspect_ratio) if int(512 * aspect_ratio)<=2048 else 2048
        else:
            # 在这种情况下，宽度是短边
            self.width = 512
            # 根据长宽比，计算新的高度
            self.height = int(512 / aspect_ratio) if int(512 / aspect_ratio)<=2048 else 2048
        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"fenjing_api.json"))
        prompt["229"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        # prompt["120"]["inputs"]["image"] = self.lineart_name if self.lineart_name != "" else "static.png"
        prompt["16"]["inputs"]["seed"]= self.seed
        prompt["139"]["inputs"]["width"],prompt["140"]["inputs"]["width"] = self.width,self.width
        prompt["139"]["inputs"]["height"],prompt["140"]["inputs"]["height"]= self.height,self.height
        prompt["1"]["inputs"]["ckpt_name"] = self.model_name
        prompt["194"]["inputs"]["Text"] = self.model_pretext
        prompt["5"]["inputs"]["text"] = self.model_prenegative

        return prompt

class Video2Video():
    def __init__(self) -> None:
        self.video_name = ""
        pass
    def init_params(self,request,filename):

        self.video_name = filename #从comfyui返回值中确定真实的视频文件名
        pass
        
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"video2video_api.json"))
        prompt["68"]["inputs"]["video"] = self.video_name
        prompt["68"]["inputs"]["videopreview"]["filename"] = self.video_name
        
        return prompt

class Video2Video_Lora(Line2img):
    def __init__(self) -> None:
        self.video_name = ""
        pass
    def init_params(self,request:Request,filename:str):

        self.video_name = filename #从comfyui返回值中确定真实的视频文件名
        

        model_index = request.form.get('CF_model')
        self.model_name = style_models[model_index]
        self.model_pretext = style_models_prepositive[model_index]
        self.model_prenegative = style_models_prenegative[model_index]
        lora_index = request.form.get("lora_index")
        lora, lora_weights, lora_keyvalue = csv_to_dicts(r"lora.csv")
        self.lora_name = lora[lora_index]
        self.lora_pretext = lora_keyvalue[lora_index]
        self.strength_model,self.strength_clip = float(lora_weights[lora_index]),float(lora_weights[lora_index])

        self.seed = generate_random_number(12)

        pass
        
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"video2video_lora_api.json")) #这个部分要换成带lora的视频流程文件
        prompt["68"]["inputs"]["video"] = self.video_name
        prompt["68"]["inputs"]["videopreview"]["filename"] = self.video_name

        prompt["13"]["inputs"]["ckpt_name"] = self.model_name
        
        prompt["25"]["inputs"]["text"] = self.model_prenegative
        prompt["88"]["inputs"]["Text"] = self.model_pretext + ","+self.lora_pretext

        prompt["131"]["inputs"]["lora_name"] = self.lora_name
        
        prompt["131"]["inputs"]["strength_model"] = self.strength_model
        prompt["131"]["inputs"]["strength_clip"] = self.strength_clip

        prompt["19"]["inputs"]["seed"] = self.seed
        
        

        
        return prompt

class MagicAnimate(Video2Video):
    def __init__(self) -> None:
        super().__init__()
        self.image_name:str = ""
        pass
    def init_params(self,request:Request,filename:list[str]):

        self.video_name = filename[0]#从comfyui返回值中确定真实的视频文件名
        self.image_name = filename[1]#从comfyui返回值中确定真实的图片文件名
        self.seed = generate_random_number(3)

        pass
        
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"magicanimate_api.json"))
        prompt["23"]["inputs"]["video"] = self.video_name
        prompt["14"]["inputs"]["image"] = self.image_name
        prompt["17"]["inputs"]["seed"] = self.seed
        return prompt


class IgameFineTuneLora(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self,request):
        init_image_base64 = request.json['init_image']
        if init_image_base64 != "":
            self.init_image = base64_to_img(init_image_base64)

            init_image = Image.open('input'+'/'+self.init_image)
            image_width, image_height = init_image.size

            aspect_ratio = image_width / image_height

            if image_width > image_height:
                # 在这种情况下，高度是短边
                self.height = 1024
                # 根据长宽比，计算新的宽度
                self.width = int(1024 * aspect_ratio) if int(1024 * aspect_ratio)<=2048 else 2048
            else:
                # 在这种情况下，宽度是短边
                self.width = 1024
                # 根据长宽比，计算新的高度
                self.height = int(1024 / aspect_ratio) if int(1024 / aspect_ratio)<=2048 else 2048

        lora_index = request.json["lora_index"]
        lora, lora_weights, lora_keyvalue = csv_to_dicts(r"lora.csv")
        self.lora_name = lora[lora_index]
        self.lora_pretext = lora_keyvalue[lora_index]
        self.strength_model,self.strength_clip = float(lora_weights[lora_index]),float(lora_weights[lora_index])
        

        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"igamexihua_lora_api.json"))
        # prompt["3"]["inputs"]["ckpt_name"] = self.model_name
        prompt["129"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        prompt["204"]["inputs"]["width"] = self.width
        prompt["204"]["inputs"]["height"] = self.height
        prompt["16"]["inputs"]["seed"],prompt["206"]["inputs"]["seed"],prompt["227"]["inputs"]["seed"],prompt["251"]["inputs"]["seed"]= self.seed,self.seed,self.seed,self.seed
        prompt["266"]["inputs"]["lora_name"] = self.lora_name
        prompt["267"]["inputs"]["prompt"] = "AFK,"+self.lora_pretext
        prompt["266"]["inputs"]["strength_model"] = self.strength_model
        prompt["266"]["inputs"]["strength_clip"] = self.strength_clip


        return prompt



class UnityTool(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        self.location = {}
        pass
    def init_params(self,request):
        init_image_base64 = request.json['init_image']
        if init_image_base64 != "":
            self.init_image = base64_to_img(init_image_base64)

            self.init_image,self.location = segimage('input'+'\\'+self.init_image)
            

        self.lora_index = request.json["lora_index"]
        print("传进来lora的索引为:"+self.lora_index)
        lora, lora_weights, lora_keyvalue = csv_to_dicts(r"lora.csv")
        self.lora_name = lora[self.lora_index]
        self.lora_pretext = lora_keyvalue[self.lora_index]
        #self.strength_model,self.strength_clip = float(lora_weights[lora_index]),float(lora_weights[lora_index])
        self.positive = request.json["positive"]
        print("unity端传入的提示词: "+self.positive)

        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):

        prompt = json.load(open(r"unity_api.json"))

        
        if self.lora_index != "0":
            
            prompt["265"]["inputs"]["lora_name"] = self.lora_name 
        else:
            prompt["265"]["inputs"]["strength_model"] = 0

        
        prompt["12"]["inputs"]["image"] = self.init_image if self.init_image != "" else "static.png"
        #prompt["204"]["inputs"]["width"] = self.width
        #prompt["204"]["inputs"]["height"] = self.height
        prompt["262"]["inputs"]["seed"] = self.seed

        
        prompt["245"]["inputs"]["text"] = "trq,TRQ style,"+self.positive
        #prompt["194"]["inputs"]["Text"] = "AFK,"+self.lora_pretext
        #prompt["266"]["inputs"]["strength_model"] = self.strength_model
        #prompt["266"]["inputs"]["strength_clip"] = self.strength_clip

        prompt['267']['inputs']['top'] = self.location['top']
        prompt['267']['inputs']['left'] = self.location['left']
        prompt['267']['inputs']['right'] = self.location['right']
        prompt['267']['inputs']['bottom'] = self.location['bottom']

        return prompt

class KV(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self, request):
        
        self.positive = request.json["positive"]
        self.monika = request.json['monika'] #判断是横板还是竖版 "0" 或"1"  "0"是横板 "1"是竖版
        print("kv端传入的提示词: "+self.positive)

        self.seed = generate_random_number(12)
        pass
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"post_api.json"))
        
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["3"]["inputs"]["seed"] = self.seed


        if self.monika == "0":
            pass
        else:
            prompt["5"]["inputs"]["width"] = 768
            prompt["5"]["inputs"]["height"] = 1368
        return prompt


class AFKh5(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self, request):
        
        self.positive = request.json["positive"]
        #print("kv端传入的提示词: "+self.positive)

        self.seed = generate_random_number(15)
        pass
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"AFKh5_api.json"))
        
        prompt["6"]["inputs"]["text"] = self.positive
        prompt["2"]["inputs"]["seed"] = self.seed


        return prompt
            
class Partyh5(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self, request):
        
        #self.positive = request.json["positive"]
        #print("kv端传入的提示词: "+self.positive)

        self.seed = generate_random_number(15)
        pass
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"partyguaji_api.json"))
        
        prompt["36"]["inputs"]["seed"] = self.seed
        prompt["132"]["inputs"]["seed"] = self.seed


        return prompt

#igame黑白草图生图c
class IgameBWD(Upscaleimg):
    def __init__(self) -> None:
        super().__init__()
        pass
    def init_params(self, request):

        init_image_base64 = request.json['init_image']
        if init_image_base64 != "":
            self.init_image = base64_to_img(init_image_base64)

        
        self.positive = request.json["positive"]
        if self.positive != "":
            try:
                self.positive = GptTagger.Tagger().translate_gpt(self.positive)
            except Exception as e:
                self.positive = "trq style,TRQ style,"
        else:
            self.positive = "trq style,TRQ style,"

        #print("kv端传入的提示词: "+self.positive)

        self.seed = generate_random_number(15)
        pass
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"igame_black_white_draft_c_api.json"))

        prompt["173"]["inputs"]["image"] = self.init_image
        prompt["176"]["inputs"]["prompts"] = self.positive
        
        prompt["36"]["inputs"]["seed"],prompt["162"]["inputs"]["seed"],prompt["181"]["inputs"]["seed"] = self.seed,self.seed,self.seed


        return prompt

#igame黑白草图生图s
class IgameBWDS(IgameBWD):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"igame_black_white_draft_s_api.json"))

        

        prompt["86"]["inputs"]["image"] = self.init_image
        prompt["32"]["inputs"]["prompts"] = self.positive
        
        prompt["3"]["inputs"]["seed"],prompt["52"]["inputs"]["seed"],prompt["79"]["inputs"]["seed"],prompt["87"]["inputs"]["seed"] = self.seed,self.seed,self.seed,self.seed



        return prompt
    
#角色低模转kv
class Clow2KV(IgameBWD):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"clow2kv_api.json"))

        prompt["98"]["inputs"]["image"] = self.init_image
        print(self.init_image)
        # prompt["141"]["inputs"]["prompts"] = self.positive
        
        prompt["36"]["inputs"]["seed"],prompt["151"]["inputs"]["seed"] = self.seed,self.seed



        return prompt


class IgameColorDraftC(IgameBWD):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"igame_colordraft_c_api.json"))

        prompt["98"]["inputs"]["image"] = self.init_image
        # prompt["141"]["inputs"]["prompts"] = self.positive
        
        prompt["36"]["inputs"]["seed"],prompt["162"]["inputs"]["seed"] = self.seed,self.seed



        return prompt

class IgameColorDraftS(IgameBWD):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"igame_colordraft_s_api.json"))

        prompt["98"]["inputs"]["image"] = self.init_image
        # prompt["141"]["inputs"]["prompts"] = self.positive
        
        prompt["36"]["inputs"]["seed"],prompt["162"]["inputs"]["seed"] = self.seed,self.seed



        return prompt

class IgameBWCDraftC(IgameBWD):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"igame_black_white_color_draft_c_api.json"))

        prompt["98"]["inputs"]["image"] = self.init_image
        # prompt["141"]["inputs"]["prompts"] = self.positive
        
        prompt["36"]["inputs"]["seed"],prompt["162"]["inputs"]["seed"] = self.seed,self.seed



        return prompt

class AFKColorDraft(IgameBWD):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def set_params_to_comfyui(self):
        prompt = json.load(open(r"afk_color_draft_api.json"))

        prompt["98"]["inputs"]["image"] = self.init_image
        print(self.init_image)
        # prompt["141"]["inputs"]["prompts"] = self.positive
        
        prompt["36"]["inputs"]["seed"],prompt["149"]["inputs"]["seed"] = self.seed,self.seed



        return prompt
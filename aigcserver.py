# -*- coding: utf-8 -*-


#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint
import random
import aiohttp
import asyncio
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import requests
import uuid
import json
import urllib.request
import urllib.parse
import random
from transformers import pipeline
import re
import io
import os
import base64
from flask import render_template,url_for
from flask import Flask, request, jsonify,abort,redirect
import secrets
from flask_cors import CORS
from PIL import Image
import math
import csv
import Workdiy
import time
from functools import wraps
#导入zipfile模块，用于解压zip文件
import zipfile
from pathlib import Path
import shutil
import string
#导入send_file模块，用于发送文件
from flask import send_file
from flask import send_from_directory
import GptTagger
import base64
import os
from PIL import Image
from io import BytesIO

import mongodb

def paste_image_on_transparent_background(image_path, x, y):
    # 创建一个4000x3000的透明背景图像
    background = Image.new('RGBA', (3840, 2160), (0, 0, 0, 0))
    
    # 打开输入的PNG图像
    foreground = Image.open(image_path).convert('RGBA')
    
    # 将前景图像粘贴到背景图像上，左上角位置是(x, y)
    background.paste(foreground, (x, y), foreground)
    
    # 返回结果图像
    return background

def merge_images(image_files):
    # 获取第一张图片的尺寸作为基准尺寸
    base_image = Image.open(image_files[0])
    base_width, base_height = base_image.size

    # 调整其他图片的尺寸
    resized_images = []
    for img_file in image_files:
        img = Image.open(img_file)
        img = img.resize((base_width, base_height))
        resized_images.append(img)

    # 合并图片
    merged_image = Image.alpha_composite(resized_images[0].convert("RGBA"), resized_images[1].convert("RGBA"))
    for img in resized_images[2:]:
        merged_image = Image.alpha_composite(merged_image, img.convert("RGBA"))

    return merged_image

def unzipfile(zip_file):
    with zipfile.ZipFile(zip_file) as zf:
        # 获取zip文件中的所有条目
        members = zf.namelist()

        # 查找所有成员的公共基础路径，这应该是顶层文件夹
        common_path = os.path.commonpath(members)

        # 为每个成员构建一个新的路径，删除公共基础路径
        for member in members:
            new_path = os.path.relpath(member, common_path)
            new_abs_path = os.path.join('input', new_path)

            # 检查这个条目是否是一个文件
            if not member.endswith('/'):
                # 使用Pathlib创建必要的目录
                Path(os.path.dirname(new_abs_path)).mkdir(parents=True, exist_ok=True)
                # 解压这个文件到新的路径
                with zf.open(member) as source, open(new_abs_path, 'wb') as target:
                    shutil.copyfileobj(source, target)

def get_filenames_in_zip(zip_file):
    # 创建一个ZipFile对象
    with zipfile.ZipFile(zip_file, 'r') as zf:
        # 获取zip文件中的所有条目
        members = zf.namelist()

        # 查找所有成员的公共基础路径，这应该是顶层文件夹
        common_path = os.path.commonpath(members)

        # 为每个成员构建一个新的路径，删除公共基础路径
        new_members = []
        for member in members:
            # 删除顶层文件夹名称
            if member.startswith(common_path):
                new_member = member[len(common_path):].lstrip('/')
            else:
                new_member = member
            # 确保新成员不为空
            if new_member:
                new_members.append(new_member)
  
    return new_members

# from flask_restx import Api, Resource
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录函数结束的时间
        print(f"Function {func.__name__} took {(end_time - start_time):.5f} seconds to run.")
        return result
    return wrapper


def csv_to_dicts(filename):
    with open(filename, mode ='r')as file:
        csvFile = csv.reader(file)
        keys = next(csvFile) # get the keys
        dicts = {key: {} for key in keys}
        for row in csvFile:
            for i, key in enumerate(keys):
                dicts[key][row[0]] = row[i]
    return dicts['lora'], dicts['lora_weights'], dicts['lora_keyvalue']
app = Flask(__name__)
# api = Api(app)

CORS(app)


global_queues = 0


pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import numpy as np
from transformers import pipeline

# 创建翻译管道
translator = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")


# server_address = "127.0.0.1:8188"

client_id_index = 0
client_ids = []

for i in range(200):
    client_id = str(uuid.uuid4())
    client_ids.append(client_id)


def init_controlnet():
    images_controlnet = {
        "openpose":"",
        "color":"",
        "tile":"",
        "scribble":"",
        "canny":"",
        "lineart":"",
        "depth":"",
        "softedge":"",
    }

    images_key = ["image1","image2","image3","image4","image5","image6","image7","image8"]
    weight_key = ['weight1','weight2','weight3','weight4','weight5','weight6','weight7','weight8']

    imagekeytocontrolnet = {
    "image1":"openpose",
    "image2":"color",
    "image3":"tile",
    "image4":"scribble",
    "image5":"canny",
    "image6":"lineart",
    "image7":"depth",
    "image8":"softedge",
    }

    controllnet_weights = {
    "openpose":0,
    "color":0,
    "tile":0,
    "scribble":0,
    "canny":0,
    "lineart":0,
    "depth":0,
    "softedge":0,
    }

    weightkeytocontrolnet = {
    "weight1":"openpose",
    "weight2":"color",
    "weight3":"tile",
    "weight4":"scribble",
    "weight5":"canny",
    "weight6":"lineart",
    "weight7":"depth",
    "weight8":"softedge",
    }
    for item in images_key:
        try:
            base64_str = request.json[item]
            if base64_str == "":
            
                images_controlnet[imagekeytocontrolnet[item]] = ""
            else:
                print(item +" controlnet图进来了")
                images_controlnet[imagekeytocontrolnet[item]] = base64_to_img(base64_str)
                

        except KeyError:
            print("抛出了异常1")
            images_controlnet[images_controlnet[imagekeytocontrolnet[item]]] = ""
            
    for item in weight_key:
        try:
            print(request.json[item])
            print(images_controlnet[weightkeytocontrolnet[item]])
            controllnet_weights[weightkeytocontrolnet[item]] = request.json[item] if images_controlnet[weightkeytocontrolnet[item]] != "" else 0
            print( "权重为",controllnet_weights[weightkeytocontrolnet[item]])
        except KeyError:
            print("抛出了异常2")
            images_controlnet[images_controlnet[imagekeytocontrolnet[item]]] = 0


    return images_controlnet,controllnet_weights

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

# lora = {
#         "0":None,
#         "1":"Brutus.safetensors",
#         "2":"Dolly.safetensors",
#         "3":"Ella.safetensors",
#         "4":"Fay.safetensors",
#         "5":"Seth.safetensors",
#         "6":"Valen.safetensors"
#     }

# lora_weights = {
#         "0":0,
#         "1":0.85,
#         "2":0.75,
#         "3":0.7,
#         "4":0.85,
#         "5":0.75,
#         "6":0.85

#     }

# lora_keyvalue = {
#         "1":"Brutus, 1boy,lion ears,lion boy, lion,furry,muscular, shoulder armor,",
#         "2":"Dolly,1girl,  hair ribbon, ponytail, braid,",
#         "3":"Eall,1girl,hat,blue cape,",
#         "4":"Fay, 1girl,",
#         "5":"Seth,yellow eyes, claws, colored skin, muscular, animal ears, capelet,",
#         "6":"Valen, 1boy, black hair,purple eyes,"
#     }



lora, lora_weights, lora_keyvalue = csv_to_dicts(r"lora.csv")

def workmain(Workobj,request):
    #---------------------------------------------------------------------
    client_id = str(uuid.uuid4())
    server_address = "127.0.0.1:8188"
    workobj = Workobj()
    # print(request.json)
    workobj.init_params(request)

    prompt = workobj.set_params_to_comfyui()
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))


    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    print("comfyui执行完")
    print("图片的个数为："+str(len(filenamelist)))
    print(filenamelist)
    # print(images)
    # print(type(images))
    result = Workdiy.get_images(images,workobj)

    return result
#将一个base64的字符串保存为png图片
def base64_to_img(base64_str):

    img_type = base64_str.split(';')[0].split('/')[1]
    head,context = base64_str.split(",")
    
    imgdata = base64.b64decode(context)
    img = Image.open(io.BytesIO(imgdata))
    #保存到input文件夹下
    random_filename = secrets.token_hex(12) + '.' + img_type
    img.save('input/' + random_filename)
    return random_filename

def filter_tuples_remove_brackets_and_numbers(str_lst, threshold):
    # Use regular expressions to find tuples in the string
    tuples = re.findall(r'\(([^\)]+),\s([^\)]+)\)', str_lst)
    
    # Filter tuples based on the threshold
    filtered_tuples = [(name, float(score)) for name, score in tuples if float(score) >= threshold]
    
    # Convert the tuples back to a string, and remove the brackets and numbers
    filtered_str = ', '.join(name for name, _ in filtered_tuples)
    
    return filtered_str


@app.route('/getqueue',methods=['GET'])
def get_queues():

    global global_queues

    return jsonify({'queues':global_queues})



# Use a pipeline as a high-level helper

@app.route('/')
def index():

    return render_template('vueindex.html')



@app.route('/tag',methods = ['POST'])
@timing_decorator
def tag():

    
    print("得到反推请求")
    # global client_id_index

    # client_id = client_ids[client_id_index]

    # client_id_index += 1

    # if client_id_index >= len(client_ids):
    #     client_id_index = 0
    
    # print(client_id)

    # input_folder = "input"
    # prompt = json.load(open("tagger_api.json"))
    # thresold = request.json['thresold']
    # prompt['2']['inputs']['threshold'] = thresold
    init_image_base64 = request.json['init_image']

    # print("反推请求的图",init_image_base64)
    init_image_name = base64_to_img(init_image_base64)
    # prompt['1']['inputs']['image'] = init_image_name

    # # print(init_image_name)

    # ws = websocket.WebSocket()
    # ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    # server_address="127.0.0.1:8188"
    # get_images(ws, prompt,client_id,server_address)

    # #用json模块导入prompt.json文件内容
    # tagdict = json.load(open("prompt.json"))
    # #读取json文件中的内容
    # tag = tagdict['0']

    # # filtered_tag = filter_tuples_remove_brackets_and_numbers(tag, thresold)

    

    # #翻译
    # text = tag+'.'
    # print(text)
    # translation = translator(text, max_length=5000)

    # ch = translation[0]['translation_text']

    # print(ch)

    tag = GptTagger.tag_image("input/"+init_image_name)

    return jsonify({'en':tag,'ch':"由Gpt支持"})


@app.route('/upscaleimg',methods = ['POST'])
@timing_decorator
def upscaleimg():

    #---------------------------------------------------------------------
    result = workmain(Workdiy.Upscaleimg,request)
    # print(encoded_images)
    return jsonify(result)

    pass


@app.route('/upscaleimgpro',methods = ['POST'])
@timing_decorator
def upscaleimgpro():

    #---------------------------------------------------------------------
    result = workmain(Workdiy.UpscaleimgPro,request)
    # print(result)
    # print(encoded_images)
    return jsonify(result)

@app.route('/texttoimg',methods=['POST'])
@timing_decorator
def texttoimg():

    global client_id_index

    client_id = client_ids[client_id_index]

    client_id_index += 1

    if client_id_index >= len(client_ids):
        client_id_index = 0
    
    print(client_id)

    
    # print(request.json)
    # print("dafsfcjajdkahkdahkfhaknd")
    # Define the folder where you want to save the images
    input_folder = "input"
    prompt = json.load(open("texttoimg_api.json"))

    

    if (model_index := request.json['CF_model']):
        print("模型索引",model_index)

        model_name = style_models[model_index]
        print("模型名称",model_name)
        model_prepositive = style_models_prepositive[model_index]
        model_prenegative = style_models_prenegative[model_index]
        print("模型负面",model_prenegative)
        pre_keyvalue = model_prepositive

        prompt['1']['inputs']['ckpt_name'] = model_name
    else:
        print("未传入模型！！！！！！！")
        return "no model post in",404

    images_controlnet,controllnet_weights = init_controlnet()

    
    #一个seed
    seed = request.json['seed']
    seed = int(seed)
    #一个prompt
    param1 = request.json['positive'] #正面提示词
    print(param1)

    param2 = request.json['negative'] #负面提示词
    print(param2)


    #一个lora索引
    lora_index = request.json['lora_index']
    #get image_width
    image_width = request.json['image_width']
    #get image_height
    image_height = request.json['image_height']
    #get batch_size
    batch_size = request.json['batch_size']
    batch_size = int(batch_size)
    if batch_size>5:
        batch_size = 5

    global global_queues
    global_queues = global_queues +batch_size


    print(global_queues,"----------------------------------")

    

    # Loop through the files
    # for index,file in enumerate(files):
    #     if file:
    #         # Generate a random filename
    #         random_filename = secrets.token_hex(8) + '.jpg'
    #         file.save(os.path.join(input_folder, random_filename))
    #         controllnet_images[index] = random_filename
    
    # print(controllnet_images)

    

    if lora_index:
        if lora_index == "0":
            prompt["44"]["inputs"]["strength_model"] = 0
            prompt["44"]["inputs"]["strength_clip"] = 0
        else:
            print(lora_index)
            prompt["44"]["inputs"]["lora_name"] = lora[lora_index]
            #设置lora权重
            prompt["44"]["inputs"]["strength_model"] = lora_weights[lora_index]
            prompt["44"]["inputs"]["strength_clip"] = lora_weights[lora_index]
            pre_keyvalue = pre_keyvalue + ","+lora_keyvalue[lora_index]
    else:
        prompt["44"]["inputs"]["strength_model"] = 0
        prompt["44"]["inputs"]["strength_clip"] = 0 #strength_clip


    if image_width:
        prompt["71"]["inputs"]["width"] = image_width
        # prompt["33"]["inputs"]["width"] = image_width*2
        # print(image_width*2,"adefkosojfvksnfvjn")
    if image_height:
        prompt["71"]["inputs"]["height"] = image_height
        # prompt["33"]["inputs"]["height"] = image_height*2
    
    if batch_size:
        prompt["71"]["inputs"]["batch_size"] = batch_size
    

    prompt["20"]["inputs"]["image"] = images_controlnet["softedge"] if images_controlnet["softedge"] != "" else "static.png"
    prompt["21"]["inputs"]["image"] = images_controlnet["color"] if images_controlnet["color"] != "" else "static.png"
    prompt["22"]["inputs"]["image"] = images_controlnet["openpose"] if images_controlnet["openpose"] != "" else "static.png"
    prompt["23"]["inputs"]["image"] = images_controlnet["tile"] if images_controlnet["tile"] != "" else "static.png"
    prompt["72"]["inputs"]["image"] = images_controlnet["scribble"] if images_controlnet["scribble"] != "" else "static.png"

    
    
    prompt["5"]["inputs"]["strength"] = controllnet_weights['softedge']
    prompt["8"]["inputs"]["strength"] = controllnet_weights['color']
    prompt["10"]["inputs"]["strength"] = controllnet_weights['openpose']
    prompt["12"]["inputs"]["strength"] = controllnet_weights['tile']
    prompt["74"]["inputs"]["strength"] = controllnet_weights['scribble']
    
    

    if batch_size >=2:

        randomseed = generate_random_number(12)
        prompt["14"]["inputs"]["seed"] = randomseed
        # prompt["28"]["inputs"]["seed"] = randomseed
        seed = randomseed
    else:
        if seed != -1 and seed != None:
            prompt["14"]["inputs"]["seed"] = seed
            # prompt["28"]["inputs"]["seed"] = seed
        else:
            randomseed = generate_random_number(12)
            prompt["14"]["inputs"]["seed"] = randomseed
            # prompt["28"]["inputs"]["seed"] = randomseed
            seed = randomseed

    # print("----------------------------------",param1)

    if (param1 != '' and param1 != None) or lora_index != "0":

        
    
        # Set the text you want to translate
        text_to_translate = param1 #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
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

        
        # print(pre_keyvalue)
        prompt["69"]["inputs"]["text"] =  pre_keyvalue +  translation
        print("正面提示词：",prompt["69"]["inputs"]["text"])
    
    else:
        # return "no prompt input, please retry",404
        prompt["69"]["inputs"]["text"] = pre_keyvalue +", solo, 1girl,"
        print(prompt["69"]["inputs"]["text"])
    

    if param2 != '' and param2 != None:

        
    
        # Set the text you want to translate
        text_to_translate = param2 #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
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

        
        # print(pre_keyvalue)
        prompt["4"]["inputs"]["text"] =  model_prenegative +  translation
        print("负面提示词：",prompt["4"]["inputs"]["text"])
    
    else:
        prompt["4"]["inputs"]["text"] =  model_prenegative
        pass
    



 

    server_address = "127.0.0.1:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    # print(prompt)

    
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    # print(images)
    # print(type(images))
    encoded_images = []
    for node_id in images:
        print(node_id)
        index = 0
    
        for image_data in images[node_id]:
            # 将字节数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 将PIL Image对象转换为JPEG格式的字节串
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')

            # 对字节串进行base64编码并转换为ascii字符串
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')  
            
            

            imagedict = {"url":encoded_image,"seed":seed,"index":index}
            encoded_images.append(imagedict)
            # seed +=1
            index +=1

            # print(imagedict)
    
    
    # 然后，将所有的encoded_images打包成一个JSON对象
    result = {'images': encoded_images}
    print("文生图返回张数：",len(encoded_images))
    global_queues = global_queues - batch_size
    # print(encoded_images)
    return jsonify(result)


# {
#     images:[
#         {"url":'image字符串',"seed":12,"index":0},
#         {"url":'image字符串',"seed":13,"index":1},
#     ]
# }

@app.route('/imgtoimg',methods = ['POST'])
@timing_decorator
def imgtoimg():
    global client_id_index

    client_id = client_ids[client_id_index]

    client_id_index += 1

    if client_id_index >= len(client_ids):
        client_id_index = 0
    

    
    input_folder = "input"
    prompt = json.load(open("imgtoimg_api.json"))

    

    
    
    if (model_index := request.json['CF_model']):
        
        print("模型索引",model_index)
        model_name = style_models[model_index]
        print(model_name)
        model_prepositive = style_models_prepositive[model_index]
        model_prenegative = style_models_prenegative[model_index]
        pre_keyvalue = model_prepositive

        prompt['1']['inputs']['ckpt_name'] = model_name
    else:
        print("未传入模型！！！！！！！")
        return "no model post in",404


    init_image_base64 = request.json['init_image']

    
    init_image_name = base64_to_img(init_image_base64)
    prompt['45']['inputs']['image'] = init_image_name

    #使用PIL计算出init_image_name这个图片的宽和长　
    init_image = Image.open(input_folder+'/'+init_image_name)
    image_width, image_height = init_image.size
    print("原图宽度",image_width)
    print("原图高度",image_height)

    
    aspect_ratio = image_width / image_height

    if image_width > image_height:
        # 在这种情况下，高度是短边
        prompt['98']['inputs']['height'] = 512
        # 根据长宽比，计算新的宽度
        prompt['98']['inputs']['width'] = int(512 * aspect_ratio)
    else:
        # 在这种情况下，宽度是短边
        prompt['98']['inputs']['width'] = 512
        # 根据长宽比，计算新的高度
        prompt['98']['inputs']['height'] = int(512 / aspect_ratio)




    ratio = request.json['ratio']




    # #以下是目标尺寸
    # if image_width:
    #     # prompt["71"]["inputs"]["width"] = image_width
    #     prompt["33"]["inputs"]["width"] = image_width*ratio
    #     print(image_width*2,"adefkosojfvksnfvjn")
    # if image_height:
    #     # prompt["71"]["inputs"]["height"] = image_height
    #     prompt["33"]["inputs"]["height"] = image_height*ratio

    

    print("原图名字",init_image_name)

    Denoising_strength = request.json['Denoising_strength']
    prompt['14']['inputs']['denoise'] = (Denoising_strength)
    # prompt['28']['inputs']['denoise'] = min(0.45,Denoising_strength)

    print("发散强度",Denoising_strength)
    
    # ratio = 1


    images_controlnet,controllnet_weights = init_controlnet()

    #一个seed
    seed = request.json['seed']
    seed = int(seed)
    #一个prompt
    param1 = request.json['positive']
    print(param1)

    param2 = request.json['negative']
    print(param2)


    #一个lora索引
    lora_index = request.json['lora_index']
    #get image_width
    # image_width = request.json['image_width']
    #get image_height
    # image_height = request.json['image_height']
    #get batch_size
    batch_size = request.json['batch_size']
    batch_size = int(batch_size)
    if batch_size>5:
        batch_size = 5

    global global_queues
    global_queues = global_queues +batch_size


    print(global_queues,"----------------------------------")

    

    # Loop through the files
    # for index,file in enumerate(files):
    #     if file:
    #         # Generate a random filename
    #         random_filename = secrets.token_hex(8) + '.jpg'
    #         file.save(os.path.join(input_folder, random_filename))
    #         controllnet_images[index] = random_filename
    
    # print(controllnet_images)

    if lora_index:
        if lora_index == "0":
            prompt["44"]["inputs"]["strength_model"] = 0
            prompt["44"]["inputs"]["strength_clip"] = 0
        else:
            print(lora_index)
            prompt["44"]["inputs"]["lora_name"] = lora[lora_index]
            #设置lora权重
            prompt["44"]["inputs"]["strength_model"] = lora_weights[lora_index]
            prompt["44"]["inputs"]["strength_clip"] = lora_weights[lora_index]
            pre_keyvalue = pre_keyvalue + ","+lora_keyvalue[lora_index]
    else:
        prompt["44"]["inputs"]["strength_model"] = 0
        prompt["44"]["inputs"]["strength_clip"] = 0 #strength_clip

    
    
    
    if batch_size:
        prompt["59"]["inputs"]["amount"] = batch_size
    

    prompt["20"]["inputs"]["image"] = images_controlnet["softedge"] if images_controlnet["softedge"] != "" else "static.png"
    prompt["21"]["inputs"]["image"] = images_controlnet["color"] if images_controlnet["color"] != "" else "static.png"
    prompt["22"]["inputs"]["image"] = images_controlnet["openpose"] if images_controlnet["openpose"] != "" else "static.png"
    prompt["23"]["inputs"]["image"] = images_controlnet["tile"] if images_controlnet["tile"] != "" else "static.png"
    prompt["73"]["inputs"]["image"] = images_controlnet["scribble"] if images_controlnet["scribble"] != "" else "static.png"
    prompt["83"]["inputs"]["image"] = images_controlnet["lineart"] if images_controlnet["lineart"] != "" else "static.png"
    
    
    prompt["5"]["inputs"]["strength"] = controllnet_weights['softedge']
    prompt["8"]["inputs"]["strength"] = controllnet_weights['color']
    prompt["10"]["inputs"]["strength"] = controllnet_weights['openpose']
    prompt["12"]["inputs"]["strength"] = controllnet_weights['tile']
    prompt["71"]["inputs"]["strength"] = controllnet_weights['scribble']
    prompt["85"]["inputs"]["strength"] = controllnet_weights['lineart']
    

    if batch_size >=2:

        randomseed = generate_random_number(12)
        prompt["14"]["inputs"]["seed"] = randomseed
        # prompt["28"]["inputs"]["seed"] = randomseed
        seed = randomseed
    else:
        if seed != -1 and seed != None:
            prompt["14"]["inputs"]["seed"] = seed
            # prompt["28"]["inputs"]["seed"] = seed
        else:
            randomseed = generate_random_number(12)
            prompt["14"]["inputs"]["seed"] = randomseed
            # prompt["28"]["inputs"]["seed"] = randomseed
            seed = randomseed

    # print("----------------------------------",param1)
    import GptTagger
    gpt_success_tag = True
    try:
        tag = GptTagger.tag_image('input/'+init_image_name)
    except Exception as e:
        print("GPT反推失败")
        gpt_success_tag = False
        tag = ""

    if  gpt_success_tag == False:
        #若gpt反推成功，则wd14反推需要关闭
        print("GPT反推成功，关闭WD14")
        prompt["108"]["inputs"]["threshold"] = 0.99
        prompt["108"]["inputs"]["character_threshold"] = 0.99
    
    if (param1 != '' and param1 != None) or lora_index != "0":

        print("Lora预设")
    
        # Set the text you want to translate
        text_to_translate = param1 #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
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
        
        # print(pre_keyvalue)
        #pretext
        prompt["111"]["inputs"]["prompt"] =  pre_keyvalue + ","+ translation+","+tag
        # print("正面提示词：",prompt["47"]["inputs"]["pre_text"])
    
    else:
        # return "no prompt input, please retry",404
        prompt["111"]["inputs"]["prompt"] = pre_keyvalue+","+tag
        # print(prompt["69"]["inputs"]["text"])
    

    if param2 != '' and param2 != None:

        
    
        # Set the text you want to translate
        text_to_translate = param2 #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
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

        
        # print(pre_keyvalue)
        prompt["4"]["inputs"]["text"] =  model_prenegative +  translation
        print("负面提示词：",prompt["4"]["inputs"]["text"])
    
    else:
        prompt["4"]["inputs"]["text"] =  model_prenegative
        


    

    server_address = "127.0.0.1:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    
    
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    # print(images)
    # print(type(images))
    encoded_images = []
    for node_id in images:
        print(node_id)
        index = 0
    
        for image_data in images[node_id]:
            # 将字节数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 将PIL Image对象转换为JPEG格式的字节串
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')

            # 对字节串进行base64编码并转换为ascii字符串
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')  
            
            
            # taggerprompt = ""

            
            imagedict = {"url":encoded_image,"seed":seed,"index":index}
            encoded_images.append(imagedict)
            # seed +=1
            index +=1

            # print(imagedict)
    
    
    # 然后，将所有的encoded_images打包成一个JSON对象
    result = {'images': encoded_images}

    global_queues = global_queues - batch_size
    # print(encoded_images)
    return jsonify(result)


    pass



@app.route('/igamestyle',methods = ['POST'])
@timing_decorator
def igamestyle():

    print("得到igame请求")
    model_index = request.json['CF_model']


    global client_id_index

    client_id = client_ids[client_id_index]

    client_id_index += 1

    if client_id_index >= len(client_ids):
        client_id_index = 0
    
    print(client_id)
    prompt = json.load(open("igameimgtoimg_api.json"))

    input_folder = "input"
    
    init_image_base64 = request.json['init_image']
    

    init_image_name = base64_to_img(init_image_base64)

    prompt['118']['inputs']['image'] = init_image_name
    #使用PIL计算出init_image_name这个图片的宽和长　
    init_image = Image.open(input_folder+'/'+init_image_name)
    image_width, image_height = init_image.size
    print("原图宽度",image_width)
    print("原图高度",image_height)

    

    aspect_ratio = image_width / image_height

    if image_width > image_height:
        # 在这种情况下，高度是短边
        prompt['120']['inputs']['height'] = 512
        # 根据长宽比，计算新的宽度
        prompt['120']['inputs']['width'] = int(512 * aspect_ratio)
    else:
        # 在这种情况下，宽度是短边
        prompt['120']['inputs']['width'] = 512
        # 根据长宽比，计算新的高度
        prompt['120']['inputs']['height'] = int(512 / aspect_ratio)


    if (model_index := request.json['CF_model']):
        
        print("模型索引",model_index)
        model_name = style_models[model_index]
        print(model_name)
        model_prepositive = style_models_prepositive[model_index]
        model_prenegative = style_models_prenegative[model_index]
        pre_keyvalue = model_prepositive

        prompt['1']['inputs']['ckpt_name'] = model_name
    else:
        print("未传入模型！！！！！！！")
        return "no model post in",404

    prompt['148']['inputs']['string'] = pre_keyvalue
    prompt['5']['inputs']['text'] = model_prenegative

    import GptTagger
    tag = GptTagger.tag_image('input/'+init_image_name)

    prompt["149"]["inputs"]["string"] = tag +','
    


    randomseed = generate_random_number(12)
    prompt['16']['inputs']['seed'] = randomseed
    prompt['96']['inputs']['seed'] = randomseed
    server_address = "127.0.0.1:8188"

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    print(prompt)
    
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    # print(images)
    # print(type(images))
    encoded_images = []
    for node_id in images:
        print(node_id)
        index = 0
    
        for image_data in images[node_id]:
            # 将字节数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 将PIL Image对象转换为JPEG格式的字节串
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')

            # 对字节串进行base64编码并转换为ascii字符串
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')  
            print(encoded_image)
            
            # taggerprompt = ""

            seed = randomseed
            imagedict = {"url":encoded_image,"seed":seed,"index":index}
            encoded_images.append(imagedict)
            # seed +=1
            index +=1

            # print(imagedict)
    
    
    # 然后，将所有的encoded_images打包成一个JSON对象
    result = {'images': encoded_images}

    # global_queues = global_queues - batch_size
    # print(encoded_images)
    return jsonify(result)

@app.route('/igamestyle_xl',methods = ['POST','GET'])
@timing_decorator
def igamestyle_xl():

    print("得到igamexl风格转换请求")
    global client_id_index

    client_id = client_ids[client_id_index]

    client_id_index += 1

    if client_id_index >= len(client_ids):
        client_id_index = 0
    
    print(client_id)
    prompt = json.load(open("igameimgtoimg_xl_api.json"))


    input_folder = "input"
    
    init_image_base64 = request.json['init_image']
    


    init_image_name = base64_to_img(init_image_base64)



    prompt['98']['inputs']['image'] = init_image_name
    #使用PIL计算出init_image_name这个图片的宽和长　
    init_image = Image.open(input_folder+'/'+init_image_name)
    image_width, image_height = init_image.size
    print("原图宽度",image_width)
    print("原图高度",image_height)


    print("到这里吗？")
    

    aspect_ratio = image_width / image_height

    if image_width > image_height:
        # 在这种情况下，高度是短边
        prompt['133']['inputs']['height'] = 1024
        # 根据长宽比，计算新的宽度
        prompt['133']['inputs']['width'] = int(1024 * aspect_ratio)
    else:
        # 在这种情况下，宽度是短边
        prompt['133']['inputs']['width'] = 1024
        # 根据长宽比，计算新的高度
        prompt['133']['inputs']['height'] = int(1024 / aspect_ratio)


    if (model_index := request.json['CF_model']):
        
        print("模型索引",model_index)
        model_name = style_models[model_index]
        print(model_name)
        model_prepositive = style_models_prepositive[model_index]
        model_prenegative = style_models_prenegative[model_index]
        pre_keyvalue = model_prepositive

        prompt['14']['inputs']['ckpt_name'] = model_name

        #此部分有特殊考虑。只有模型是igamexl才能用细节增强lora,否则都不能用
        if model_index != "6":
            prompt["140"]["inputs"]["strength_model"] = 0
    else:
        print("未传入模型！！！！！！！")
        return "no model post in",404

    #prompt['148']['inputs']['string'] = pre_keyvalue
    #prompt['5']['inputs']['text'] = model_prenegative


    #xl的风格转绘暂时不考虑gpt推图，先用wd推图
    # import GptTagger
    # tag = GptTagger.tag_image('input/'+init_image_name)

    # prompt["149"]["inputs"]["string"] = tag +','
    


    randomseed = generate_random_number(15)
    prompt['36']['inputs']['seed'] = randomseed
    prompt['162']['inputs']['seed'] = randomseed
    server_address = "127.0.0.1:8188"

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    print(prompt)
    
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    # print(images)
    # print(type(images))
    encoded_images = []
    for node_id in images:
        print(node_id)
        index = 0
    
        for image_data in images[node_id]:
            # 将字节数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 将PIL Image对象转换为JPEG格式的字节串
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')

            # 对字节串进行base64编码并转换为ascii字符串
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')  
            print(encoded_image)
            
            # taggerprompt = ""

            seed = randomseed
            imagedict = {"url":encoded_image,"seed":seed,"index":index}
            encoded_images.append(imagedict)
            # seed +=1
            index +=1

            # print(imagedict)
    
    
    # 然后，将所有的encoded_images打包成一个JSON对象
    result = {'images': encoded_images}

    # global_queues = global_queues - batch_size
    # print(encoded_images)
    return jsonify(result)


# Function to replace standalone 'a' or 'A' with '1'
def replace_standalone_a(text):
    return re.sub(r'\b[Aa]\b', '1', text)




def generate_random_number(length):
    if length < 1:
        raise ValueError("Length cannot be less than 1")

    first_digit = random.randint(1, 9)  # ensure the first digit is not zero
    other_digits = [random.randint(0, 9) for _ in range(length - 1)]

    # combine all digits into a single number
    random_number = int(str(first_digit) + ''.join(map(str, other_digits)))

    return random_number
def queue_prompt(prompt,client_id,server_address):
    print(client_id,"888888888888888888888888888888888888")
    p = {"prompt": prompt, "client_id": client_id}
    
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    print("prompt执行完")
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type,server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id,server_address):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt,client_id,server_address):
    prompt_id = queue_prompt(prompt,client_id,server_address)['prompt_id']
    print(prompt_id)
    output_images = {}
    while True:
        out = ws.recv() 
        # print(out)
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data
    #print("在获取历史记录之前")
    history = get_history(prompt_id,server_address)[prompt_id]
    # for o in history['outputs']:
    #     print(o)
    filenamelist = []
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        #print("node_output",node_output)
        if 'images' in node_output:
            images_output = []
            
            for image in node_output['images']:
                filenamelist.append(image['filename'])
                image_data = get_image(image['filename'], image['subfolder'], image['type'],server_address)
                images_output.append(image_data)
            output_images[node_id] = images_output
        if 'gifs' in node_output:
            videos_output = []
            for video in node_output['gifs']:
                filenamelist.append(video['filename'])
                # video_data = get_image(video['filename'], video['subfolder'], video['type'],server_address)
                # videos_output.append(video_data)
            # output_images[node_id] = videos_output
    
   
    return output_images,filenamelist


    # print(images)

    # Commented out code to display the output images:

    # for node_id in images:
    #     for image_data in images[node_id]:
    #         from PIL import Image
    #         import io
    #         image = Image.open(io.BytesIO(image_data))
    #         image.show()


@app.route('/linetoimg',methods = ['POST'])
@timing_decorator
def line2img():
    #---------------------------------------------------------------------
    result = workmain(Workdiy.Line2img,request)
    # print(encoded_images)
    return jsonify(result)
    

    #---------------------------------------------------------------------
    

    pass


@app.route('/linetoimg_nolora',methods = ['POST'])
@timing_decorator
def line2img_nolora():
    #---------------------------------------------------------------------
    print("line2img_nolora进入")
    result = workmain(Workdiy.Line2img_nolora,request)
    # print(encoded_images)
    return jsonify(result)
    

    #---------------------------------------------------------------------
    

    pass

@app.route('/igamexihua',methods = ['POST'])
@timing_decorator
def igamexihua():
    #---------------------------------------------------------------------
    print("igamexihua进入")
    result = workmain(Workdiy.IgameFineTune,request)

    # print(encoded_images)
    return jsonify(result)


@app.route('/itemcreate',methods = ['POST'])
@timing_decorator
def itemcreate():
    #---------------------------------------------------------------------
    result = workmain(Workdiy.ItemCreate,request)

    # print(encoded_images)
    return jsonify(result)

@app.route('/creativedivergence',methods = ['POST'])
@timing_decorator
def creativedivergenceb():
    #---------------------------------------------------------------------
    result = workmain(Workdiy.Ideabomb,request)

    # print(encoded_images)
    return jsonify(result)

@app.route("/makepose",methods = ['POST'])
@timing_decorator
def makepose():
    #---------------------------------------------------------------------
    print("makepose start")
    result = workmain(Workdiy.MakePose,request)

    # print(encoded_images)
    return jsonify(result)


@app.route('/rem',methods = ['POST'])
@timing_decorator
def rem():
    #---------------------------------------------------------------------
    result = workmain(Workdiy.Rem,request)

    # print(encoded_images)
    return jsonify(result)

@app.route('/multimirror',methods = ['POST'])
@timing_decorator
def multimirror():
    #从request中取出前端上传的zip压缩包文件
    zip_file = request.files['zip_file']
    print(zip_file)

    file_list = get_filenames_in_zip(zip_file)

    #file_list的长度超过50，则向前端返回一个长度超过的限制标志
    if (length:=len(file_list)) > 50:
        return jsonify({"error": "Too many files in the zip file","length":length})
    print(file_list)
    unzipfile(zip_file)

    client_id = str(uuid.uuid4())
    server_address = "127.0.0.1:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    workobj = Workdiy.Fenjing()
    print("在初始化参数之前")
    all_images = [] 
    # print(request.json)
    for i in range(len(file_list)):
        print("开始执行")
        filename = file_list[i]
        workobj.init_params(request,filename)
        prompt = workobj.set_params_to_comfyui()
        images,filenamelist = get_images(ws, prompt,client_id,server_address)
        for item in filenamelist:
            all_images.append(item)
    
    #将all_images中的文件打包为一个zip压缩文件，all_images中是文件名且都在input文件夹下
    # 生成一个 12 位的随机字符串
    random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(20))

    # 将这个随机字符串作为压缩文件的文件名，并放在 output 文件夹下
    zip_filename = "output/" + random_string + ".zip"
    with zipfile.ZipFile(zip_filename, "w") as zip_file:
        for filename in all_images:
            zip_file.write('output'+'/'+filename, arcname=filename)
            
    #将打包后的zip压缩文件返回给前端
    # return send_file(zip_filename, as_attachment=True)
    print("打包后的文件名为：",zip_filename)

    return jsonify({"zip_filename": random_string + ".zip","length":length})


@app.route('/downloads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory='output', path=filename, as_attachment=True)

@app.route('/video2video',methods = ['POST'])
def video2video():
    videoobj = request.files['video']
    # 生成一个随机的文件名，确保唯一性
    filename = str(uuid.uuid4())
    print(filename)
    # 获取文件的扩展名
    ext = os.path.splitext(videoobj.filename)[1]
    # 拼接文件保存的路径和文件名
    filepath = os.path.join('input', filename + ext)
    # 保存视频文件
    videoobj.save(filepath)


    url = 'http://10.3.20.117:8188/upload/image'
    # 以下是模拟表单数据，根据实际情况修改
    data = { 
        "overwrite": "true",  # 是否覆盖同名文件（可选）
    }

    # 替换为实际的图像文件路径
    files = {
        "image": open(filepath, "rb")
    }

    response = requests.post(url, data=data, files=files)

    # 输出服务器返回的JSON响应
    print(response.json())

    filename = response.json()['name']

    client_id = str(uuid.uuid4())
    server_address = "10.3.20.117:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    workobj = Workdiy.Video2Video()
    print(filename)
    workobj.init_params("",filename)
    prompt = workobj.set_params_to_comfyui()
    # print(prompt)
    print("准备视频转绘")
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    print(filenamelist)

    generated_mp4_name = filenamelist[0]
    

    return jsonify({"mp4": generated_mp4_name})

@app.route('/video2video_lora',methods = ['POST'])
def video2video_load():
    videoobj = request.files['video']
    # 生成一个随机的文件名，确保唯一性
    filename = str(uuid.uuid4())
    print(filename)
    # 获取文件的扩展名
    ext = os.path.splitext(videoobj.filename)[1]
    # 拼接文件保存的路径和文件名
    filepath = os.path.join('input', filename + ext)
    # 保存视频文件
    videoobj.save(filepath)


    url = 'http://10.3.20.117:8188/upload/image'
    # 以下是模拟表单数据，根据实际情况修改
    data = { 
        "overwrite": "true",  # 是否覆盖同名文件（可选）
    }

    # 替换为实际的图像文件路径
    files = {
        "image": open(filepath, "rb")
    }

    response = requests.post(url, data=data, files=files)

    # 输出服务器返回的JSON响应
    print(response.json())

    filename = response.json()['name']

    client_id = str(uuid.uuid4())
    server_address = "10.3.20.117:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    workobj = Workdiy.Video2Video_Lora()
    print(filename)
    workobj.init_params(request,filename)
    prompt = workobj.set_params_to_comfyui()
    # print(prompt)
    print("准备视频转绘")
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    print(filenamelist)

    generated_mp4_name = filenamelist[0]
    
    

    return jsonify({"mp4": generated_mp4_name})


@app.route('/text2image_chatgpt',methods = ['POST'])
def text2image_chatgpt():

    text = request.json['prompt']
    result = GptTagger.text2image(text)

    return jsonify({"url": result})
    
    pass
@app.route('/standard_chat',methods = ['POST'])
def standard_chat():
    text = request.json['prompt']

    try:
        name = request.json['uname']
        print(f"gpt传进来了 : {name}" )
    except Exception as e:
        name = "Lilith"
    my_chat = {"role":"user","content":text}
    mongodb.add_data(name,my_chat)


    history = mongodb.get_last_10(name)
    result = GptTagger.standard_chat(text,history)

    
    ta_chat = {"role":"assistant","content":result}
    mongodb.add_data(name,ta_chat)


    return jsonify({"message": result})
    pass

@app.route('/analyze_file',methods = ['POST'])
def analyze_file():

    file = request.files['file']
    text = request.form.get('prompt')
    #将file保存到input文件夹下
    file.save(os.path.join('input', file.filename))
    #获取文件的绝对路径
    file_path = os.path.join('input', file.filename)
    #调用GptTagger类的analyze_file方法
    result = GptTagger.analyze_file(text,file_path)
    return jsonify({"message": result})

    pass

@app.route('/magic_animate',methods = ['POST'])
def magic_animate():
    videoobj = request.files['video']
    # 生成一个随机的文件名，确保唯一性
    video_filename = str(uuid.uuid4())
    print(video_filename)
    # 获取文件的扩展名
    ext = os.path.splitext(videoobj.filename)[1]
    # 拼接文件保存的路径和文件名
    filepath = os.path.join('input',video_filename + ext)
    # 保存视频文件
    videoobj.save(filepath)

    imageobj = request.files['image']
    # 生成一个随机的文件名，确保唯一性
    image_filename = str(uuid.uuid4())
    print(image_filename)
    # 获取文件的扩展名
    ext = os.path.splitext(imageobj.filename)[1]
    # 拼接文件保存的路径和文件名
    imagepath = os.path.join('input',image_filename + ext)
    # 保存视频文件
    imageobj.save(imagepath)


    url = 'http://10.3.20.117:8188/upload/image'
    # 以下是模拟表单数据，根据实际情况修改
    data = { 
        "overwrite": "true",  # 是否覆盖同名文件（可选）
    }

    # 替换为实际的图像文件路径
    files = {
        "image": open(filepath, "rb")
    }

    response = requests.post(url, data=data, files=files)

    # 输出服务器返回的JSON响应
    print(response.json())

    video_filename = response.json()['name']
    files = {
        "image": open(imagepath, "rb")
    }
    response = requests.post(url, data=data, files=files)
    # 输出服务器返回的JSON响应
    print(response.json())
    image_filename = response.json()['name']

    filenames = [video_filename,image_filename]

    client_id = str(uuid.uuid4())
    server_address = "10.3.20.117:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    workobj = Workdiy.MagicAnimate()
    print(filenames)
    workobj.init_params(request,filenames)
    prompt = workobj.set_params_to_comfyui()
    # print(prompt)
    print("开始magicanimate")
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    print(filenamelist)

    generated_mp4_name = filenamelist[0]
    
    

    return jsonify({"mp4": generated_mp4_name})


@app.route('/get_tips_id',methods = ['POST','GET'])
def get_tips_id():

    # \PUBAIGC\\pubaigc.lilith.com\Tips
    #列出\PUBAIGC\\pubaigc.lilith.com\Tips下所有文件的名字
    tips_dir = r'\\pubaigc.lilith.com\PUBAIGC\Tips'
    tips_files = os.listdir(tips_dir)
    #按照文件的创建时间从早到晚排序
    tips_files.sort(key=lambda x: os.path.getctime(os.path.join(tips_dir, x)))

    
    print(tips_files)

    return jsonify({"files_id": tips_files})
    pass
@app.route('/get_tips_files/<path:file_id>',methods = ['POST','GET'])
def get_tips_files(file_id):


    # ip = request.remote_addr
    # if ip in ['10.3.20.103']:
    #     abort(200)

    try:
        # 在这里根据文件路径获取文件
        # 假设文件都存储在一个名为 'files' 的文件夹中
        root_directory = 'Q:/Tips/'+str(file_id)
        #遍历root_directory中的所有文件
        print(os.listdir(root_directory))
        # 找出第一个os.listdir(root_directory)中图片
        for file in os.listdir(root_directory):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.JPEG'):
                image_path = os.path.join(root_directory, file)
                break
        # 使用send_file函数将文件发送给客户端
        return send_file(image_path, as_attachment=False)
    except Exception as e:
        return str(e)

    pass

@app.route('/uploadimage/unityinput',methods = ['POST'])
def uploadimage():

    ip_adr = request.remote_addr
    print("unity端ip地址:",ip_adr)
    user_agent = request.headers.get('User-Agent')

    image = request.files['image']
    filename = str(uuid.uuid4())

    image.save(os.path.join('unityinput', filename+".png"))

    return  "success"
    pass

@app.route('/igamexihualora',methods = ['POST'])
@timing_decorator
def igamexihua_lora():
    #---------------------------------------------------------------------
    print("igamexihua_lora进入")
    result = workmain(Workdiy.IgameFineTuneLora,request)

    # print(encoded_images)
    return jsonify(result)

@app.route('/merge_images', methods=['POST'])
def merge_images_route():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    images = request.files.getlist('images')
    image_paths = []
    for image in images:
        image.save('input/'+image.filename)
        image_paths.append('input/'+image.filename)

    merged_image = merge_images(image_paths)

    merged_image_filename = 'output/'+str(uuid.uuid4())+'.png'
    merged_image.save(merged_image_filename)
    print("即将传回合成的图片")
    return send_file(merged_image_filename, mimetype='image/png', as_attachment=True), 200

    #return jsonify({'merged_image': merged_image_filename}), 200

@app.route('/texttoimg_xl',methods=['POST'])
@timing_decorator
def texttoimg_xl():

    global client_id_index

    client_id = client_ids[client_id_index]

    client_id_index += 1

    if client_id_index >= len(client_ids):
        client_id_index = 0
    
    print(client_id)

   
    # print(request.json)
    # print("dafsfcjajdkahkdahkfhaknd")
    # Define the folder where you want to save the images
    input_folder = "input"
    prompt = json.load(open("texttoimg_xl_api.json"))

    

    pre_keyvalue = "TRQ style,"
    # print(request.json)
    #打印出request的所有属性
    # print(dir(request))

    # print(request.json)
    #四张controllnet图

    if (model_index := request.json['CF_model']):
        print("模型索引",model_index)

        model_name = style_models[model_index]
        print("模型名称",model_name)
        #model_prepositive = style_models_prepositive[model_index]
        #model_prenegative = style_models_prenegative[model_index]
        #print("模型负面",model_prenegative)
        #pre_keyvalue = model_prepositive

        prompt['1']['inputs']['ckpt_name'] = model_name
    else:
        print("未传入模型！！！！！！！")
        return "no model post in",404

    
    

    images_controlnet,controllnet_weights = init_controlnet()

    
    #一个seed
    seed = request.json['seed']
    seed = int(seed)
    #一个prompt
    param1 = request.json['positive'] #正面提示词
    print(param1)

    param2 = request.json['negative'] #负面提示词
    print(param2)


    #一个lora索引
    lora_index = request.json['lora_index']
    #get image_width
    image_width = request.json['image_width']
    #get image_height
    image_height = request.json['image_height']
    #get batch_size
    batch_size = request.json['batch_size']
    batch_size = int(batch_size)
    if batch_size>5:
        batch_size = 5

    global global_queues
    global_queues = global_queues +batch_size


    print(global_queues,"----------------------------------")

    

    # Loop through the files
    # for index,file in enumerate(files):
    #     if file:
    #         # Generate a random filename
    #         random_filename = secrets.token_hex(8) + '.jpg'
    #         file.save(os.path.join(input_folder, random_filename))
    #         controllnet_images[index] = random_filename
    
    # print(controllnet_images)

    

    if lora_index:
        if lora_index == "0":
            prompt["44"]["inputs"]["strength_model"] = 0
            prompt["44"]["inputs"]["strength_clip"] = 0
        else:
            print(lora_index)
            prompt["44"]["inputs"]["lora_name"] = lora[lora_index]
            #设置lora权重
            prompt["44"]["inputs"]["strength_model"] = lora_weights[lora_index]
            prompt["44"]["inputs"]["strength_clip"] = lora_weights[lora_index]
            pre_keyvalue = pre_keyvalue + ","+lora_keyvalue[lora_index]
    else:
        prompt["44"]["inputs"]["strength_model"] = 0
        prompt["44"]["inputs"]["strength_clip"] = 0 #strength_clip


    if image_width:
        prompt["71"]["inputs"]["width"] = image_width*2
        # prompt["33"]["inputs"]["width"] = image_width*2
        # print(image_width*2,"adefkosojfvksnfvjn")
    if image_height:
        prompt["71"]["inputs"]["height"] = image_height*2
        # prompt["33"]["inputs"]["height"] = image_height*2
    
    if batch_size:
        prompt["71"]["inputs"]["batch_size"] = batch_size
    

    prompt["20"]["inputs"]["image"] = images_controlnet["softedge"] if images_controlnet["softedge"] != "" else "static.png"
    prompt["21"]["inputs"]["image"] = images_controlnet["color"] if images_controlnet["color"] != "" else "static.png"
    prompt["22"]["inputs"]["image"] = images_controlnet["openpose"] if images_controlnet["openpose"] != "" else "static.png"
    prompt["23"]["inputs"]["image"] = images_controlnet["tile"] if images_controlnet["tile"] != "" else "static.png"
    prompt["72"]["inputs"]["image"] = images_controlnet["scribble"] if images_controlnet["scribble"] != "" else "static.png"
    
    prompt["5"]["inputs"]["strength"] = controllnet_weights['softedge']
    prompt["8"]["inputs"]["strength"] = controllnet_weights['color']
    prompt["10"]["inputs"]["strength"] = controllnet_weights['openpose']
    prompt["12"]["inputs"]["strength"] = controllnet_weights['tile']
    prompt["74"]["inputs"]["strength"] = controllnet_weights['scribble']
    
    

    if batch_size >=2:

        randomseed = generate_random_number(12)
        prompt["14"]["inputs"]["seed"] = randomseed
        # prompt["28"]["inputs"]["seed"] = randomseed
        seed = randomseed
    else:
        if seed != -1 and seed != None:
            prompt["14"]["inputs"]["seed"] = seed
            # prompt["28"]["inputs"]["seed"] = seed
        else:
            randomseed = generate_random_number(12)
            prompt["14"]["inputs"]["seed"] = randomseed
            # prompt["28"]["inputs"]["seed"] = randomseed
            seed = randomseed

    # print("----------------------------------",param1)

    if (param1 != '' and param1 != None) or lora_index != "0":

        
    
        # Set the text you want to translate
        param1 = GptTagger.Tagger().translate_gpt(param1)
        

        # 将param1翻译为英文
        
        # print(pre_keyvalue)
        prompt["69"]["inputs"]["text"] =  pre_keyvalue +  param1
        print("正面提示词：",prompt["69"]["inputs"]["text"])
    
    else:
        # return "no prompt input, please retry",404
        prompt["69"]["inputs"]["text"] = pre_keyvalue +", solo, 1girl,"
        print(prompt["69"]["inputs"]["text"])
    

    if param2 != '' and param2 != None:

        
    
        # Set the text you want to translate
        text_to_translate = param2 #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
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

        model_prenegative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,"
        # print(pre_keyvalue)
        prompt["4"]["inputs"]["text"] =  model_prenegative +  translation
        print("负面提示词：",prompt["4"]["inputs"]["text"])
    
    else:
        model_prenegative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,"
        prompt["4"]["inputs"]["text"] =  model_prenegative
        pass
    



 

    server_address = "127.0.0.1:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    # print(prompt)
 
    
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    # print(images)
    # print(type(images))
    encoded_images = []
    for node_id in images:
        print(node_id)
        index = 0
    
        for image_data in images[node_id]:
            # 将字节数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 将PIL Image对象转换为JPEG格式的字节串
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')

            # 对字节串进行base64编码并转换为ascii字符串
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')  
       
            

            imagedict = {"url":encoded_image,"seed":seed,"index":index}
            encoded_images.append(imagedict)
            # seed +=1
            index +=1

            # print(imagedict)
    
    
    # 然后，将所有的encoded_images打包成一个JSON对象
    result = {'images': encoded_images}
    print("文生图返回张数：",len(encoded_images))
    global_queues = global_queues - batch_size
    # print(encoded_images)
    return jsonify(result)


@app.route('/imgtoimg_xl',methods = ['POST'])
@timing_decorator
def imgtoimg_xl():

    global client_id_index

    client_id = client_ids[client_id_index]

    client_id_index += 1

    if client_id_index >= len(client_ids):
        client_id_index = 0
    

    print("得到请求")
    # print(request.json)
    # print("dafsfcjajdkahkdahkfhaknd")
    # Define the folder where you want to save the images
    input_folder = "input"
    prompt = json.load(open("imgtoimg_xl_api.json"))

  

    
    # print(request.json)
    #打印出request的所有属性
    # print(dir(request))

    # print(request.json)
    #四张controllnet图
    if (model_index := request.json['CF_model']):
        
        print("模型索引",model_index)
        model_name = style_models[model_index]
        print(model_name)
        #model_prepositive = style_models_prepositive[model_index]
        #model_prenegative = style_models_prenegative[model_index]
        #pre_keyvalue = model_prepositive

        prompt['1']['inputs']['ckpt_name'] = model_name
    else:
        print("未传入模型！！！！！！！")
        return "no model post in",404
    

    init_image_base64 = request.json['init_image']

    
    init_image_name = base64_to_img(init_image_base64)
    prompt['45']['inputs']['image'] = init_image_name

    #使用PIL计算出init_image_name这个图片的宽和长　
    init_image = Image.open(input_folder+'/'+init_image_name)
    image_width, image_height = init_image.size
    print("原图宽度",image_width)
    print("原图高度",image_height)

    
    aspect_ratio = image_width / image_height

    if image_width > image_height:
        # 在这种情况下，高度是短边
        prompt['98']['inputs']['height'] = 1024
        # 根据长宽比，计算新的宽度
        prompt['98']['inputs']['width'] = int(1024 * aspect_ratio)
    else:
        # 在这种情况下，宽度是短边
        prompt['98']['inputs']['width'] = 1024
        # 根据长宽比，计算新的高度
        prompt['98']['inputs']['height'] = int(1024 / aspect_ratio)




    ratio = request.json['ratio']




    # #以下是目标尺寸
    # if image_width:
    #     # prompt["71"]["inputs"]["width"] = image_width
    #     prompt["33"]["inputs"]["width"] = image_width*ratio
    #     print(image_width*2,"adefkosojfvksnfvjn")
    # if image_height:
    #     # prompt["71"]["inputs"]["height"] = image_height
    #     prompt["33"]["inputs"]["height"] = image_height*ratio

    

    print("原图名字",init_image_name)

    Denoising_strength = request.json['Denoising_strength']
    prompt['14']['inputs']['denoise'] = (Denoising_strength)
    # prompt['28']['inputs']['denoise'] = min(0.45,Denoising_strength)

    print("发散强度",Denoising_strength)
    
    # ratio = 1

    images_controlnet,controllnet_weights = init_controlnet()
    

    #一个seed
    seed = request.json['seed']
    seed = int(seed)
    #一个prompt
    param1 = request.json['positive']
    print(param1)

    param2 = request.json['negative']
    print(param2)


    #一个lora索引
    lora_index = request.json['lora_index']
    #get image_width
    # image_width = request.json['image_width']
    #get image_height
    # image_height = request.json['image_height']
    #get batch_size
    batch_size = request.json['batch_size']
    batch_size = int(batch_size)
    if batch_size>5:
        batch_size = 5

    global global_queues
    global_queues = global_queues +batch_size


    

    # Loop through the files
    # for index,file in enumerate(files):
    #     if file:
    #         # Generate a random filename
    #         random_filename = secrets.token_hex(8) + '.jpg'
    #         file.save(os.path.join(input_folder, random_filename))
    #         controllnet_images[index] = random_filename
    
    # print(controllnet_images)

    if lora_index:
        if lora_index == "0":
            prompt["44"]["inputs"]["strength_model"] = 0
            prompt["44"]["inputs"]["strength_clip"] = 0
        else:
            print(lora_index)
            prompt["44"]["inputs"]["lora_name"] = lora[lora_index]
            #设置lora权重
            prompt["44"]["inputs"]["strength_model"] = lora_weights[lora_index]
            prompt["44"]["inputs"]["strength_clip"] = lora_weights[lora_index]
            pre_keyvalue = pre_keyvalue + ","+lora_keyvalue[lora_index]
    else:
        prompt["44"]["inputs"]["strength_model"] = 0
        prompt["44"]["inputs"]["strength_clip"] = 0 #strength_clip

    
    
    
    if batch_size:
        prompt["59"]["inputs"]["amount"] = batch_size
    

    


 
    prompt["20"]["inputs"]["image"] = images_controlnet["softedge"] if images_controlnet["softedge"] != "" else "static.png"
    prompt["21"]["inputs"]["image"] = images_controlnet["color"] if images_controlnet["color"] != "" else "static.png"
    prompt["22"]["inputs"]["image"] = images_controlnet["openpose"] if images_controlnet["openpose"] != "" else "static.png"
    prompt["23"]["inputs"]["image"] = images_controlnet["tile"] if images_controlnet["tile"] != "" else "static.png"
    prompt["73"]["inputs"]["image"] = images_controlnet["scribble"] if images_controlnet["scribble"] != "" else "static.png"
    prompt["83"]["inputs"]["image"] = images_controlnet["depth"] if images_controlnet["depth"] != "" else "static.png"
    
    
    prompt["5"]["inputs"]["strength"] = controllnet_weights['softedge']
    prompt["8"]["inputs"]["strength"] = controllnet_weights['color']
    prompt["10"]["inputs"]["strength"] = controllnet_weights['openpose']
    prompt["12"]["inputs"]["strength"] = controllnet_weights['tile']
    prompt["71"]["inputs"]["strength"] = controllnet_weights['scribble']
    prompt["85"]["inputs"]["strength"] = controllnet_weights['depth']
    
    

    

    if batch_size >=2:

        randomseed = generate_random_number(12)
        prompt["14"]["inputs"]["seed"] = randomseed
        # prompt["28"]["inputs"]["seed"] = randomseed
        seed = randomseed
    else:
        if seed != -1 and seed != None:
            prompt["14"]["inputs"]["seed"] = seed
            # prompt["28"]["inputs"]["seed"] = seed
        else:
            randomseed = generate_random_number(12)
            prompt["14"]["inputs"]["seed"] = randomseed
            # prompt["28"]["inputs"]["seed"] = randomseed
            seed = randomseed

    # print("----------------------------------",param1)
    import GptTagger
    gpt_success_tag = True
    try:
        tag = GptTagger.tag_image('input/'+init_image_name)
    except Exception as e:
        print("GPT反推失败")
        gpt_success_tag = False
        tag = ""

    if  gpt_success_tag == False:
        #若gpt反推成功，则wd14反推需要关闭
        print("GPT反推成功，关闭WD14")
        prompt["108"]["inputs"]["threshold"] = 1
        prompt["108"]["inputs"]["character_threshold"] = 1
    
    if (param1 != '' and param1 != None) or lora_index != "0":

        print("Lora预设")
    
        # # Set the text you want to translate
        # text_to_translate = param1 #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
        # # Encode the text, returning a dictionary with tensors ready to feed the model
        # encoded_text = tokenizer.encode(text_to_translate, return_tensors="pt")
        # # Generate a translation. This returns a tensor with the predicted token ids
        # translated_tokens = model.generate(encoded_text)
        # # Decode the tokens to get the string of the translation
        # # Decode the tokens to get the string of the translation
        # translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # # Replace 'A' and 'a' in the translation with '1'
        # translation = replace_standalone_a(translation)
        # print(translation)
        
        # print(pre_keyvalue)
        #pretext
        param1 = GptTagger.Tagger().translate_gpt(param1)
        pre_keyvalue = "TRQ style,"
        prompt["132"]["inputs"]["prompt"] =  pre_keyvalue + ","+ param1+","+tag
        # print("正面提示词：",prompt["47"]["inputs"]["pre_text"])
    
    else:
        # return "no prompt input, please retry",404
        pre_keyvalue = "TRQ style,"
        prompt["132"]["inputs"]["prompt"] = pre_keyvalue+","+tag
        # print(prompt["69"]["inputs"]["text"])
    

    if param2 != '' and param2 != None:

        
    
        # Set the text you want to translate
        text_to_translate = param2 #"一个女孩，半身像，笑脸，大头，可爱风，黑头发，黑眼睛，背景冰山与冰川，黄色小熊，粉色上衣"
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

        
        # print(pre_keyvalue)
        model_prenegative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,"
        prompt["4"]["inputs"]["text"] =  model_prenegative +  translation
        print("负面提示词：",prompt["4"]["inputs"]["text"])
    
    else:
        model_prenegative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,"

        prompt["4"]["inputs"]["text"] =  model_prenegative
        


  

    server_address = "127.0.0.1:8188"
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
   
    
    images,filenamelist = get_images(ws, prompt,client_id,server_address)
    # print(images)
    # print(type(images))
    encoded_images = []
    for node_id in images:
        print(node_id)
        index = 0
    
        for image_data in images[node_id]:
            # 将字节数据转换为PIL Image对象
            image = Image.open(io.BytesIO(image_data))

            # 将PIL Image对象转换为JPEG格式的字节串
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')

            # 对字节串进行base64编码并转换为ascii字符串
            encoded_image = base64.b64encode(byte_arr.getvalue()).decode('ascii')  
            
            
            # taggerprompt = ""

            
            imagedict = {"url":encoded_image,"seed":seed,"index":index}
            encoded_images.append(imagedict)
            # seed +=1
            index +=1

            # print(imagedict)
    

    #工具里可能没有layer参数
    try:
        layer = request.json['layer']
        result = {
        'images': encoded_images,
        'layer':layer,
        }
    except Exception as e:
        result = {
        'images': encoded_images,
        }
        pass
    # 然后，将所有的encoded_images打包成一个JSON对象
    

    global_queues = global_queues - batch_size
    # print(encoded_images)
    return jsonify(result)


    pass


@app.route('/structer_lcm',methods = ['POST'])
def structer_lcm():

    #target_requirement = request.form['target_requirement']
    Elements = request.json['Elements']
    Interaction = request.json['Interaction']
    Composition = request.json['Composition']
    Lighting = request.json['Lighting']
    Visual_Hierarchy = request.json['Visual_Hierarchy']
    Atmospheric_Expression = request.json['Atmospheric_Expression']
    Logo = request.json['Logo']
    
    print(Elements)
    print(Interaction)
    
    
    example_texts = []
    #读取名为kv文件夹下的所有以.txt结尾的文本文件内容，存储到以example_texts =[] 的列表中。
    for filename in os.listdir('kv'):
        if filename.endswith('.txt'):
            with open(os.path.join('kv', filename), 'r',encoding='utf-8') as f:
                example_texts.append(f.read())


    #print(example_texts[5])
    #在example_texts = []中随机选取4个元素

    random_texts = random.sample(example_texts, 4)

    #print(random_texts)

    control_prompt_for_gpt35 = f'''
    Key Visual Definition and Design Elements

    In the field of graphic design, Key Visual is an important design element commonly used to represent a brand, product, or event, conveying its core message and image. Below is a detailed introduction to Key Visual:

    1. Definition and Function:
    - Definition: Key Visual refers to the most important and core visual element used in advertising, brand promotion, and event marketing.
    - Function:
    - Reinforce brand image: Key Visual showcases the core features, ideas, and image of the brand.
    - Increase brand recognition: Unique and attractive Key Visuals can help a brand stand out in a competitive market.
    - Convey information: Through images and design elements, Key Visuals convey the themes, ideas, and characteristics of a product or event.
    - Attract target audience: Well-designed Key Visuals can grab the attention of the target audience and spark their interest.

    2. Design Elements:
    - Image or graphic: Usually represents the brand, product, or event.
    - Color: Color selection is crucial for Key Visuals as it directly impacts the visual communication effect and emotional expression.
    - Typography: Font choice is also a key factor in design; appropriate fonts can enhance the overall feel of the Key Visual and its message delivery.
    - Layout: How to arrange and organize elements such as images and text to achieve the best visual effect.
    - Iconic elements: Each brand or event may have its own unique iconic elements, such as logos, symbols, etc., which are often integrated into the Key Visual.
    <dim>
    Elements: {Elements},
    Interaction: {Interaction},
    Composition: {Composition},
    Lighting: {Lighting},
    Visual Hierarchy: {Visual_Hierarchy},
    Atmospheric Expression: {Atmospheric_Expression},
    Logo: {Logo}
    </dim>
    <dim_explain>
    Elements: Characters and Setting [Number of characters, who the main character is, what the setting is (weather, environment, architecture, vegetation, etc.)]
    Interaction: Character Actions [What the main character is doing, interactions between characters]
    Color: Generally, provide a main color tone.
    Composition: Scenery, Perspective, Viewpoint [Mid-shot, close-up/fisheye, wide-angle, human-eye perspective/overhead, looking up, looking straight]
    Lighting: 3D rendering will involve, referring to photographic terms.
    Visual Hierarchy: Typically centered around a specific character, but not necessarily in the center.
    Atmospheric Expression: Emotional atmosphere.
    Logo: A visual representation of the brand or event.
    </dim_explain>
    <example>
    {random_texts[0]}
    </example>
    <example>
    {random_texts[1]}
    </example>
    <example>
    {random_texts[2]}
    </example>
    <example>
    {random_texts[3]}
    </example>
    <prompt>
    {Elements},{Interaction},{Composition},{Lighting},{Visual_Hierarchy},{Atmospheric_Expression},{Logo},
    </prompt>
    你是一位提示词工程师，你仔细学习和模仿<example>标签中提示词的例子，这是一种描述图片构图的提示词。现在，你根据<prompt>中提供的提示词需求，要重点突出人物的位置与结构，人物与景物的空间关系,logo的文字内容和位置，仿照<example>的格式写一个提示词，不要把维度写出来，直接整体的，英文的自然语言，不超过150个英文单词。
    '''

    from openai import AzureOpenAI

    #faxing-gpt4,faxing-gpt35,faxing-gpt4-vp

    client = AzureOpenAI(
    azure_endpoint = "https://faxing.openai.azure.com/", 
    api_key="bb36bca535a54b1bafe3d9e6216a3c4f", 
    api_version="2024-02-15-preview"
    )

    message_text = [{"role":"user","content":control_prompt_for_gpt35}]
    completion = client.chat.completions.create(
    model="faxing-gpt35", # model = "deployment_name"
    messages = message_text,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )

    print(completion.choices[0].message.content) 

    return jsonify({'completion': completion.choices[0].message.content})

    pass


#这里是我们webgl生图接口,角色
@app.route('/unitytool',methods = ['POST'])
@timing_decorator
def unitytool():
    #---------------------------------------------------------------------
    print("unity端生图进入")
    result = workmain(Workdiy.UnityTool,request)

    # print(encoded_images)
    return jsonify(result)


#这里是我们webgl生图接口,场景
@app.route('/unitytool_scene',methods = ['POST'])
@timing_decorator
def unitytool_scene():
    #---------------------------------------------------------------------
    print("unity端生图进入")
    result = workmain(Workdiy.UnityTool_Scene,request)

    # print(encoded_images)
    return jsonify(result)


@app.route('/kv',methods = ['POST'])
def kv():
    #---------------------------------------------------------------------
    print("kv端生图进入")
    result = workmain(Workdiy.KV, request)

    # print(encoded_images)
    return jsonify(result)


def save_base64_image(base64_data, save_path):
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    
    # 检查是否存在重名文件，如果存在则修改文件名
    if os.path.exists(save_path):
        filename, file_extension = os.path.splitext(save_path)
        i = 1
        while os.path.exists(save_path):
            save_path = f"{filename}_{i}{file_extension}"
            i += 1
    
    image.save(save_path, format='PNG')


with open('words.json', 'r') as f:
    words = json.load(f)
def random_words(words):
    # 随机抽取单词
    selected_word = random.choice(words)
    
    return selected_word




@app.route('/partyworkspace',methods = ['POST'])
def afkh5():

    filename = random_words(words)

    request.json['positive'] = f"3d, 3d render,TRQ style,(1 {filename},centered:1.5),isometric,(black background:1.2)"

    result = workmain(Workdiy.AFKh5, request)

    base64 = result['images'][0]['url']

    url = 'http://10.3.20.117:5000/'

    data = {"base64": base64,
            "filename": filename}
    

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("图片保存成功")
        else:
            print(f"文件保存请求失败，状态码：{response.status_code}")
    except Exception as e:
        print("发生异常:", e)

    #将base64格式的图片转化为png图片，存储到指定文件夹内，如果有重名的不能覆盖

    return "dadad"


@app.route('/igame_black_white_draft_c',methods = ['POST'])
def igame_black_white_draft_c():

    print("igame黑白草图端c生图进入")
    result = workmain(Workdiy.IgameBWD, request)

    # print(encoded_images)
    return jsonify(result)

    pass
@app.route('/igame_black_white_draft_s',methods = ['POST'])
def igame_black_white_draft_s():

    print("igame黑白草图端s生图进入")
    result = workmain(Workdiy.IgameBWDS, request)

    # print(encoded_images)
    return jsonify(result)

    pass

@app.route('/partyh5',methods = ['POST'])
def partyh5():


    #创建一个filename，随机值

    filename = str(uuid.uuid4())
    #filename = random_words(words)

    #request.json['positive'] = filename + ",TRQ style,lusila,1girl,heterochromia, yellow eyes,red eyes,white hair,solo,white background,simple background,looking at viewer, upper body,"

    result = workmain(Workdiy.Partyh5, request)

    base64 = result['images'][0]['url']

    url = 'http://10.3.20.117:5000/'

    data = {"base64": base64,
            "filename": filename}
    

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("图片保存成功")
        else:
            print(f"文件保存请求失败，状态码：{response.status_code}")
    except Exception as e:
        print("发生异常:", e)

    #将base64格式的图片转化为png图片，存储到指定文件夹内，如果有重名的不能覆盖

    return "dadad"

@app.route('/get_photos',methods = ['POST'])
def get_photos():
    numbers = request.json['numbers']
    index = int(request.json['index'])

    images_paths = r'\\pubaigc.lilith.com\\PUBAIGC\画廊\\'
    subfolders = [f.path for f in os.scandir(images_paths) if f.is_dir()]

    images_path = subfolders[index]
    # Check if the directory exists
    if not os.path.exists(images_path):
        return jsonify({'error': 'Directory not found'}), 404

    # List only files with jpg, jpeg, and png extensions
    image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Encode images as base64
    encoded_images = []
    for image_file in image_files:
        with open(image_file, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
            encoded_images.append(encoded_image)

    print(len(encoded_images))
    # Randomly select 'numbers' elements from encoded_images if its length is greater than 'numbers'
    if len(encoded_images) > numbers:
        selected_images = random.sample(encoded_images, numbers)
    else:
        selected_images = encoded_images

    return jsonify({'images': selected_images}), 200

    

#此部分想要获取提示词库对应的图片
@app.route('/get_tag_photos',methods = ['POST'])
def get_tag_photos():
    def resize_image(image, max_size=256):
        """
        Resize the image while preserving aspect ratio.
        """
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(max_size * height / width)
            else:
                new_height = max_size
                new_width = int(max_size * width / height)
            return image.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            return image

    numbers = request.json['numbers']
    index = request.json['index']
    # print(index)
    images_paths = {"Igame":r'\\pubaigc.lilith.com\PUBAIGC\aiout\wikiprompt\IG',"Dgame":r'\\pubaigc.lilith.com\PUBAIGC\aiout\wikiprompt\DG',
                    "Party":r'\\pubaigc.lilith.com\PUBAIGC\aiout\wikiprompt\Party',"Samo":r'\\pubaigc.lilith.com\PUBAIGC\aiout\wikiprompt\Samo',}
    images_path = images_paths[index]

    # 检查目录是否存在
    if not os.path.exists(images_path):
        return jsonify({'error': 'Directory not found'}), 404

    # 获取目录中的所有文件
    image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    # 编码图片为 base64 格式，同时保存文件名和对应的文本内容
    encoded_images = []
    for image_file in image_files:
        if image_file.endswith(('.jpg', '.jpeg', '.png','.webp')):  # Assuming image files are jpg, modify if different
            with open(os.path.join(images_path, image_file), "rb") as img_file:
                #img = Image.open(img_file)
                #img = resize_image(img)  # Resize the image


                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                # encoded_images.append(encoded_image)
                # encoded_image = base64.b64encode(img.tobytes()).decode('utf-8')
                file_name = os.path.basename(image_file)  # 提取文件名
                # 将文件名和 base64 编码的图片添加到列表中
                txt_file_path = os.path.join(images_path, os.path.splitext(image_file)[0] + '.txt')
                if os.path.exists(txt_file_path):
                    with open(txt_file_path, 'r') as txt_file:
                        txt_content = txt_file.read()
                else:
                    txt_content = ""
                encoded_images.append({'text_content': txt_content,'base64': encoded_image})

    print(len(encoded_images))

    # 如果列表中的图片数量大于请求的数量，则随机选择指定数量的图片
    if len(encoded_images) > numbers:
        selected_images = random.sample(encoded_images, numbers)
    else:
        selected_images = encoded_images

    return jsonify({'images': selected_images}), 200



@app.route('/upscale_basic',methods = ['POST'])
def upscale_basic():

    result = workmain(Workdiy.UpscaleimgBasic, request)

    # print(encoded_images)
    return jsonify(result)


    pass


#角色低模转kv
@app.route('/clow2kv',methods = ['POST'])
def clow2kv():


    result = workmain(Workdiy.Clow2KV, request)

    # print(encoded_images)
    return jsonify(result)


    pass

@app.route('/igame_color_draft_c',methods = ['POST'])
def igame_color_draft_c():


    result = workmain(Workdiy.IgameColorDraftC, request)

    # print(encoded_images)
    return jsonify(result)

@app.route('/igame_color_draft_s',methods = ['POST'])
def igame_color_draft_s():


    result = workmain(Workdiy.IgameColorDraftS, request)

    # print(encoded_images)
    return jsonify(result)

@app.route('/igame_black_white_color_draft_c',methods = ['POST'])
def igame_black_white_color_draft_c():


    result = workmain(Workdiy.IgameBWCDraftC, request)

    # print(encoded_images)
    return jsonify(result)

@app.route('/afk_color_draft',methods = ['POST'])
def afk_color_draft():


    result = workmain(Workdiy.AFKColorDraft, request)

    # print(encoded_images)
    return jsonify(result)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8250,debug=True)

    #structer_lcm()


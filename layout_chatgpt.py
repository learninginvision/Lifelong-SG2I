import os
import openai
from openai import OpenAI
import csv 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from urllib.request import urlopen
import pandas as pd
import pickle
import json 

client = OpenAI(api_key="your api key")


# openai.api_key = os.getenv("OPENAI_API_KEY")
def read_example_prompts(file_path):
    
    with open(file_path, 'r') as f:
       data = f.read()
    return data

def load_json(path_file):
    with open(path_file) as f:
        data = json.load(f)
    return data

def generate_box_gpt4(inputs):

    if not isinstance(inputs, list):
        message = 'Provide box coordinates for an image with' + inputs
        # message = 'Summarize the following text to 50 words or less' + inputs

        messages = load_json('sg_datasets/exampleSG.json')
        messages.append(
                {"role": "user", "content": message},
            )
    
    else:
        messages = load_json('sg_datasets/exampleSG.json')
        for message_input in inputs:
            message = 'Provide box coordinates for an image with' + message_input
            messages.append(
                {"role": "user", "content": messages},
            ) 
    
    chat = client.chat.completions.create(
            model="gpt-4", messages=messages
        )
    completed_text = chat.choices[0].message.content
    print('text gpt', completed_text)
    boxes = completed_text.split('\n')
    d = {}
    name_objects = []
    boxes_of_object = []
    background_prompt = None
    for b in boxes:
        if b == '': continue
        if not '(' in b and not 'BG' in b: continue 
        b_split = b.split(":")
        if b_split[0] != 'BG':
            name_objects.append(b_split[0])
            boxes_of_object.append(text_list(b_split[1]))
        else:
            background_prompt = b_split[1]
    return name_objects, boxes_of_object, background_prompt

def text_list(text):
    text =  text.replace(' ','')
    text =  text.replace('\n','')
    text =  text.replace('\t','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    return tuple(result)
def read_csv(path_file, t):
    list_prompts = []
    with open(path_file,'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >0: 
                if  row[1] == t:
                    list_prompts.append(row)
    return list_prompts

def read_txt_label(file_path):
    labels = {}
    with open(file_path, 'r') as f:
        for x in f:
            x = x.replace(' \n', '')
            x = x.replace('\n', '')
            x = x.split(',')
            labels.update({x[0]: x[2]})
    return labels

def draw_box(text, boxes,output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    os.makedirs(output_folder, exist_ok=True)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Roboto-LightItalic.ttf", size=20)
    for i, box in enumerate(boxes):
        t = text[i]
        draw.rectangle([(box[0], box[1]),(box[2], box[3])], outline=128, width=2)
        mean_box_x, mean_box_y = int((box[0] + box[2] )/ 2) , int((box[1] + box[3] )/ 2)
        draw.text((mean_box_x, mean_box_y), t, fill=200,font=font )
    image.save(os.path.join(output_folder, img_name))

def save_img(folder_name, img, prompt, iter_id, img_id):
    os.makedirs(folder_name, exist_ok=True)
    img_name = str(img_id) + '_' + str(iter_id) + '_' + prompt.replace(' ','_')+'.jpg'
    img.save(os.path.join(folder_name, img_name))

def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    meta = []
    syn_prompt = []

    for sample in gt_data:
        meta.append([sample['meta_prompt']])
        syn_prompt.append([sample['synthetic_prompt']])
    return meta, syn_prompt

def load_box(pickle_file):
    with open(pickle_file,'rb') as f:
        data = pickle.load(f)
    return data
def read_txt_hrs(filename):
    result = []
    with open(filename) as f: 
        for x in f:
            result.append([x.replace('\n','')])
    return result

def format_box(names, boxes):
    result_name = []
    resultboxes = []
    for i, name in enumerate(names):
        name = remove_numbers(name)
        result_name.append('a ' + name.replace('_',' '))
        if name == 'person': 
            boxes[i] = boxes[i]
        resultboxes.append([boxes[i]])
    return result_name, np.array(resultboxes)

def remove_numbers(text):
    result = ''.join([char for char in text if not char.isdigit()])
    return result
def process_box_phrase(names, bboxes):
    d = {}
    for i, phrase in enumerate(names):
        phrase = phrase.replace('_',' ')
        list_noun = phrase.split(' ')
        for n in list_noun:
            n = remove_numbers(n)
            if not n in d.keys():
                d.update({n:[np.array(bboxes[i])/512]})
            else:
                d[n].append(np.array(bboxes[i])/512)
    return d

def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.','')
    prompt = prompt.replace(',','')
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'s' in prompt_list:
                obj_first_index = prompt_list.index(word+'s') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'es' in prompt_list:
                obj_first_index = prompt_list.index(word+'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
        if in_prompt :
            bbox_to_self_att.append(np.array(name_box[obj]))
        
            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att

if __name__ == '__main__':

    name_objects = ['cat','car','chair']
    boxes_of_object = [[51, 82, 263, 279],[288, 128, 472, 299],[27, 355, 418, 494]]
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code was developed based on the following GUI.
https://github.com/monemati/YOLOv8-DeepSORT-Streamlit/tree/main
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import csv
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle, Arrow
from matplotlib.ticker import *
from matplotlib.font_manager import FontProperties

import subprocess
import argparse
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
import json
import os

def iou(a, b):
    """
    Function to calculate IoU
    """
    # a and b include [xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def Flowcharting1_gencsv(conf, model, st_frame, image, is_display_tracking=None,
                         tracker=None, csv_path=None, cnt=[0]):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    txt_file = "./tempwork/result.txt"
    # Predict the objects in the image using YOLOv8 model
    i=0
    cnt[0] += 1
    if is_display_tracking:
        res = model.track(image, persist=True, tracker=tracker)
        for result in res:
            Answer = []
            i +=1
            #print(str(cnt[0])) # counter
            result_list = result.custom_save_txt()
            for j in result_list: # counter and detected objects are added to Anser
                Ans = str(cnt[0]) +" "+ j
                Answer.append(Ans)
            with open(txt_file, 'a') as f: # Writing detected result
                f.writelines(text + '\n' for text in Answer)
            with open(txt_file, "r") as file:
                f_list = file.readlines()
                file.close()

            A = [] # Detected data
            Title =["frame","class","x_min","y_min","x_max","y_max","ID"]
            A.append(Title)
            for i in f_list:
                sp = i.split() # リストに分割

                val = []                
                for j in sp:
                    value = int(float(j))
                    val.append(value)

                # Translate Class ID to name
                if val[1] == 0:
                    val[1] = "hand"
                elif val[1] == 1:
                    val[1] = "conical beaker"
                elif val[1] == 2:
                    val[1] = "Erlenmeyer flask"
                elif val[1] == 3:
                    val[1] = "reagent bottle"
                elif val[1] == 4:
                    val[1] = "pipette"
                elif val[1] == 5:
                    val[1] = "eggplant shaped flask"
                elif val[1] == 6:
                    val[1] = "separatory funnel"
                else:
                    print("ID Error")
                A.append(val)

            # Obj detection to CSV
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(A)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

    df = pd.read_csv(csv_path)
    reindex = df.reindex(columns=["frame", "x_min", "y_min", "x_max", "y_max", "ID", "class"])
    reindex.to_csv(csv_path, index=False)


def Flowcharting2_genchart(csv_path, video_path):
    ##### Action Recognition
    Ans, seg = ActionRecognition(video_path)
    if Ans["results"][video_path.strip(".mp4")][0]["label"] == "AddingReagent":
        Action = [seg//2, "Adding"]
    elif Ans["results"][video_path.strip(".mp4")][0]["label"] == "Stirring":
        Action = [seg//2, "Stirring"]
    ######

    filename = csv_path
    with open(filename, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        content = [row for row in csvreader] # CSV to list

    key = content[0] # column name
    del content[0] # delete header
    mydict = []
    mydict = [dict(zip(key,item)) for item in content]
    
    def join_values(elem):
        return elem['ID']

    content_sorted = {}
    content_sorted = sorted(mydict, key=join_values) # sort list for ID
    content_sorted_list=[]
    for i in content_sorted:
        content_sorted_list.append(list(i.values()))

    frame = content[-1][0] # last frame number

    # result for every 50 frame
    A=[]
    for i in range(1, int(frame), 50):
        for j in range(len(content_sorted_list)):
            if content_sorted_list[j][0] == str(i):
                A.append(content_sorted_list[j])

    for k in range(len(A)):
        if A[k][6] == "hand": # process for hand
            fr = A[k][0]
            hand_area = [int(A[k][1]), int(A[k][2]), int(A[k][3]), int(A[k][4])]
            for l in range(len(A)):
                if A[l][0] == fr:
                    obj_area = [int(A[l][1]), int(A[l][2]), int(A[l][3]), int(A[l][4])] # obj area
                    if iou(hand_area, obj_area) >= 0.10: # calc IoU
                        A[l][6] = A[l][6] + "*" # label for held

    base = []
    for i in range(len(A)):
        if A[i][-1] in ["hand", "hand*"]:
            pass
        else:
            base.append(["Obj", A[i][0], A[i][-2], A[i][-1]])

    held_id = []
    for i in range(len(base)):
        if base[i][-1][-1] == "*":
            held_id.append(base[i][-2])

    temp = []
    for i in range(len(base)):
        if base[i][-1][-1] == "*":
            if base[i][-2] != held_id[0]:
                continue
        temp.append(base[i])
    base = temp

    for i in range(len(base)):
        if int(base[i][1]) > Action[0]:
            base.insert(i, ["Act", Action[0], "", Action[1]])
            break

    unique_frame = sorted(list(set([int(i[1]) for i in base])))
    #print(unique_frame)

    temp = [[] for _ in range(len(unique_frame))]
    for i in range(len(base)):
        for f in range(len(unique_frame)):
            if int(base[i][1]) == unique_frame[f]:
                temp[f].append(base[i])
    base = temp

    temp = []
    for i in range(len(base)):
        if i == 0:
            temp.append(base[i])
        else:
            change_flag = 0
            for j in range(len(base[i])):
                if j > len(base[i-1]) - 1:
                    temp.append(base[i])
                    continue   
                if base[i][j][2:] != base[i-1][j][2:]:
                    change_flag = 1
            if change_flag == 0:
                pass
            else:
                temp.append(base[i])
    base = temp

    temp = []
    for i in range(len(base)):
        #print(len(base[i]))
        if len(base[i]) == 2:
            if base[i][0][-2] in held_id:
                temp.append([base[i][1], base[i][0]])
            else:
                temp.append(base[i])
        else:
            temp.append(base[i])
    base = temp

    height = len(base)
    width = 1
    for i in range(len(base)):
        width = max(width, len(base[i]))

    #Flowcharting
    fon='./times.ttf'
    fp=FontProperties(fname=fon)

    # definition if the range of the axis
    xmin = 0
    xmax = 2 * 2 + 12 * width + 3 * (width - 1)
    ymin = 0
    ymax = 2 * 2 + 2 * height + 1 * (height - 1)

    Raspect = 0.8
    iw = abs(xmax-xmin)/2
    ih = abs(ymin-ymax)/abs(xmax-xmin)*iw*Raspect

    fig = plt.figure(figsize=(iw,ih),facecolor='w') # figure size
    ax1 = fig.add_subplot(1,1,1, adjustable='box', aspect=Raspect)
    ax1.set_xlim([xmin,xmax]) # x range
    ax1.set_ylim([ymin,ymax]) # y range
    plt.axis('off')

    fsz=22 # font size
    plt.rcParams["font.size"] = fsz
    plt.rcParams['font.family'] ='sans-serif'

    dxtext = 12  # width of textbox
    dytext = 2   # height of textbox
    dyarrow=1 # height of allow

    y_pointer = ymax - 4.
    for n in range(len(base)): # 
        ss = base[n] # single box

        for i in range(len(base[n])): # number of obj for single frame
            xi = 2 + (3 + dxtext) * i # left edge of box
            center_box_x = xi + (dxtext / 2)    # center of text

            if ss[i][0] == "Obj":
                ax1.add_patch(Rectangle((xi, y_pointer), dxtext, dytext, fc='#ffffff',ec='#000000')) # rectangle
                xx = xi + (dxtext / 2)
                ax1.text(xx, y_pointer+1, ss[i][-1]+"  ID: "+ss[i][-2], fontsize=fsz,ha='center',va='center',fontproperties=fp) # text
            
                if n == len(base)-1:
                    pass
                else:
                    plt.plot([xi+6, xi+6], [y_pointer, y_pointer-dyarrow], color="black") # line

            if ss[i][0] == "Act":
                if ss[i][-1] == "Adding":
                    plt.plot([xi+6, xi+6], [y_pointer+2, y_pointer-dyarrow], color="black")

                    r2 = 0
                    for nn in range(n):
                        if len(base[nn]) == 2:
                            r2 = nn
                    llen = n - nn + 1

                    plt.plot([xi+6+(3+dxtext), xi+6+(3+dxtext)], [y_pointer+(3*llen), y_pointer+1], color="black") # 縦線
                    plt.plot([xi+6, xi+6+(3+dxtext)], [y_pointer+1, y_pointer+1], color="black")
                    plt.quiver(xi+7, y_pointer+1, -1, 0, width = 0.0011, scale_units='xy', scale=1) # arrow

                elif ss[i][-1] == "Stirring":
                    ax1.add_patch(Rectangle((xi, y_pointer), dxtext, dytext, fc='#ffffff',ec='#000000')) # rectangle
                    xx = xi + (dxtext / 2)
                    ax1.text(xx, y_pointer+1, ss[i][-1], fontsize=fsz,ha='center',va='center',fontproperties=fp) # text
                    plt.plot([xi+6, xi+6], [y_pointer, y_pointer-dyarrow], color="black") # line
                    pass

        y_pointer -= (dytext + dyarrow) # slide y coord

    fnameF='./tempwork/flowchart.png' # save flowchart
    plt.savefig(fnameF, dpi=300, bbox_inches="tight", pad_inches=0.1) # save

#########################
@st.cache_resource
def load_model(model_path): # select model
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


# Preprocessiong for Videos using FFprobe
def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix:
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()
    ffprobe_cmd.append("./"+str(video_file_path))
    #print(ffprobe_cmd)
    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    #print(res)
    if len(res) < 4:
        return
    
    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
    duration = float(res[3])
    n_frames = int(frame_rate * duration)

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    if n_exist_frames >= n_frames:
        return

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print('\n')


def ActionRecognition(video_path): 
    jpg_path = "./tempwork/videojpg/"
    video_process(Path(video_path), Path(jpg_path), ".mp4")
    labels = ["AddingReagent", "Stirring", "Transferring"]
    
    dst_json_path = Path("./tempwork/arinp.json")
    video_name = Path(video_path).name.strip(Path(video_path).suffix)
    count = 0
    for path in Path(jpg_path + video_name).iterdir():
        if path.is_file():
            count += 1
    
    database = {}
    database[video_name]={}
    database[video_name]["subset"]="validation"
    database[video_name]["annotations"]={"label":"AddingReagent", "segment":[1,count+1]}
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)
    
    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)
        
    #subprocess.run("python ./3D-ResNets-PyTorch-master/main.py --root_path ./3D-ResNets-PyTorch-master/data --video_path ../../tempwork/videojpg --annotation_path ../../tempwork/arinp.json --result_path ../../tempwork/arresult --dataset ucf101 --resume_path ../../save_3218.pth --model_depth 34 --n_classes 3 --n_threads 4 --no_train --no_val --inference --output_topk 1 --inference_batch_size 1 --no_cuda", shell=True)
    #subprocess.run("python ./3D-ResNets-PyTorch-master/main.py --root_path ./3D-ResNets-PyTorch-master/data --video_path ../../tempwork/videojpg --annotation_path ../../tempwork/arinp.json --result_path ../../tempwork/arresult --dataset ucf101 --resume_path ../../weights/action/save_3218.pth  --model_depth 34 --n_classes 3 --n_threads 4 --no_train --no_val --inference --output_topk 3 --inference_batch_size 1 --no_cuda", shell=True)
    subprocess.run("python ./3D-ResNets-PyTorch-master/main.py --root_path ./3D-ResNets-PyTorch-master/data --video_path ../../tempwork/videojpg --annotation_path ../../tempwork/arinp.json --result_path ../../tempwork/arresult --dataset ucf101 --resume_path ../../weights/action/action_recog.pth  --model_depth 34 --n_classes 3 --n_threads 4 --no_train --no_val --inference --output_topk 3 --inference_batch_size 1 --no_cuda", shell=True)
    
    json_path = "./tempwork/arresult/val.json"
    with open(json_path) as f:
        di = json.load(f)
    
    return di, count

def track_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )
    is_display_tracker, tracker = "Yes", "botsort.yaml"
    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image,
                                                     is_display_tracker,
                                                     tracker
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")

def Barcode():
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    from PIL import Image, ImageDraw, ImageFont
    from pyzbar.pyzbar import decode, ZBarSymbol
    import urllib.request
    from bs4 import BeautifulSoup
    import requests
    
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            # adding the uploaded image to the page with caption
            uploaded_image = Image.open(source_img)
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                decoded_list = decode(uploaded_image)

                Barcode_num = len(decoded_list)
                Barcode_list = []
                for i in range(Barcode_num):
                    Ans = decoded_list[i].data
                    Barcode_list.append(str(Ans)[2:-1])
                print(Barcode_list)

                # Mark barcode area
                draw = ImageDraw.Draw(uploaded_image)
                font = ImageFont.truetype('./times.ttf', size=20)  # Set 'arial.ttf' for Windows

                for d in decode(uploaded_image):
                    draw.rectangle(((d.rect.left, d.rect.top), (d.rect.left + d.rect.width, d.rect.top + d.rect.height)),
                                   outline=(0, 0, 255), width=3)
                    draw.polygon(d.polygon, outline=(0, 255, 0), width=3)
                    draw.text((d.rect.left, d.rect.top + d.rect.height), d.data.decode(),
                              (255, 0, 0), font=font)
                uploaded_image.save('./tempwork/barcode_result.jpg')

                url = "https://labchem-wako.fujifilm.com/jp/product/result/product.html?ja="+Barcode_list[0] # Search WAKO Inc.
                r = requests.get(url)
                soup = BeautifulSoup(r.text)
                barcode_result = soup.em.q.text
                #print(soup.em.q.text)

                with col2:
                    st.image(image='./tempwork/barcode_result.jpg',
                             caption = "JAN code: " + Barcode_list[0] + "  " + barcode_result,
                             use_column_width=True)


# Task = flowchartの場合の動作
def Flowchart(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    if(os.path.isfile("./tempwork/result.txt")):
        os.remove("./tempwork/result.txt")
    csv_path = "./tempwork/test.csv"

    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )
    is_display_tracker, tracker = "Yes", "botsort.yaml"
    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()

                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                           Flowcharting1_gencsv(conf,
                                                model,
                                                st_frame,
                                                image,
                                                is_display_tracker,
                                                tracker,
                                                csv_path
                                                )
                        else:
                            vid_cap.release()
                            break
                    Flowcharting2_genchart(csv_path, source_video.name)
                    st.image(
                        image="./tempwork/flowchart.png",
                        caption="Flowchat",
                        use_column_width=True
                    )
                except Exception as e:
                    st.error(f"Error loading video: {e}")
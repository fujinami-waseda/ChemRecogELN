#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code was developed based on the following GUI.
https://github.com/monemati/YOLOv8-DeepSORT-Streamlit/tree/main
"""
from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, track_uploaded_video, Barcode, Flowchart

# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Track", "Barcode", "Flowchart"]
)

model_type = None
if task_type == "Track":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 30, 100, 50)) / 100
elif task_type == "Barcode":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 30, 100, 50)) / 100
elif task_type == "Flowchart":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
if task_type == "Track":
    st.sidebar.header("Video Config")
    source_selectbox = st.sidebar.selectbox(
        "Select Source",
        config.SOURCES_LIST_VIDEO
        )
    source_img = None
    if source_selectbox == config.SOURCES_LIST_VIDEO[0]:
        track_uploaded_video(confidence, model)
if task_type == "Barcode":
    st.sidebar.header("Image Config")
    source_selectbox = st.sidebar.selectbox(
        "Select Source",
        config.SOURCES_LIST_IMAGE
        )
    source_img = None
    if source_selectbox == config.SOURCES_LIST_IMAGE[0]:
        Barcode()
if task_type == "Flowchart":
    st.sidebar.header("Video Config")
    source_selectbox = st.sidebar.selectbox(
        "Select Source",
        config.SOURCES_LIST_VIDEO
        )
    source_img = None
    if source_selectbox == config.SOURCES_LIST_VIDEO[0]: # Video
        Flowchart(confidence, model)
# Self Driving Car Perception System

This project demonstrates a simplified ADAS (Advanced Driver Assistance System)
using Computer Vision and Deep Learning.

## Features

• Lane Detection  
• Vehicle Detection using YOLOv8  
• Object Tracking (SORT)  
• Risk Level Estimation (SAFE / CLOSE / DANGER)  
• Sudden Intrusion Detection  
• Speed Breaker Detection  

## Technologies Used

Python  
OpenCV  
YOLOv8  
NumPy  

## How it Works

Video Input → Lane Detection → Object Detection → Object Tracking → Risk Estimation → Intrusion Detection → Speed Breaker Detection

## Run the Project

pip install ultralytics opencv-python numpy

python src/autonomous_system.py

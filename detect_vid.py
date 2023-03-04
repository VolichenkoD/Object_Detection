#!/usr/bin/env python
# coding: utf-8

import torchvision
import cv2
import argparse
import torch
import time
import detect_utils
from PIL import Image

# сборка аргументов парсера
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# загрузка модели
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    min_size=args['min_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# получаем высоту и ширину кадра
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
# определяем кодек и создаем видео которое будет на выходе 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))





frame_count = 0 # подсчет кадров
total_fps = 0 # финальные кадры в секунду
# загружаем модель на вычислительное устройство
model = model.eval().to(device)





# чтение видео до конца
while(cap.isOpened()):
    # берем каждый кадр видео
    ret, frame = cap.read()
    if ret == True:
        # начальное время
        start_time = time.time()
        with torch.no_grad():
            # получаем прогнозы для текущего кадра
            boxes, classes, labels = detect_utils.predict(frame, model, device, 0.8)
        
        # рисуем поля и показываем текущий кадр на экране
        image = detect_utils.draw_boxes(boxes, classes, labels, frame)
        # конечное время
        end_time = time.time()
        # получаем fps
        fps = 1 / (end_time - start_time)
        # добавляем fps
        total_fps += fps
        # увеличеваем количество кадров
        frame_count += 1
        # нажми `q` для выхода
        wait_time = max(1, int(fps/4))
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break





# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")


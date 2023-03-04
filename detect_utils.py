#!/usr/bin/env python
# coding: utf-8

import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names


# разные цвета для каждого класса
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# преобразования изображения
transform = transforms.Compose([
    transforms.ToTensor(),
])




def predict(image, model, device, detection_threshold):
    # трансформируем из изображения в тензор
    image = transform(image).to(device)
    image = image.unsqueeze(0) # добавляем измерение
    outputs = model(image) # получаем предсказания по изображению
    # получаем все предсказанные имена классов
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # получаем скор за все предсказанные объекты
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # получаем все предсказанные bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # берем bounding boxes выше порогового балла
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']




def draw_boxes(boxes, classes, labels, image):
    # читаем изображение с помощью OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image


import os
import sys
import numpy as np
import torch
import cv2

from face_spinner.utils import Window, draw_face, rotate_bound
from face_spinner.models import PCN1, PCN2, PCN3
from face_spinner.facade import PCN


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Insufficient input. Please input filename after the command.")
        exit(1)

    img_path = sys.argv[1]

    pcn1_state_dict = torch.load("saved_models/model_0.sdict")
    pcn1 = PCN1()
    pcn1.load_state_dict(pcn1_state_dict)

    pcn2_state_dict = torch.load("saved_models/model_1.sdict")
    pcn2 = PCN2()
    pcn2.load_state_dict(pcn2_state_dict)

    pcn3_state_dict = torch.load("saved_models/model_2.sdict")
    pcn3 = PCN3()
    pcn3.load_state_dict(pcn3_state_dict)

    model = PCN(pcn1, pcn2, pcn3)

    img = cv2.imread(img_path)
    faces = model.detect(img)

    if len(faces) == 0:
        print("No face detected. Try again")
        exit(1)
    elif len(faces) > 1:
        print("Too many faces detected. Try again")
        exit(1)
    else:
        face = faces[0]
        rotated_img = rotate_bound(img, face.angle)
        print("Complete successfully. Writing to output.jpg")
        cv2.imwrite("output.jpg", rotated_img)

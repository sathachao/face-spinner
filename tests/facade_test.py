import unittest

import torch
import cv2
import pytest

from face_spinner.utils import rotate_bound
from face_spinner.models import PCN1, PCN2, PCN3
from face_spinner.facade import PCN


class TestPCN(unittest.TestCase):
    def setUp(self):
        pcn1_state_dict = torch.load("saved_models/model_0.sdict")
        pcn1 = PCN1()
        pcn1.load_state_dict(pcn1_state_dict)

        pcn2_state_dict = torch.load("saved_models/model_1.sdict")
        pcn2 = PCN2()
        pcn2.load_state_dict(pcn2_state_dict)

        pcn3_state_dict = torch.load("saved_models/model_2.sdict")
        pcn3 = PCN3()
        pcn3.load_state_dict(pcn3_state_dict)

        self._model = PCN(pcn1, pcn2, pcn3)

    def testNormalFace(self):
        img = cv2.imread("tests/img/face.jpg")
        faces = self._model.detect(img)
        assert len(faces) == 1, "Too many or few faces detected"
        face = faces[0]
        assert face.angle > -30 and face.angle < 30, "Face aligned incorrectly"

    def testRotated90Face(self):
        img = cv2.imread("tests/img/face-90.jpg")
        faces = self._model.detect(img)
        assert len(faces) == 1, "Too many or few faces detected"
        face = faces[0]
        upper_bound = 120
        lower_bound = 60

        actual_angle = face.angle if face.angle > 0 else face.angle + 360
        assert (
            actual_angle > lower_bound and actual_angle < upper_bound
        ), "Face aligned incorrectly"

    def testRotated180Face(self):
        img = cv2.imread("tests/img/face-180.jpg")
        faces = self._model.detect(img)
        assert len(faces) == 1, "Too many or few faces detected"
        face = faces[0]
        upper_bound = 210
        lower_bound = 150

        actual_angle = face.angle if face.angle > 0 else face.angle + 360
        assert (
            actual_angle > lower_bound and actual_angle < upper_bound
        ), "Face aligned incorrectly"

    def testRotated270Face(self):
        img = cv2.imread("tests/img/face-270.jpg")
        faces = self._model.detect(img)
        assert len(faces) == 1, "Too many or few faces detected"
        face = faces[0]
        upper_bound = 300
        lower_bound = 240

        actual_angle = face.angle if face.angle > 0 else face.angle + 360
        assert (
            actual_angle > lower_bound and actual_angle < upper_bound
        ), "Face aligned incorrectly"

    def testNoFaces(self):
        img = cv2.imread("tests/img/space-shuttle.jpg")
        faces = self._model.detect(img)
        assert len(faces) == 0, "Too many faces detected"

    def testManyFaces(self):
        img = cv2.imread("tests/img/many-faces.jpg")
        faces = self._model.detect(img)
        assert len(faces) > 1, "Too few faces detected"

import io
import os
import sys

import cv2
import numpy as np
import torch
from flask import Flask, request, make_response, send_file

from face_spinner.utils import rotate_bound
from face_spinner.models import PCN1, PCN2, PCN3
from face_spinner.facade import PCN


def setup_model():
    """
    Load PCN-x models and initialize a facade PCN

    Returns:
        face_spinner.facade.PCN -> PCN model
    """
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
    return model


def create_app():
    """
    Create a Flask application for face alignment

    Returns:
        flask.Flask -> Flask application
    """
    app = Flask(__name__)

    model = setup_model()
    app.config.from_mapping(MODEL=model)

    @app.route("/", methods=["GET"])
    def howto():
        instruction = (
            "Send POST request to /align to fix face orientation in input image"
            "\nex."
            "\n\tcurl -X POST -F 'image=@/path/to/face.jpg' --output output.jpg localhost:5000/align"
        )

        return instruction

    @app.route("/align", methods=["POST"])
    def align():
        data = request.files["image"]
        img_str = data.read()
        nparr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        faces = model.detect(img)
        if len(faces) == 0:
            return "No face found. Try again", 400
        elif len(faces) > 1:
            return "Too many faces found. Try again", 400
        else:
            face = faces[0]
            rotated_image = rotate_bound(img, face.angle)

            # Encode image
            is_completed, buf = cv2.imencode(".jpg", rotated_image)
            if not is_completed:
                return "Unexpected encoding error. Try again", 400
            byte_buffer = io.BytesIO(buf.tostring())

            return send_file(
                byte_buffer,
                "image/jpeg",
                as_attachment=True,
                attachment_filename="output.jpg",
            )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run()

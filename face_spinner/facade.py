import os
import numpy as np
import torch
import cv2
import torch

from face_spinner.utils import to_torch_tensor

EPS = 1e-5


def is_legal(x, y, img):
    """
    Check if (x, y) is a valid coordinate in img

    Args:
        x (int): x-coordinate
        y (int): y-coordinate
        img (numpy.array): Image
    
    Returns:
        bool -> True if valid, False otherwise
    """
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        return True
    else:
        return False


def is_inside(x, y, window):
    """
    Check if (x, y) is a valid coordinate in input window

    Args:
        x (int): x-coordinate
        y (int): y-coordinate
        window (face_spinner.Window): Window

    Returns:
        bool -> True if valid, False otherwise
    """
    if window.x <= x < (window.x + window.w) and window.y <= y < (window.y + window.h):
        return True
    else:
        return False


def calculate_IoU(window1, window2):
    """
    Calculates IoU (Intersection over Union) value

    Args:
        window1 (face_spinner.Window): Window 1
        window2 (face_spinner.Window): Window 2
    
    Returns:
        float -> 
    """
    x_overlap = max(
        0,
        min(window1.x + window1.w - 1, window2.x + window2.w - 1)
        - max(window1.x, window2.x)
        + 1,
    )
    y_overlap = max(
        0,
        min(window1.y + window1.h - 1, window2.y + window2.h - 1)
        - max(window1.y, window2.y)
        + 1,
    )
    intersection = x_overlap * y_overlap
    union = window1.w * window1.h + window2.w * window2.h - intersection
    return intersection / union


class Window:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf


class PCN:
    def __init__(
        self, pcn1, pcn2, pcn3, thres1=0.37, thres2=0.43, thres3=0.97, min_face_size=20
    ):
        """
        PCN is an implementation of Progressive Calibration Networks to perform face detection

        Args:
            pcn1 (face_spinner.models.PCN1): Model for stage 1
            pcn2 (face_spinner.models.PCN2): Model for stage 2
            pcn3 (face_spinner.models.PCN3): Model for stage 3
            thres1 (float): Confidence threshold for model pcn1
            thres2 (float): Confidence threshold for model pcn2
            thres3 (float): Confidence threshold for model pcn3
            min_face_size (float): Minimum face size to be detected
        """
        self._pcn1 = pcn1
        self._pcn2 = pcn2
        self._pcn3 = pcn3

        self._thres1 = thres1
        self._thres2 = thres2
        self._thres3 = thres3

        self._min_face_size = 1.4 * (min_face_size if min_face_size >= 20 else 20)

        self._nms_thres1 = 0.8
        self._nms_thres2 = 0.8
        self._nms_thres3 = 0.3

        self._angle_range = 45
        self._scale = 1.414
        self._stride = 8

        self._mean = (104, 117, 123)

    def _pad_image(self, img):
        """
        Pads input image

        Args:
            img (`numpy.array`): Input image to be padded
        
        Returns:
            An instance of `numpy.array` of a padded image
        """
        row = min(int(img.shape[0] * 0.2), 100)
        col = min(int(img.shape[1] * 0.2), 100)
        padded = cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)
        return padded

    def _resize_image(self, img, scale):
        """
        Scale input image

        Args:
            img (`numpy.array`): Input image to be scaled
            scale (float): Scale
        
        Returns:
            An instance of `numpy.array` of a scaled image
        """
        height, width = img.shape[:2]
        new_height, new_width = int(height / scale), int(width / scale)
        img = img.astype(np.float32)  # fix opencv type error
        scaled_img = cv2.resize(img, (new_width, new_height))
        return scaled_img

    def _preprocess_img(self, img, dim=None):
        """
        Preprocesses image

        Args:
            img (numpy.array): Input image
        
        Returns:
            An instance of face_spinner.Window
        """
        if dim:
            img = cv2.resize(img, (dim, dim))
        preprocessed_img = img - np.array(self._mean)
        return preprocessed_img

    def _suppress(self, windows, local, threshold):
        """
        Performs non-maximum suppression (NMS) on windows

        Args:
            windows (list of `face_spinner.Window`): Windows
            local (bool): Use local function
            threshold (float): Overlap threshold

        Returns:
            list of `face_spinner.Window` -> Suppressed windows
        """
        win_list = windows[:]
        length = len(windows)
        if length == 0:
            return win_list
        win_list.sort(key=lambda x: x.conf, reverse=True)
        flags = [0] * length
        for i in range(length):
            if flags[i]:
                continue
            for j in range(i + 1, length):
                if local and abs(win_list[i].scale - win_list[j].scale) > EPS:
                    continue
                if calculate_IoU(win_list[i], win_list[j]) > threshold:
                    flags[j] = 1
        suppressed_windows = [win_list[i] for i in range(length) if not flags[i]]
        return suppressed_windows

    def _delete_fp(self, windows):
        """
        Deletes false positives from windows

        Args:
            windows (list of `face_spinner.Window`): List of windows
        
        Returns:
            list of `face_spinner.Window` -> Windows with false positives deleted
        """
        window_list = windows[:]
        length = len(window_list)
        if length == 0:
            return window_list
        window_list.sort(key=lambda x: x.conf, reverse=True)
        flag = [0] * length
        for i in range(length):
            if flag[i]:
                continue
            for j in range(i + 1, length):
                win = window_list[j]
                if is_inside(win.x, win.y, window_list[i]) and is_inside(
                    win.x + win.w - 1, win.y + win.h - 1, window_list[i]
                ):
                    flag[j] = 1
        suppressed_windows = [window_list[i] for i in range(length) if not flag[i]]
        return suppressed_windows

    def _stage1(self, img, padded_img, net, thres):
        """
        Performs PCN-1

        Args:
            img (numpy.array): Input image
            padded_img (numpy.array): Padded image
            net (face_spinner.models.PCN1): Input network
            thres (float): Threshold

        Returns:
            list of `face_spinner.Window` -> Windows of detected faces
        """

        row = (padded_img.shape[0] - img.shape[0]) // 2
        col = (padded_img.shape[1] - img.shape[1]) // 2

        window_list = []
        netSize = 24
        curScale = self._min_face_size / netSize
        img_resized = self._resize_image(img, curScale)
        while min(img_resized.shape[:2]) >= netSize:
            img_resized = self._preprocess_img(img_resized)
            # net forward
            net_input = to_torch_tensor(img_resized)
            with torch.no_grad():
                net.eval()
                cls_prob, rotate, bbox = net(net_input)

            w = netSize * curScale
            for i in range(cls_prob.shape[2]):  # cls_prob[2]->height
                for j in range(cls_prob.shape[3]):  # cls_prob[3]->width
                    if cls_prob[0, 1, i, j].item() > thres:
                        sn = bbox[0, 0, i, j].item()
                        xn = bbox[0, 1, i, j].item()
                        yn = bbox[0, 2, i, j].item()
                        rx = (
                            int(
                                j * curScale * self._stride
                                - 0.5 * sn * w
                                + sn * xn * w
                                + 0.5 * w
                            )
                            + col
                        )
                        ry = (
                            int(
                                i * curScale * self._stride
                                - 0.5 * sn * w
                                + sn * yn * w
                                + 0.5 * w
                            )
                            + row
                        )
                        rw = int(w * sn)
                        if is_legal(rx, ry, padded_img) and is_legal(
                            rx + rw - 1, ry + rw - 1, padded_img
                        ):
                            if rotate[0, 1, i, j].item() > 0.5:
                                window_list.append(
                                    Window(
                                        rx,
                                        ry,
                                        rw,
                                        rw,
                                        0,
                                        curScale,
                                        cls_prob[0, 1, i, j].item(),
                                    )
                                )
                            else:
                                window_list.append(
                                    Window(
                                        rx,
                                        ry,
                                        rw,
                                        rw,
                                        180,
                                        curScale,
                                        cls_prob[0, 1, i, j].item(),
                                    )
                                )
            img_resized = self._resize_image(img_resized, self._scale)
            curScale = img.shape[0] / img_resized.shape[0]
        return window_list

    def _stage2(self, img, img180, net, thres, dim, windows):
        """
        Performs PCN-2

        Args:
            img (numpy.array): Input image
            img180 (numpy.array): Flipped input image
            net (face_spinner.models.PCN1): Input network
            thres (float): Threshold
            dim (int): Preprocess dimension size
            windows (list of `face_spinner.Window`): List of detected faces from PCN-1

        Returns:
            list of `face_spinner.Window` -> List of detected faces
        """
        window_list = windows[:]
        length = len(window_list)
        if length == 0:
            return window_list
        datalist = []
        height = img.shape[0]
        for win in window_list:
            if abs(win.angle) < EPS:
                datalist.append(
                    self._preprocess_img(
                        img[win.y : win.y + win.h, win.x : win.x + win.w, :], dim
                    )
                )
            else:
                y2 = win.y + win.h - 1
                y = height - 1 - y2
                datalist.append(
                    self._preprocess_img(
                        img180[y : y + win.h, win.x : win.x + win.w, :], dim
                    )
                )
        # net forward
        net_input = to_torch_tensor(datalist)
        with torch.no_grad():
            net.eval()
            cls_prob, rotate, bbox = net(net_input)

        new_windows = []
        for i in range(length):
            if cls_prob[i, 1].item() > thres:
                sn = bbox[i, 0].item()
                xn = bbox[i, 1].item()
                yn = bbox[i, 2].item()
                cropX = window_list[i].x
                cropY = window_list[i].y
                cropW = window_list[i].w
                if abs(window_list[i].angle) > EPS:
                    cropY = height - 1 - (cropY + cropW - 1)
                w = int(sn * cropW)
                x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
                y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
                maxRotateScore = 0
                maxRotateIndex = 0
                for j in range(3):
                    if rotate[i, j].item() > maxRotateScore:
                        maxRotateScore = rotate[i, j].item()
                        maxRotateIndex = j
                if is_legal(x, y, img) and is_legal(x + w - 1, y + w - 1, img):
                    angle = 0
                    if abs(window_list[i].angle) < EPS:
                        if maxRotateIndex == 0:
                            angle = 90
                        elif maxRotateIndex == 1:
                            angle = 0
                        else:
                            angle = -90
                        new_windows.append(
                            Window(
                                x,
                                y,
                                w,
                                w,
                                angle,
                                window_list[i].scale,
                                cls_prob[i, 1].item(),
                            )
                        )
                    else:
                        if maxRotateIndex == 0:
                            angle = 90
                        elif maxRotateIndex == 1:
                            angle = 180
                        else:
                            angle = -90
                        new_windows.append(
                            Window(
                                x,
                                height - 1 - (y + w - 1),
                                w,
                                w,
                                angle,
                                window_list[i].scale,
                                cls_prob[i, 1].item(),
                            )
                        )
        return new_windows

    def _stage3(self, padded_img, img180, img90, img_neg90, net, thres, dim, windows):
        """
        Performs PCN-3

        Args:
            padded_img (numpy.array): Padded image
            img180 (numpy.array): Flipped image
            img90 (numpy.array): 90-degree rotated image
            img_neg90 (numpy.array): -90-degree rotated image
            net (face_spinner.models.PCN1): Input network
            thres (float): Threshold
            dim (int): Preprocess dimension size
            windows (list of `face_spinner.Window`): List of detected faces from PCN-1

        Returns:
            list of `face_spinner.Window` -> List of detected faces
        """
        window_list = windows[:]
        length = len(window_list)
        if length == 0:
            return window_list

        datalist = []
        height, width = padded_img.shape[:2]

        for win in window_list:
            if abs(win.angle) < EPS:
                datalist.append(
                    self._preprocess_img(
                        padded_img[win.y : win.y + win.h, win.x : win.x + win.w, :], dim
                    )
                )
            elif abs(win.angle - 90) < EPS:
                datalist.append(
                    self._preprocess_img(
                        img90[win.x : win.x + win.w, win.y : win.y + win.h, :], dim
                    )
                )
            elif abs(win.angle + 90) < EPS:
                x = win.y
                y = width - 1 - (win.x + win.w - 1)
                datalist.append(
                    self._preprocess_img(
                        img_neg90[y : y + win.h, x : x + win.w, :], dim
                    )
                )
            else:
                y2 = win.y + win.h - 1
                y = height - 1 - y2
                datalist.append(
                    self._preprocess_img(
                        img180[y : y + win.h, win.x : win.x + win.w], dim
                    )
                )
        # network forward
        net_input = to_torch_tensor(datalist)
        with torch.no_grad():
            net.eval()
            cls_prob, rotate, bbox = net(net_input)

        new_windows = []
        for i in range(length):
            if cls_prob[i, 1].item() > thres:
                sn = bbox[i, 0].item()
                xn = bbox[i, 1].item()
                yn = bbox[i, 2].item()
                cropX = window_list[i].x
                cropY = window_list[i].y
                cropW = window_list[i].w
                img_tmp = padded_img
                if abs(window_list[i].angle - 180) < EPS:
                    cropY = height - 1 - (cropY + cropW - 1)
                    img_tmp = img180
                elif abs(window_list[i].angle - 90) < EPS:
                    cropX, cropY = cropY, cropX
                    img_tmp = img90
                elif abs(window_list[i].angle + 90) < EPS:
                    cropX = window_list[i].y
                    cropY = width - 1 - (window_list[i].x + window_list[i].w - 1)
                    img_tmp = img_neg90

                w = int(sn * cropW)
                x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
                y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
                angle = self._angle_range * rotate[i, 0].item()
                if is_legal(x, y, img_tmp) and is_legal(x + w - 1, y + w - 1, img_tmp):
                    if abs(window_list[i].angle) < EPS:
                        new_windows.append(
                            Window(
                                x,
                                y,
                                w,
                                w,
                                angle,
                                window_list[i].scale,
                                cls_prob[i, 1].item(),
                            )
                        )
                    elif abs(window_list[i].angle - 180) < EPS:
                        new_windows.append(
                            Window(
                                x,
                                height - 1 - (y + w - 1),
                                w,
                                w,
                                180 - angle,
                                window_list[i].scale,
                                cls_prob[i, 1].item(),
                            )
                        )
                    elif abs(window_list[i].angle - 90) < EPS:
                        new_windows.append(
                            Window(
                                y,
                                x,
                                w,
                                w,
                                90 - angle,
                                window_list[i].scale,
                                cls_prob[i, 1].item(),
                            )
                        )
                    else:
                        new_windows.append(
                            Window(
                                width - y - w,
                                x,
                                w,
                                w,
                                -90 + angle,
                                window_list[i].scale,
                                cls_prob[i, 1].item(),
                            )
                        )
        return new_windows

    def detect(self, img):
        """
        Performs face detection on input image

        Args:
            img (numpy.array): Input image
        
        Returns:
            An instance of face_spinner.Window
        """

        padded_img = self._pad_image(img)
        img180 = cv2.flip(padded_img, 0)
        img90 = cv2.transpose(padded_img)
        img_neg90 = cv2.flip(img90, 0)

        windows = self._stage1(img, padded_img, self._pcn1, self._thres1)
        windows = self._suppress(windows, True, self._nms_thres1)
        windows = self._stage2(
            padded_img, img180, self._pcn2, self._thres2, 24, windows
        )
        windows = self._suppress(windows, True, self._nms_thres2)
        windows = self._stage3(
            padded_img, img180, img90, img_neg90, self._pcn3, self._thres3, 48, windows
        )
        windows = self._suppress(windows, False, self._nms_thres3)
        windows = self._delete_fp(windows)

        return windows

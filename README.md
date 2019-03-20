# Face Spinner

## Description

Face Spinner is an application for mitigating face orientation problems in image processing.

## Setup

### Install dependencies

```bash
# If pipenv is installed
pipenv sync

# If virtualenv or pip is preferred
pip install -r requirements.txt
```

## Example

### Run example face aligner

```bash
python -m example.basic /path/to/your/image.jpg # Result saved to output.jpg
```

### Run Flask application

```bash
python application.py # Run on localhost:5000
# Check usage using `curl localhost:5000`
# Test making a request to align an image. Result will be outputted to output.jpg
curl -X POST -F "image=@tests/img/face-270.jpg" --output output.jpg localhost:5000/align
```

### Build & Run in Docker container

```bash
docker build -t face-spinner .
docker run -p 5000:5000 face-spinner # Run on localhost:5000
# Check usage using `curl localhost:5000`
# Test making a request to align an image. Result will be outputted to output.jpg
curl -X POST -F "image=@tests/img/face-270.jpg" --output output.jpg localhost:5000/align
```

### Citing & Thanks

1. [Pytorch implementation of PCN](https://github.com/siriusdemon/pytorch-PCN) by [siriusdemon](https://github.com/siriusdemon)
2. Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks

```tex
@inproceedings{shiCVPR18pcn,
    Author = {Xuepeng Shi and Shiguang Shan and Meina Kan and Shuzhe Wu and Xilin Chen},
    Title = {Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks},
    Booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Year = {2018}
}
```

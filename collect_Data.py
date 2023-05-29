#!/usr/bin/python3
# by kichu - May 2023
#
# Last update: 20230430, 20230503, 20230515
# last change - added ocr + slot occupancy. not tested.

import io
import math
import time
import requests
import json
from PIL import Image
from requests.auth import HTTPDigestAuth
import cv2
import numpy as np
import os
import psutil
THROTTLE = 12.8

cams = json.load(open('all.json'))
confidence = 0.85
print(cams)

variable_file = 'variable.json'


def create_variable_file():
    initial_values = {key: 1 for key in cams.keys()}
    with open(variable_file, 'w') as f:
        json.dump(initial_values, f)


def read_variable_file():
    if not os.path.isfile(variable_file):
        create_variable_file()
    with open(variable_file, 'r') as f:
        return json.load(f)


def update_variable_file(variable):
    with open(variable_file, 'w') as f:
        json.dump(variable, f)


def get_mjpeg_frame(cam):
    r = requests.get('http://' + cam['ip'] + '/cgi-bin/video.cgi?type=http&cameraID=1&mjpegplay=1',
                     auth=HTTPDigestAuth(cam['user'], cam['pass']), timeout=10, stream=True)
    buf = b''
    for b in r.iter_content(65536):
        buf += b
        print('FILL:', len(buf))
        x = buf.find(b'Content-Length:')
        if x < 0: continue
        y = buf.find(b'\r', x)
        if y < 0: continue
        w = int(buf[x + 15:y])
        buf = buf[y + 4:]
        break
    for b in r.iter_content(65536):
        buf += b
        if len(buf) < w: continue
        r.close()
        return buf[:w]


# Works well with images of different dimensions
def orb_sim(img1, img2):
    orb = cv2.ORB_create()
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


variable = read_variable_file()

while True:
    for c in cams:
        cam = cams[c]
        last_frame = cv2.imread(f"./live/{c} {variable[c] - 1}.jpg", 0)

        new_frame_bytes = get_mjpeg_frame(cam)
        new_frame_pil = Image.open(io.BytesIO(new_frame_bytes))
        new_frame = cv2.cvtColor(np.array(new_frame_pil), cv2.COLOR_RGB2BGR)

        orb_similarity = orb_sim(last_frame, new_frame)
        print(c, f"- {variable[c]}", "|| :", orb_similarity)

        if orb_similarity < confidence and orb_similarity != 0:
            cv2.imwrite(f"./live/{c} {variable[c]}.jpg", new_frame)
            print("SAVED ", c, f"- {variable[c]}")
            variable[c] += 1

        v = psutil.getloadavg()[0]
        t = min(2.0, 0.1 + (math.exp(v * THROTTLE) / 100.0))
        time.sleep(t)

    update_variable_file(variable)

import numpy as np
import cv2
import logging
import pickle

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd
from pyzbar.pyzbar import decode
import face_recognition
logger = logging.getLogger()
logger.setLevel('INFO')
shinka_dict = {b"http://www.shinkalion.com?sfefbeksjebs": "resources/E5Hayabusa.png",
               b"http://www.shinkalion.com?fkyfkyysyels": "resources/E7Kagayaki.png",
               b'http://www.shinkalion.com?bfyfkyksblts': "resources/E6Komachi.png",
               b'http://www.shinkalion.com?ifffvfksllys': "resources/DrYellow.png"}
shinka_str_dict = {b"http://www.shinkalion.com?sfefbeksjebs": "E5 Hayabusa",
               b"http://www.shinkalion.com?fkyfkyysyels": "E7 Kagayaki",
               b'http://www.shinkalion.com?bfyfkyksblts': "E6 Komachi",
               b'http://www.shinkalion.com?ifffvfksllys': "Dr. Yellow",
}
henshin_mode = "E7 Kagayaki"


def warp(src, dst, src_pts, dst_pts, transform_func, warp_func, **kwargs):
    src_pts_arr = np.array(src_pts, dtype=np.float32)
    dst_pts_arr = np.array(dst_pts, dtype=np.float32)
    src_rect = cv2.boundingRect(src_pts_arr)
    dst_rect = cv2.boundingRect(dst_pts_arr)
    src_crop = src[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
    dst_crop = dst[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]]
    src_pts_crop = src_pts_arr - src_rect[:2]
    dst_pts_crop = dst_pts_arr - dst_rect[:2]

    mat = transform_func(src_pts_crop.astype(np.float32), dst_pts_crop.astype(np.float32))
    warp_img = warp_func(src_crop, mat, tuple(dst_rect[2:]), **kwargs)

    mask = np.zeros_like(dst_crop, dtype=np.float32)
    cv2.fillConvexPoly(mask, dst_pts_crop.astype(np.int), (1.0, 1.0, 1.0), cv2.LINE_AA)

    dst_crop_merge = warp_img * mask + dst_crop * (1 - mask)
    dst[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = dst_crop_merge


def warp_triangle(src, dst, src_pts, dst_pts, **kwargs):
    warp(src, dst, src_pts, dst_pts,
         cv2.getAffineTransform, cv2.warpAffine, **kwargs)


def warp_rectangle(src, dst, src_pts, dst_pts, **kwargs):
    warp(src, dst, src_pts, dst_pts,
         cv2.getPerspectiveTransform, cv2.warpPerspective, **kwargs)

def landmarks_to_my_points(landmarks, scale=1.0):
    """
    convert landmarks to self-made point lists.
    See: landmarks are defined in https://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#face_landmarks
    :param landmarks: dict of landmarks
    :return:
    """
    pts = [
        landmarks["chin"][0],
        landmarks["chin"][4],
        landmarks["chin"][8],
        landmarks["chin"][12],
        landmarks["chin"][16],
        landmarks["left_eye"][0],
        landmarks["left_eye"][2],
        landmarks["left_eye"][3],
        landmarks["left_eye"][4],
        landmarks["right_eye"][0],
        landmarks["right_eye"][2],
        landmarks["right_eye"][3],
        landmarks["right_eye"][5],
        landmarks["top_lip"][0],
        landmarks["top_lip"][3],
        landmarks["top_lip"][7],
        landmarks["bottom_lip"][3],
    ]
    pts = [list(x) for x in pts]
    pts = np.array(pts, dtype=np.float32)
    return pts * scale
# QR Code Read
# https://qiita.com/jrfk/items/76c308ef163c02e85bcb
# QR Code Write
# https://note.nkmk.me/python-pillow-qrcode/
if __name__ == "__main__":
    shinka_image_dict = {}
    for key, image_file in shinka_dict.items():
        image = cv2.imread(image_file)
        shinka_image_dict[shinka_str_dict[key]] = image
    cap = cv2.VideoCapture(0)
    card_w = 360
    card_h = 360
    card_x = 900
    card_y = 200
    id = 0
    run_id = "test3"
    src_pts = [
        [154, 539],
        [245, 707],
        [455, 866],
        [678, 707],
        [761, 539],
        [251, 534],
        [325, 542],
        [398, 561],
        [327, 594],
        [525, 565],
        [583, 545],
        [666, 538],
        [591, 599],
        [351, 708],
        [458, 770],
        [567, 717],
        [458, 770],
    ]
    src_pts = np.array(src_pts, dtype=np.float32)
    mesh_list = [
        [0, 5, 1],
        [5, 8, 1],
        [8, 13, 1],
        [5, 6, 8],
        [6, 7, 8],
        [8, 7, 14],
        [8, 14, 13],
        [7, 9, 14],
        [9, 12, 14],
        [12, 15, 14],
        [9, 10, 12],
        [10, 11, 12],
        [12, 3, 15],
        [12, 11, 3],
        [11, 4, 3],
        [1, 13, 2],
        [13, 16, 2],
        [16, 15, 2],
        [15, 3, 2],
    ]

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Calculate Histogram
        card_image = frame[card_y: card_y + card_h, card_x: card_x + card_w]
        bgr_planes = cv2.split(card_image)
        histSize = 256
        histRange = (0, 256)  # the upper boundary is exclusive
        accumulate = False
        b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
        hist_w = 512
        hist_h = 400
        bin_w = int(round(hist_w / histSize))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        for i in range(1, histSize):
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(b_hist[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(b_hist[i]))),
                    (255, 0, 0), thickness=2)
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(g_hist[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(g_hist[i]))),
                    (0, 255, 0), thickness=2)
            cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(r_hist[i - 1]))),
                    (bin_w * (i), hist_h - int(np.round(r_hist[i]))),
                    (0, 0, 255), thickness=2)
        # cv2.imshow('Source image', src)
        cv2.imshow('calcHist Demo', histImage)
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        # cv2.flip(img, 1)
        disp_frame = frame.copy()

        cv2.rectangle(disp_frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), thickness=3)
        qr_data = pd.DataFrame(decode(frame))
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame_rgb = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(frame_rgb)
        print(henshin_mode)
        if not qr_data.empty:
            print(qr_data.data[0])
            if shinka_str_dict.get(qr_data.data[0]):
                henshin_mode = shinka_str_dict[qr_data.data[0]]

        # Draw Shinkarion Image
        for top, right, bottom, left in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            face_landmarks_list = face_recognition.face_landmarks(frame_rgb)
            if henshin_mode is not None:
                print(f"print {henshin_mode}")
                # resize_shinkarion = cv2.resize(shinka_image_dict[henshin_mode], (right - left, bottom - top))
                # disp_frame[top: bottom, left: right] = resize_shinkarion
                for landmarks in face_landmarks_list:
                    dst_pts = landmarks_to_my_points(landmarks, scale=4.0)

                    for mesh in mesh_list:
                        if len(mesh) == 3:
                            warp_triangle(shinka_image_dict[henshin_mode], disp_frame, src_pts[mesh], dst_pts[mesh])
                        elif len(mesh) == 4:
                            warp_rectangle(shinka_image_dict[henshin_mode], disp_frame, src_pts[mesh], dst_pts[mesh])
            else:
                cv2.rectangle(disp_frame, (left, top), (right, bottom), (255, 255, 255))
            # pil_image = Image.fromarray(frame_rgb)
            # for face_landmarks in face_landmarks_list:


                # cv2.imshow("makeup", frame_bgr)
        #     print(qr_data)
        cv2.imshow('frame', disp_frame)
        chr = cv2.waitKey(1) & 0xFF
        if chr == ord('q'):
            break
        elif chr == ord('s'):
            filename_prefix = "{}-{}".format(run_id, id)
            id += 1
            cv2.imwrite(f"{filename_prefix}.png", frame)
            with open(f"{filename_prefix}.pkl", "wb") as f:
                pickle.dump(face_landmarks_list[0], f)
            logger.info("write: {}".format(filename_prefix))


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

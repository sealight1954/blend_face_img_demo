import numpy as np
import cv2
import logging

import pandas as pd
from pyzbar.pyzbar import decode
import face_recognition
logger = logging.getLogger()
logger.setLevel('INFO')
shinka_dict = {b"http://www.shinkalion.com?sfefbeksjebs": "resources/E5Hayabusa.png",
               b"http://www.shinkalion.com?fkyfkyysyels": "resources/E7Kagayaki,png"}
shinka_str_dict = {b"http://www.shinkalion.com?sfefbeksjebs": "E5 Hayabusa",
               b"http://www.shinkalion.com?fkyfkyysyels": "E7 Kagayaki"}
henshin_mode = None
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
    run_id = "test"
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
            if shinka_str_dict.get(qr_data.data[0]):
                henshin_mode = shinka_str_dict[qr_data.data[0]]

        # Draw Shinkarion Image
        for top, right, bottom, left in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            if henshin_mode is not None:
                print(f"print {henshin_mode}")
                resize_shinkarion = cv2.resize(shinka_image_dict[henshin_mode], (right - left, bottom - top))
                disp_frame[top: bottom, left: right] = resize_shinkarion
            else:
                cv2.rectangle(disp_frame, (left, top), (right, bottom), (255, 255, 255))

        #     print(qr_data)
        cv2.imshow('frame', disp_frame)
        chr = cv2.waitKey(1) & 0xFF
        if chr == ord('q'):
            break
        elif chr == ord('s'):
            filename = "{}-{}.png".format(run_id, id)
            id += 1
            cv2.imwrite(filename, card_image)
            logger.info("write: {}".format(filename))

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

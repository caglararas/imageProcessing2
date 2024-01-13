import cv2
import numpy as np
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

glasses_img = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
hat_img = cv2.imread('hat.png', cv2.IMREAD_COLOR)

def load_image():
    image_path = input("Lütfen resim dosyasının yolunu girin: ")
    image = cv2.imread(image_path)
    return image

def change_hair_color(image, color):
    if color == "kırmızı":
        new_color = (0, 0, 255)
    elif color == "mor":
        new_color = (255, 0, 255)
    elif color == "yeşil":
        new_color = (0, 255, 0)
    elif color == "sarı":
        new_color = (0, 255, 255)
    else:
        new_color = (0, 0, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if color != "kırmızı" and color != "mor":

            hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            hair_mask[y:y + h, x:x + w] = 255


            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=hair_mask)
            hsv_image[..., 0] = new_color[0]
            modified_hair = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            image = cv2.add(image, modified_hair)

    return image

def apply_filters(image, apply_hat_glasses=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if apply_hat_glasses:

            glasses_x = x
            glasses_y = y + int(h / 4)
            glasses_w = int(w / 2) + int(w / 2)
            glasses_h = int(h / 4) * 2

            hat_x = x
            hat_y = y - int(h / 2)
            hat_w = w
            hat_h = int(h / 2)


            resized_glasses = cv2.resize(glasses_img, (glasses_w, glasses_h))
            resized_hat = cv2.resize(hat_img, (hat_w, hat_h))


            for i in range(resized_glasses.shape[0]):
                for j in range(resized_glasses.shape[1]):
                    if resized_glasses[i, j, 3] != 0 and glasses_y + i < image.shape[0] and glasses_x + j < image.shape[1]:
                        image[glasses_y + i, glasses_x + j, :] = resized_glasses[i, j, :3]

            for i in range(resized_hat.shape[0]):
                for j in range(resized_hat.shape[1]):
                    if hat_y + i < image.shape[0] and hat_x + j < image.shape[1]:
                        image[hat_y + i, hat_x + j, :] = resized_hat[i, j, :3]
        else:
            image = change_hair_color(image, "kırmızı" if apply_hat_glasses else "mor")

    return image

for i in range(6):
    image = load_image()
    if i < 3:
        filtered_image = apply_filters(image, apply_hat_glasses=False)
    else:
        filtered_image = apply_filters(image, apply_hat_glasses=True)
        filtered_image = change_hair_color(filtered_image, "kırmızı" if i < 5 else "mor")

    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.show()

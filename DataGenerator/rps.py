import os
import albumentations as A
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# path


def printImage(x, y, w, h, image):
    dw = image.shape[1]
    dh = image.shape[0]
    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 1)

    plt.imshow(image)
    plt.show()


def generate(additional_path):
    image_path = os.path.join(additional_path, "images")
    label_path = os.path.join(additional_path, "labels")
    image_list = os.listdir(image_path)

    for img in image_list:
        image = Image.open(image_path + "/" + img)
        label = open(label_path + "/" + img[:-4] + ".txt", "r").read().split()
        print(img)
        bboxes = []
        class_labels = []
        for i in range(0, len(label), 5):
            type = label[i]
            alist = label[(i+1):5*(i+1)]
            alist = np.array(alist).astype(np.float64)
            alist = list(alist)
            alist.append(type)
            print(alist)
            bboxes.append(alist)
            class_labels.append(type)
    #     print(img, bg, bboxes)
    #     printImage(xcenter, ycenter, w, h, array_image)

        array_image = np.array(image)
        array_image = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)

        transform = A.Compose([
            A.ShiftScaleRotate(p=1),
            A.Flip(p=0.5)
        ], bbox_params=A.BboxParams(format='yolo',  min_visibility=0.3, label_fields=['class_labels']))

        for index in [9, 14, 25, 253, 97, 111, 46, 7, 944]:
            random.seed(index)
            transformed = transform(
                image=array_image, bboxes=bboxes, class_labels=class_labels)
            # save image
            name = img+"_"+str(index)
            filename = os.path.join(
                image_path, name + ".jpg")

            new_image = cv2.cvtColor(
                transformed["image"], cv2.COLOR_BGR2RGB)
            print(filename)
            Image.fromarray(new_image).save(filename)
            newBoxes = transformed['bboxes']
            newLabels = ''
            for i in range(len(newBoxes)):
                newBox = newBoxes[i]
                newLabels = newLabels + str(newBox[4]) + ' ' + str(newBox[0]) + ' ' + str(
                    newBox[1]) + ' ' + str(newBox[2]) + ' ' + str(newBox[3]) + '\n'
#       printImage((newBoxes[0]), (newBoxes[1]), (newBoxes[2]), (newBoxes[3]), transformed['image'])

            filename = os.path.join(
                label_path, name+".txt")

            label_file = open(filename, "w")
            label_file.write(newLabels)


generate("train")
generate("valid")

# additional_path = "paper"
# classification = "0"
# outputPath = "./"

# image_list = os.listdir(os.path.join(datasetsPath, "image", additional_path))

# for img in image_list:
#     hand = Image.open(os.path.join(
#         datasetsPath, "image", additional_path, img))
#     width, height = hand.size
#     scale_down = 8
#     width = round(width/scale_down)
#     height = round(height/scale_down)
#     hand.thumbnail((width, height), Image.ANTIALIAS)

#     for bg in bg_list:
#         background = Image.open(backgroundPath+"/"+bg)
#         center_width = round((background.width - hand.width)/2)
#         center_height = round((background.height - hand.height)/2)
#         background.paste(hand, (center_width, center_height), hand)
#         background.save(os.path.join(outputPath, "result",
#                         "images", img+"_"+bg), format="png")

#         center_hand_imageX, center_hand_imageY = [
#             center_width + hand.width/2, center_height + hand.height/2]
#         xcenter = (center_hand_imageX) / background.width
#         ycenter = (center_hand_imageY) / background.height
#         w = numpy.clip(hand.width / background.width, 0, 0.99)
#         h = numpy.clip(hand.height / background.height, 0, 0.99)

#         # Finish ordinary image
#         label_file = open(os.path.join(outputPath, "result",
#                           "labels", img+"_"+bg)+".txt", "w")
#         label_file.write(classification + " " + str(xcenter) +
#                          " " + str(ycenter) + " " + str(w) + " " + str(h))
#         label_file.close()

#         array_image = numpy.array(background)
#         array_image = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)
#         bboxes = [[xcenter, ycenter, w, h]]
# #     print(img, bg, bboxes)
# #     printImage(xcenter, ycenter, w, h, array_image)

#         size = [background.width, background.height]
#         transform = A.Compose([
#             A.ShiftScaleRotate(p=1, rotate_limit=[0, 360]),
#         ], bbox_params=A.BboxParams(format='yolo',  min_visibility=0.3, label_fields=['class_labels']))

#         for index in [9, 14, 253, 514]:
#             random.seed(index)
#             transformed = transform(
#                 image=array_image, bboxes=bboxes, class_labels=["paper"])
#             # save image
#             index = str(index)
#             name = img+"_"+bg+"_"+index
#             filename = os.path.join(
#                 outputPath, "result", "images", name + ".png")
#             new_image = cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB)
#             Image.fromarray(new_image).save(filename)
#             newBoxes = transformed['bboxes'][0]
# #       printImage((newBoxes[0]), (newBoxes[1]), (newBoxes[2]), (newBoxes[3]), transformed['image'])
#             filename = os.path.join(
#                 outputPath, "result", "labels", name)+".txt"
#             label_file = open(filename, "w")
#             label_file.write(classification + " " + str(newBoxes[0]) + " " + str(
#                 newBoxes[1]) + " " + str(newBoxes[2]) + " " + str(newBoxes[3]))


# additional_path = "scissors"
# classification = "2"
# outputPath = "./"

# image_list = os.listdir(os.path.join(datasetsPath, "image", additional_path))

# for img in image_list:
#     hand = Image.open(os.path.join(
#         datasetsPath, "image", additional_path, img))
#     width, height = hand.size
#     scale_down = 8
#     width = round(width/scale_down)
#     height = round(height/scale_down)
#     hand.thumbnail((width, height), Image.ANTIALIAS)

#     for bg in bg_list:
#         background = Image.open(backgroundPath+"/"+bg)
#         center_width = round((background.width - hand.width)/2)
#         center_height = round((background.height - hand.height)/2)
#         background.paste(hand, (center_width, center_height), hand)
#         background.save(os.path.join(outputPath, "result",
#                         "images", img+"_"+bg), format="png")

#         center_hand_imageX, center_hand_imageY = [
#             center_width + hand.width/2, center_height + hand.height/2]
#         xcenter = (center_hand_imageX) / background.width
#         ycenter = (center_hand_imageY) / background.height
#         w = numpy.clip(hand.width / background.width, 0, 0.99)
#         h = numpy.clip(hand.height / background.height, 0, 0.99)

#         # Finish ordinary image
#         label_file = open(os.path.join(outputPath, "result",
#                           "labels", img+"_"+bg)+".txt", "w")
#         label_file.write(classification + " " + str(xcenter) +
#                          " " + str(ycenter) + " " + str(w) + " " + str(h))
#         label_file.close()

#         array_image = numpy.array(background)
#         array_image = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)
#         bboxes = [[xcenter, ycenter, w, h]]
# #     print(img, bg, bboxes)
# #     printImage(xcenter, ycenter, w, h, array_image)

#         size = [background.width, background.height]
#         transform = A.Compose([
#             A.ShiftScaleRotate(p=1, rotate_limit=[0, 360]),
#         ], bbox_params=A.BboxParams(format='yolo',  min_visibility=0.3, label_fields=['class_labels']))

#         for index in [9, 14, 25, 253]:
#             random.seed(index)
#             transformed = transform(
#                 image=array_image, bboxes=bboxes, class_labels=["scissors"])
#             # save image
#             index = str(index)
#             name = img+"_"+bg+"_"+index
#             filename = os.path.join(
#                 outputPath, "result", "images", name + ".png")
#             new_image = cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB)
#             Image.fromarray(new_image).save(filename)
#             newBoxes = transformed['bboxes'][0]
# #       printImage((newBoxes[0]), (newBoxes[1]), (newBoxes[2]), (newBoxes[3]), transformed['image'])
#             filename = os.path.join(
#                 outputPath, "result", "labels", name)+".txt"
#             label_file = open(filename, "w")
#             label_file.write(classification + " " + str(newBoxes[0]) + " " + str(
#                 newBoxes[1]) + " " + str(newBoxes[2]) + " " + str(newBoxes[3]))

import os
import cv2
import numpy as np
import easyocr
# Computer vision engineer : https://youtu.be/73REqZM1Fy0

class LiscensePlateNumberDetection:
    def __init__(self, cfg_path, weights_path):

        self.model_cfg_path = cfg_path
        self.model_weights = weights_path
        self.reader = easyocr.Reader(['en'])
        self.net = cv2.dnn.readNetFromDarknet(
            self.model_cfg_path, self.model_weights)


    def directoryInput(self,input_dir):
        result=[]
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            obj=self.detect(img, img_name)
            result.append(obj)

        return result
            


    def get_output(self, blob):
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1]
                         for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(output_layers)
        outs = [c for out in outs for c in out if c[4] > 0.1]
        return outs

    def NMS(self, boxes, class_ids, confidences, overlapThresh=0.5):
        boxes = np.asarray(boxes)
        class_ids = np.asarray(class_ids)
        confidences = np.asarray(confidences)
        if len(boxes) == 0:
            return [], [], []
        x1 = boxes[:, 0] - (boxes[:, 2] / 2)
        y1 = boxes[:, 1] - (boxes[:, 3] / 2)
        x2 = boxes[:, 0] + (boxes[:, 2] / 2)
        y2 = boxes[:, 1] + (boxes[:, 3] / 2)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.arange(len(x1))
        for i, box in enumerate(boxes):
            temp_indices = indices[indices != i]
            xx1 = np.maximum(
                box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
            yy1 = np.maximum(
                box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
            xx2 = np.minimum(
                box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
            yy2 = np.minimum(
                box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / areas[temp_indices]
            if np.any(overlap) > overlapThresh:
                indices = indices[indices != i]
        return boxes[indices], class_ids[indices], confidences[indices]

    

    def detect(self, img, image_name):
        bboxes = []
        class_ids = []
        scores = []


        H, W, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
        detections = self.get_output(blob)

        for detection in detections:
            bbox = detection[:4]

            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
            bboxes.append(bbox)

            class_id = np.argmax(detection[5:])
            class_ids.append(class_id)

            score = np.amax(detection[5:])
            scores.append(score)

        bboxes, class_ids, scores = self.NMS(bboxes, class_ids, scores)

        outputs=[]
        for bbox_, bbox in enumerate(bboxes):
            xc, yc, w, h = bbox
            license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
            
            license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
           
            _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

            output = self.reader.readtext(license_plate_thresh)
            for out in output:
                outputs.append(out[1])

        return {image_name:outputs}
            
                

if __name__=="__main__":
    LD=LiscensePlateNumberDetection(r'yolov3-from-opencv-object-detection\model\cfg\darknet-yolov3.cfg',r'yolov3-from-opencv-object-detection\model\model.weights')
    print(LD.directoryInput(r'data'))

    image_name = 'test.jpg'
    image_path = r'photo\download.png'

    image_object=cv2.imread(image_path)
    print(LD.detect(image_name=image_name,img=image_object))
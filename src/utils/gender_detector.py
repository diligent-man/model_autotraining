import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from cvzone import cornerRect
from collections import Counter
from vgg import *
import imageio


class GenderClassifier:
    def __init__(self, cfg, model_checkpoint_path, yolo_path):
        self.face_detector = YOLO(yolo_path)
        self.tracking = sv.ByteTrack()
        self.model = VGG(make_layers(cfg), DROPOUT=0.5, NUM_CLASSES=1, init_weights=False)
        self.model = self.model.to('cuda')
        checkpoint = torch.load(model_checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.mapping = {0: "Female", 1: "Male"}
        self.dict = {}
        self.frames_processed = 0

    def crop_face(self, frame):
        face_list = []
        id_list = []
        face_coordinates = []

        top_y_padding = 15
        bottom_y_padding = 10
        top_x_padding = 10
        bottom_x_padding = 10

        results = self.face_detector.predict(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracking.update_with_detections(detections)
        for detection in detections:
            coords = detection[0].astype(int)
            x1, y1, x2, y2 = coords
            h, w = y2 - y1, x2 - x1
            tracker_id = int(detection[-2])
            if h > 0 and w > 0:
                face = frame[y1 - top_y_padding:y1 + h + bottom_y_padding, x1 - top_x_padding:x1 + w + bottom_x_padding]
                if face.size != 0:
                    image = cv2.resize(face, (224, 224))
                    face_list.append(image)
                    id_list.append(tracker_id)
                    face_coordinates.append((x1, y1, w, h))
                    output = "D:\\abc"
                    cv2.imwrite(f"{output}/image_{self.frames_processed}.jpg", image)
                    self.frames_processed += 1

        return face_list, id_list, face_coordinates

    def predict_frame(self, frame):
        faces, ids, face_coords = self.crop_face(frame)
        labels = []
        index_of_ids = []
        for id in ids:
            if id in self.dict:
                if len(self.dict[id]) == 30:
                    index_of_id = ids.index(id)
                    x, y, w, h = face_coords[index_of_id]
                    most_common_element = Counter(self.dict[id]).most_common(1)[0][0]
                    cornerRect(frame, [x, y, w, h], l=5, rt=1)
                    cv2.putText(frame, most_common_element, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    index_of_ids.append(index_of_id)
        index_of_ids.sort(reverse=True)
        for index in index_of_ids:
            faces.pop(index)
            ids.pop(index)
            face_coords.pop(index)

        if faces:
            faces = torch.from_numpy(np.array(faces, dtype=np.float32)).view(len(faces), 3, 224, 224)
            faces = faces.to('cuda')
            predictions = torch.nn.functional.sigmoid(self.model(faces)).to('cuda')
            prediction = torch.where(predictions < .1, 0, 1)
            predictions = predictions.squeeze().detach().cpu().numpy().tolist()
            if isinstance(predictions, list):
                for i, prediction in enumerate(predictions):
                    class_label = self.mapping.get(int(prediction), "Unknown")
                    labels.append(class_label)
            else:
                class_label = self.mapping.get(int(predictions), "Unknown")
                labels.append(class_label)

            for id, label in zip(ids, labels):
                if id not in self.dict:
                    self.dict[id] = [label]
                self.dict[id].append(label)

            for i, (x, y, w, h) in enumerate(face_coords):
                cornerRect(frame, [x, y, w, h], l=5, rt=1)

        cv2.imshow('Real time processing', frame)
        return frame

    def process_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result_frame = self.predict_frame(frame)
            out.write(result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

import cv2
import numpy as np
import random

num_iteration=30
PADDING = 20

def detect_and_save_faces(video_url, cascade_path, output_folder, frames_per_second):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(video_url)

    frame_count = 0
    frames_processed = 0
    global num_iteration
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate / frames_per_second)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nếu là frame được chọn để xử lý, tiến hành nhận diện khuôn mặt và lưu lại
        if frame_count % frame_interval == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, maxSize=(50,50))

            for (x, y, w, h) in faces:
                face_roi = frame[y-20:y+h+10, x-5:x+w+5]
                cv2.imwrite(f"{output_folder}/face_{frames_processed}_{num_iteration}.jpg", face_roi)
                frames_processed += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

video_url = ['https://v-vm.vnecdn.net/720p/hanoi2023/38e78b86c272b23760fa7cf09b70e8e5.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/2ed7e9784518bb2a8e81d8e58564bcb1.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/90ba5200b9a8a61e74452d4a15ae9f7d.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/f0784cfde6f35de436e20a701a2f2499.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/e0be2e0b7938e432f830677cd80c1d57.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/31ea2766a0da508ff4b28ef1b0dea927.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/c8ae0ca75239271e60dcc9da3cd97b48.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/315e13a06dd97b7d59023abbdc34dec7.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/b8c3ba38e8d131705d057e0e70889831.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/484af4f9e6fd863ba228832b35d265cf.mp4',
'https://v-vm.vnecdn.net/720p/hanoi2023/efd87e1e23ea23c282261e4c595c9661.mp4',
"https://v-vm.vnecdn.net/720p/hanoi2023/e8fe5baf5e9ddf7c760ff9ad6153379a.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/c4e86b68d158ae16ab8e2c50b354bef4.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/7ad79865f3e443dab5aa3900b6832ad3.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/6f7f81dffd05ff5d1fbe0977e802ad83.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/2e442857ddbacd2e20181e61865bf1a7.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/4b42197de305e35ebf6d37aedaba2bd0.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/4fcce30df41ca522f2c39b7fad174d5e.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/bbbddae2cf1339bd052215a0a6828217.mp4",
"https://v-vm.vnecdn.net/720p/hanoi2023/315e13a06dd97b7d59023abbdc34dec7.mp4"
]

cascade_path = "C:\\Users\\phuon\\Downloads\\haarcascade_frontalface_default (1).xml"
output_folder = "C:\\Users\\phuon\\Documents\\VN_EXPRESS"

for i in video_url:
    detect_and_save_faces(i, cascade_path, output_folder, frames_per_second=5)
    num_iteration+=1

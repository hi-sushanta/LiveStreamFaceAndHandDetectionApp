import cv2
import numpy as np
import streamlit as st
import av

from streamlit_webrtc import webrtc_streamer

st.title("Human Face Detection in Live Stream")
# Model parameters used to train model.
mean = [104, 117, 123]
scale = 1.0
in_width = 300
in_height = 300
#
# # Set the detection threshold for face detections.
detection_threshold = 0.5

# Annotation settings.
font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

one_time_run = 0
net = cv2.dnn.readNetFromCaffe('./model/deploy.prototxt',
                               './model/res10_300x300_ssd_iter_140000.caffemodel')

def callback(img):
    frame = img.to_ndarray(format="bgr24")
    h = frame.shape[0]
    w = frame.shape[1]
    # Flip THE video frame horizontally (not required, just for convenience).
    frame = cv2.flip(frame, 1)

    # Convert the image into a blob format.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False,
                                 crop=False)
    # Pass the blob to the DNN model.
    net.setInput(blob)
    # Retrieve detections from the DNN model.
    detections = net.forward()

    # Process each detection.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            # Annotate the video frame with the detection results.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = 'Human Face, Confidence: %.4f' % confidence
            label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255),
                          cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=callback,media_stream_constraints={
            "video": True,
            "audio": False
        })





# run= st.checkbox("Run")
# s = 0 # Use web camera.
# video_cap = cv2.VideoCapture(s)
# win_name = 'Live Detect Faces'

# while run:
#     one_time_run += 1
#     if 0 < one_time_run < 2:
#         st.subheader(win_name)
#     has_frame, frame = webrtc_streamer("Hello",video_frame_callback)
#     one_time_run += 1
#     if not has_frame:
#         break
#     h = frame.shape[0]
#     w = frame.shape[1]
#     # Flip THE video frame horizontally (not required, just for convenience).
#     frame = cv2.flip(frame, 1)
#
#     # Convert the image into a blob format.
#     blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False,
#                                  crop=False)
#     # Pass the blob to the DNN model.
#     net.setInput(blob)
#     # Retrieve detections from the DNN model.
#     detections = net.forward()
#
#     # Process each detection.
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > detection_threshold:
#             # Extract the bounding box coordinates from the detection.
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x1, y1, x2, y2) = box.astype('int')
#
#             # Annotate the video frame with the detection results.
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = 'Human Face, Confidence: %.4f' % confidence
#             label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
#             cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255),
#                           cv2.FILLED)
#             cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))
#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
#     key = cv2.waitKey(1)
#     if key == ord('Q') or key == ord('q') or key == 27:
#         break
#
#
# if not run:
#     one_time_run = 0
#
# video_cap.release()
#
#

import cv2
import numpy as np

# Specify the paths to the YOLOv4 weights, config file, and COCO names file
weightsPath = r'E:\internship project\vehicle counting and classification for traffic surveillance\yolov4.weights'
configPath = r'E:\internship project\vehicle counting and classification for traffic surveillance\yolov4.cfg'
namesPath = r'E:\internship project\vehicle counting and classification for traffic surveillance\coco.names'

# Load the class labels
with open(namesPath, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Function to get the output layer names in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    # Correction for OpenCV >= 4.x
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the YOLO network
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    # Processing detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if classes[class_id] in ['car', 'bus', 'truck', 'motorbike']:
                    # Draw bounding box for vehicles
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{classes[class_id]}: {int(confidence * 100)}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Vehicle Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
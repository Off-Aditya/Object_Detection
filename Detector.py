import cv2
import numpy as np
import time
import threading
from sound import speak

np.random.seed(20)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classPath = classPath
        
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        
        self.readClasses()

        self.last_speech_time = time.time()
        self.speech_interval = 6  # Seconds

        self.speech_lock = threading.Lock()

    def readClasses(self):
        with open(self.classPath, 'r') as f:
            self.classesList = f.read().splitlines()
            
        self.classesList.insert(0, '__Background__')
        
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        
    def ContrastTextColor(self, boxColor):
        luminance = (0.299 * boxColor[2] + 0.587 * boxColor[1] + 0.114 * boxColor[0])
        if luminance > 127:
            return (0, 0, 0) 
        else:
            return (255, 255, 255)
        
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        
        if not cap.isOpened():
            print("Error opening file...")
            return
        
        (success, image) = cap.read()
        
        startTime = 0
        paused = False
        
        while success:
            if not paused:
                currentTime = time.time()
                fps = 1 / (currentTime - startTime)
                startTime = currentTime
                classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)
                
                bboxs = list(bboxs)
                confidences = list(np.array(confidences).reshape(1, -1)[0])
                confidences = list(map(float, confidences))
                
                bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
                
                object_count = {}
                
                if len(bboxIdx) != 0:
                    for i in range(0, len(bboxIdx)):
                        bbox = bboxs[np.squeeze(bboxIdx[i])]
                        classConfidence = confidences[np.squeeze(bboxIdx[i])]
                        classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                        classLabel = self.classesList[classLabelID]
                        classColor = [int(c) for c in self.colorList[classLabelID]]
                        textColor = self.ContrastTextColor(classColor)
                        
                        displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                        x, y, w, h = bbox
                        
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                        cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, textColor, 2)
                        
                        lineWidth = min(int(w * 0.3), int(h * 0.3))
                        
                        cv2.line(image, (x, y), (x + lineWidth, y), classColor, thickness=5)
                        cv2.line(image, (x, y), (x, y + lineWidth), classColor, thickness=5)
                        
                        cv2.line(image, (x + w, y), (x + w - lineWidth, y), classColor, thickness=5)
                        cv2.line(image, (x + w, y), (x + w, y + lineWidth), classColor, thickness=5)
                        
                        cv2.line(image, (x, y + h), (x + lineWidth, y + h), classColor, thickness=5)
                        cv2.line(image, (x, y + h), (x, y + h - lineWidth), classColor, thickness=5)
                        
                        cv2.line(image, (x + w, y + h), (x + w - lineWidth, y + h), classColor, thickness=5)
                        cv2.line(image, (x + w, y + h), (x + w, y + h - lineWidth), classColor, thickness=5)
                        
                        # Count objects
                        if classLabel not in object_count:
                            object_count[classLabel] = 1
                        else:
                            object_count[classLabel] += 1

                summary = "In the frame, I see "
                for object_label, count in object_count.items():
                    summary += f"{count} {object_label}, "

                summary = summary.rstrip(", ")

                current_time = time.time()
                with self.speech_lock:
                    if current_time - self.last_speech_time >= self.speech_interval:
                        speak(summary)
                        self.last_speech_time = current_time 
                
                cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.imshow("Result", image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                paused = not paused
            if not paused:
                (success, image) = cap.read()
        
        cv2.destroyAllWindows()

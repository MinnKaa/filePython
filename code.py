from ultralytics import YOLO
import cv2
from urllib.request import urlopen
import numpy as np
import requests
import time

# Khởi tạo mô hình YOLO (đảm bảo file "yolo11n.pt" có sẵn)
model = YOLO("yolo11n.pt")

# Địa chỉ URL của ESP32-CAM (chụp ảnh) và ESP8266 (điều khiển LED)
capture_url = "http://172.20.10.11/capture"  # ESP32-CAM
esp8266_led_url = "http://172.20.10.12/led"  # ESP8266

# Khởi tạo thời gian phát hiện cuối cùng và trạng thái LED
last_detection_time = 0
led_state = False

while True:
    try:
        # Lấy hình ảnh từ ESP32-CAM
        response = urlopen(capture_url)
        img_array = np.asarray(bytearray(response.read()), dtype="uint8")
        frame = cv2.imdecode(img_array, -1)
        
        # Phân tích hình ảnh với YOLO
        results = model(frame)
        person_detected = False

        for result in results:
            for box in result.boxes:
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0])
                
                # Nếu phát hiện người với độ tin cậy > 0.5
                if class_id == 0 and confidence > 0.5:
                    person_detected = True
                    last_detection_time = time.time()
                    # Vẽ khung và ghi chú
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Nếu phát hiện người và LED chưa bật, gửi lệnh bật LED đến ESP8266
        if person_detected and not led_state:
            try:
                requests.get(f"{esp8266_led_url}?state=on", timeout=1)
                led_state = True
                print("LED ON - Gửi tín hiệu đến ESP8266")
            except requests.RequestException:
                print("Không thể kết nối với ESP8266 để bật đèn")
        
        # Nếu không phát hiện người trong 5 giây và LED đang bật, gửi lệnh tắt LED đến ESP8266
        if time.time() - last_detection_time > 5 and led_state:
            try:
                requests.get(f"{esp8266_led_url}?state=off", timeout=1)
                led_state = False
                print("LED OFF - Gửi tín hiệu đến ESP8266")
            except requests.RequestException:
                print("Không thể kết nối với ESP8266 để tắt đèn")

        # Hiển thị hình ảnh với bounding box
        cv2.imshow("YOLO Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Lỗi:", e)
        time.sleep(1)  # Tạm dừng khi gặp lỗi

cv2.destroyAllWindows()

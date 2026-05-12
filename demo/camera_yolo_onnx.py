import cv2
import numpy as np
import onnxruntime as ort
import os
import time

def preprocess(img, imgsz=64):
    """將影像預處理成 YOLOv8-cls ONNX 格式"""
    # 1. 調整大小
    img = cv2.resize(img, (imgsz, imgsz))
    # 2. BGR 轉 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. 轉換為浮點數並正規化 (/255.0)
    img = img.astype(np.float32) / 255.0
    # 4. HWC 轉 CHW (高度x寬度x通道 -> 通道x高度x寬度)
    img = img.transpose(2, 0, 1)
    # 5. 增加 Batch 維度 (1, 3, 64, 64)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    # 載入 ONNX 模型
    model_path = os.path.join(os.path.dirname(__file__), 'rps_yolo_model.onnx')
    if not os.path.exists(model_path):
        print(f"❌ 找不到 ONNX 模型檔案: {model_path}")
        return

    # 啟動 ONNX 推論引擎
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # 開啟相機
    cap = cv2.VideoCapture(0)
    
    # 標籤對應
    label_names = ['Rock', 'Paper', 'Scissors']
    
    # 計時器設定
    last_prediction_time = time.time()
    prediction_interval = 5
    last_label = "Waiting..."
    last_conf = 0.0

    print(f"啟動 YOLO ONNX (輕量版)... 每 {prediction_interval} 秒判定一次。按 'q' 鍵退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - last_prediction_time
        countdown = max(0, int(prediction_interval - elapsed_time))

        if elapsed_time >= prediction_interval:
            # 1. 預處理
            input_tensor = preprocess(frame, imgsz=64)
            
            # 2. 推論
            outputs = session.run(None, {input_name: input_tensor})
            
            # 3. 解析結果 (Softmax 轉換)
            raw_output = outputs[0][0]
            # 簡單的 Softmax 或 Argmax
            prediction = np.argmax(raw_output)
            last_label = label_names[prediction]
            
            # 取得信心度 (簡易 Softmax)
            exp_out = np.exp(raw_output - np.max(raw_output))
            probs = exp_out / exp_out.sum()
            last_conf = probs[prediction]
            
            last_prediction_time = current_time
            print(f"📸 ONNX 判定結果: {last_label} ({last_conf:.2f})")

        # --- 在畫面上繪製資訊 ---
        timer_text = f"Next YOLO(ONNX) Photo in: {countdown}s"
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        result_text = f"Result: {last_label} ({last_conf:.2f})"
        cv2.putText(frame, result_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if elapsed_time < 0.3:
             cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)

        cv2.imshow("YOLO ONNX RSP 5s Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

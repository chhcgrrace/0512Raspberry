# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import os
import time

def main():
    # 載入模型
    model_path = os.path.join(os.path.dirname(__file__), 'rps_yolo_model.pt')
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return

    model = YOLO(model_path)
    
    # 開啟相機
    cap = cv2.VideoCapture(0)
    
    # 初始化變數
    last_prediction_time = time.time()
    prediction_interval = 5  # 每 5 秒拍一次
    last_label = "Waiting..."
    last_conf = 0.0
    
    print(f"啟動相機中... 每 {prediction_interval} 秒判定一次。按 'q' 鍵退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - last_prediction_time
        countdown = max(0, int(prediction_interval - elapsed_time))

        # 時間到，執行拍照判定
        if elapsed_time >= prediction_interval:
            results = model(frame, verbose=False)
            for result in results:
                probs = result.probs
                if probs is not None:
                    top1_idx = probs.top1
                    last_conf = probs.top1conf.item()
                    last_label = result.names[top1_idx]
            
            last_prediction_time = current_time
            print(f"📸 拍照判定結果: {last_label} ({last_conf:.2f})")

        # --- 在畫面上繪製資訊 ---
        # 顯示倒數計時
        timer_text = f"Next photo in: {countdown}s"
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 顯示上次判定的結果
        result_text = f"Last Result: {last_label} ({last_conf:.2f})"
        # 根據信心度改變顏色
        color = (0, 255, 0) if last_conf > 0.6 else (0, 165, 255)
        cv2.putText(frame, result_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 拍照瞬間閃一下紅框（提示感）
        if elapsed_time < 0.3:
             cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[2] if len(frame.shape)>2 else frame.shape[0]), (0, 0, 255), 10)

        cv2.imshow("YOLO RSP 5s Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

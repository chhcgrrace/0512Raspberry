
import cv2
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
    last_label = "Waiting..."
    last_conf = 0.0
    
    print("啟動相機中... 即時預測手勢。按 'q' 鍵退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 執行即時辨識
        results = model(frame, verbose=False)
        for result in results:
            probs = result.probs
            if probs is not None:
                top1_idx = probs.top1
                last_conf = probs.top1conf.item()
                raw_label = result.names[top1_idx]
                
                label_map = {"rock": "Rock", "paper": "Paper", "scissors": "Scissors"}
                mapped_label = label_map.get(str(raw_label).lower(), str(raw_label))
                
                # 信心度太低或者是未知手勢，歸類為 Error
                if last_conf < 0.6 or mapped_label not in ["Rock", "Paper", "Scissors"]:
                    last_label = "Error"
                else:
                    last_label = mapped_label

        # 顯示即時判定結果
        result_text = f"Result: {last_label} ({last_conf:.2f})"
        # 根據信心度改變顏色
        color = (0, 255, 0) if last_conf > 0.6 else (0, 165, 255)
        cv2.putText(frame, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("YOLO RSP Real-time Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import joblib
import numpy as np
import os
import time

def main():
    # 載入 SVM 模型
    model_path = os.path.join(os.path.dirname(__file__), 'rps_svm_model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        print("請先執行 train/train_svm.py 進行訓練。")
        return

    clf = joblib.load(model_path)
    
    # 開啟相機
    cap = cv2.VideoCapture(0)
    
    # 初始化變數
    last_prediction_time = time.time()
    prediction_interval = 5  # 每 5 秒拍一次
    last_label = "Waiting..."
    last_conf = 0.0
    
    # 標籤對應 (需與訓練時一致)
    label_names = ['Rock', 'Paper', 'Scissors']
    
    print(f"啟動 SVM 相機... 每 {prediction_interval} 秒判定一次。按 'q' 鍵退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - last_prediction_time
        countdown = max(0, int(prediction_interval - elapsed_time))

        # 時間到，執行拍照判定
        if elapsed_time >= prediction_interval:
            # --- SVM 影像前處理 ---
            # 1. 轉灰階
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2. 縮放至 64x64
            resized = cv2.resize(gray, (64, 64))
            # 3. 攤平並正規化
            flat_data = resized.flatten().reshape(1, -1) / 255.0
            
            # 4. 預測
            prediction = clf.predict(flat_data)[0]
            last_label = label_names[prediction]
            
            # 取得信心度 (如果模型有支援 probability=True)
            try:
                probs = clf.predict_proba(flat_data)[0]
                last_conf = probs[prediction]
            except:
                last_conf = 1.0 # 如果當初訓練沒開機率功能，預設顯示 1.0
            
            last_prediction_time = current_time
            print(f"📸 SVM 判定結果: {last_label}")

        # --- 在畫面上繪製資訊 ---
        timer_text = f"Next SVM Photo in: {countdown}s"
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        result_text = f"SVM Result: {last_label}"
        cv2.putText(frame, result_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 拍照瞬間閃一下框
        if elapsed_time < 0.3:
             cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10)

        cv2.imshow("SVM RSP 5s Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

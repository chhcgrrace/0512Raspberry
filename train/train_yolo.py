import os
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

def main():
    # 取得專案根目錄
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset')
    
    # 1. 載入預訓練的 YOLOv8n-cls 模型 (Nano 分類版本，最適合 RPi)
    # 這裡會自動從網路上下載 yolov8n-cls.pt
    model = YOLO('yolov8n-cls.pt')

    # 2. 開始訓練
    print("=== 開始訓練 YOLOv8-Nano 分類模型 ===")
    results = model.train(
        data=dataset_path, 
        epochs=10, 
        imgsz=64, 
        project=os.path.join(base_dir, 'train', 'runs'),
        name='rps_yolo'
    )

    # 3. 匯出模型到 demo 資料夾
    # 使用 results.save_dir 自動取得正確的輸出路徑
    save_dir = str(results.save_dir)
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    target_path = os.path.join(base_dir, 'demo', 'rps_yolo_model.pt')
    
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, target_path)
        print(f"✅ 模型訓練完成並已複製到: {target_path}")
    else:
        print(f"⚠️ 找不到權重檔案於: {best_model_path}")

if __name__ == "__main__":
    main()

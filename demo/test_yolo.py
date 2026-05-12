import os
from ultralytics import YOLO

def main():
    # 定義路徑
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'demo', 'rps_yolo_model.pt')
    dataset_path = os.path.join(base_dir, 'dataset')

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return

    # 載入模型
    model = YOLO(model_path)

    # 評估模型 (驗證模式)
    print("=== 正在評估 YOLO 模型在測試集上的表現 ===")
    # split='test' 表示使用 dataset/test 資料夾
    metrics = model.val(data=dataset_path, split='test')

    # 輸出結果
    print("\n📊 測試結果:")
    if hasattr(metrics, 'top1'):
        print(f"🎯 Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
        print(f"🎯 Top-5 Accuracy: {metrics.top5 * 100:.2f}%")
    else:
        print("模型評估完成，請查看產生的報告。")

if __name__ == "__main__":
    main()

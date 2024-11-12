import cv2
import torch
from facenet_pytorch import MTCNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    cap = cv2.VideoCapture(0)
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)
            if boxes is not None:
                for box in boxes:
                    box = box.astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

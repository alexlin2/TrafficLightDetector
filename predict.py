import torch 
import numpy as np
import cv2 
import glob
from model import trafficLightDetectionModel
from torchvision.transforms import functional as F

class Predictor():

    def __init__(self, threshold = 0.5, write_dir = 'results/', device = 'cuda'):
        self.device = device
        self.threshhold = threshold
        self.write_dir = write_dir
        self.last_label = len(glob.glob(write_dir + 'frame_*.png'))

    def read_img(self, img_path):
        return cv2.imread(img_path)

    def predict_video(self, model, classes, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_size = (frame_width,frame_height)
        fps = 30
        output = cv2.VideoWriter('results/video.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), fps, frame_size)
        success, img = cap.read()
        while(success):
            print("processing")
            x = self.process_img(img).to(self.device)
            predictions = self.predict(model, x)
            self.draw_image(img, predictions, classes)
            output.write(img)
            success, img = cap.read()
        cap.release()
        output.release()


    def predict_image(self, model, classes, img_path):
        img = self.read_img(img_path)
        x = self.process_img(img).to(self.device)
        predictions = p1.predict(model, x)
        p1.draw_image(img, predictions, classes)
        p1.save_result(img)

    def process_img(self, img):
        return F.to_tensor(img)
        
    def predict(self, model, x):
        with torch.no_grad():
            predictions = model([x])
            predictions = {k: v.to('cpu').data.numpy()
                           for k, v in predictions[0].items()}

        return predictions

    def draw_image(self, img, predictions, classes):
        boxes, labels, scores = predictions['boxes'], predictions['labels'], predictions['scores']
        for i, box in enumerate(boxes):
            if scores[i] > self.threshhold:
                x1,y1,x2,y2 = box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
				(0, 255, 255), 2)
                cv2.putText(img, classes[labels[i]], (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    def save_result(self, img):
        cv2.imwrite(self.write_dir + 'frame_{:03d}.png'.format(self.last_label), img)
        self.last_label += 1

if __name__ == "__main__":
    classes = ['background', 'GreenLeft', 'RedStraightLeft', 'RedLeft', 'off', 'GreenStraight', 'GreenStraightRight',
             'GreenStraightLeft', 'RedStraight', 'GreenRight', 'Green', 'Yellow', 'RedRight', 'Red']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoints = torch.load('checkpoints/last_checkpoint.pt')
    model = trafficLightDetectionModel(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoints['weights'])
    model.eval()

    p1 = Predictor()
    img = p1.read_img('/home/alexlin/traffic_net/dataset_train_rgb/rgb/train/2015-10-05-10-55-33_bag/82800.png')
    x = p1.process_img(img).to(device)
    predictions = p1.predict(model, x)
    p1.draw_image(img, predictions, classes)
    while True:
        cv2.imshow('test', img)
        if cv2.waitKey(1) == 27:   
            break
    p1.save_result(img)
    cv2.destroyAllWindows()
    

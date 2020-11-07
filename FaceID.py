import cv2
import os
from os.path import join
import shutil
import numpy as np
import torch
from collections import defaultdict
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import copy
import argparse


class FaceID:
    def __init__(self, device='cpu'):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device, keep_all=True)

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def extract_dataset_faces(self, dataset_path):
        cls_names = os.listdir(dataset_path)
        for cls_name in cls_names:
            aligned_folder = join(dataset_path, cls_name, 'aligned')
            if os.path.isdir(aligned_folder):
                shutil.rmtree(aligned_folder)
            os.mkdir(aligned_folder)

            embeddings = {}
            cls_folder_path = join(dataset_path, cls_name, 'nonaligned')
            for img_name in os.listdir(cls_folder_path):
                img_path = join(cls_folder_path, img_name)
                img = cv2.imread(img_path)

                imgs_aligned, prob = self.mtcnn(img, return_prob=True)

                for i, img_aligned in enumerate(imgs_aligned):
                    with torch.no_grad():
                        embedding = self.resnet(img_aligned[None]).detach().numpy()
                        embeddings[str(i) + img_name] = embedding.squeeze()
                    dst_path = join(aligned_folder, str(i) + img_name)
                    img_aligned = self.tensor2numpy(img_aligned)
                    cv2.imwrite(dst_path, img_aligned)
            emmbedings_dst_path = join(dataset_path, cls_name, 'emmbedings.pkl')
            with open(emmbedings_dst_path, 'wb') as f:
                pickle.dump(embeddings, f)

    def tensor2numpy(self, tensor_img):
        numpy_img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
        numpy_img = ((numpy_img + 1) * 128).astype(np.uint8)
        return numpy_img

    def load_embedings(self, dataset_dir):
        self.cls_emmbedings = defaultdict(list)
        cls_names = os.listdir(dataset_dir)
        for cls_name in cls_names:
            pickle_path = join(dataset_dir, cls_name, 'emmbedings.pkl')
            with open(pickle_path, 'rb') as f:
                emmbedings_dict = pickle.load(f)
                for img_name, emmbding in emmbedings_dict.items():
                    img_path = join(dataset_dir, cls_name, 'aligned', img_name)
                    if not os.path.isfile(img_path):
                        continue
                    self.cls_emmbedings[cls_name].append(emmbding)

        for name, emmbedings in copy.copy(self.cls_emmbedings).items():
            self.cls_emmbedings[name] = np.array(emmbedings).mean(axis=0)

    def detect_and_process(self, img):
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
        faces = self.mtcnn.extract(img, batch_boxes, save_path=None)

        ret_list = []
        if faces is None:
            return ret_list

        for face, box in zip(faces, batch_boxes):
            with torch.no_grad():
                emmbeding = self.resnet(face.unsqueeze(dim=0)).detach().numpy()[0]

            min_distance = 1e26
            final_cls = 'NoName'
            for cls_name, c_emmbedings in self.cls_emmbedings.items():
                dist = (c_emmbedings - emmbeding)**2
                dist = np.sum(dist)
                dist = np.sqrt(dist)

                if dist < min_distance and dist < 1.05:
                    min_distance = dist
                    final_cls = cls_name

            ret_list.append((box, final_cls))
        return ret_list


def draw_anno_and_show(img, dets, waitkey_delay):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    for det in dets:
        box, cls_name = det
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        org = (box[0], box[1])
        img = cv2.putText(img, cls_name, org, font,
               fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('FaceID', img)
    cv2.waitKey(waitkey_delay)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--prep_features', action='store_true')
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--webcam_id', type=int)
    parser.add_argument('--img_path', type=str)
    args = parser.parse_args()


    face_id = FaceID()
    if not args.prep_features:
        face_id.load_embedings(args.dataset_dir)

    if args.prep_features:
        face_id.extract_dataset_faces(args.dataset_dir)
    elif not args.video_path is None or not args.webcam_id is None:
        video2load = args.webcam_id if args.video_path is None else args.video_path
        cap = cv2.VideoCapture(video2load)
        ret, img = cap.read()
        while ret:
            dets = face_id.detect_and_process(img)
            draw_anno_and_show(img, dets, waitkey_delay=1)
            ret, img = cap.read()
    elif not args.img_path is None:
        img = cv2.imread(args.img_path)
        dets = face_id.detect_and_process(img)
        draw_anno_and_show(img, dets, waitkey_delay=0)

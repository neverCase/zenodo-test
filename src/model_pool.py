import os

os.environ['HF_HOME'] = '/root/data/cache/'

import logging
from queue import Queue
from threading import Lock

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort
import scipy.special

class Fold5ModelPool:
    def __init__(self, pool_size=1, model_prefix="2026-03-03-16-15-27"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.model_prefix = model_prefix
        self.pool_size = pool_size  # 池的大小
        self.model_pool = Queue(maxsize=self.pool_size)  # 使用队列管理模型池
        self.lock = Lock()  # 线程安全

        self._initialize_pool()  # 初始化模型池

    def _initialize_pool(self):
        logging.info("Initializing model pool")
        for _ in range(self.pool_size):
            # 初始化模型并放入池中
            model = self._create_model()
            self.model_pool.put(model)

    def _create_model(self):
        logging.info("init models")

        fold5_models = []
        for one in range(5):
            model_path = f"{self.model_prefix}_best_model_fold{one + 1}.pth"
            model = models.resnet18(pretrained=True)
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(model.fc.in_features, 1)
            )
            model = model.to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            fold5_models.append(model)

        return fold5_models

    def get_model(self):
        with self.lock:
            try:
                # 等待直到池中有可用的模型
                return self.model_pool.get(block=True, timeout=5)
            except Exception as e:
                logging.error(f"Error getting model from pool: {e}")
                return None

    def return_model(self, model):
        with self.lock:
            if self.model_pool.full():
                logging.info("Model pool is full, not adding more models.")
            else:
                self.model_pool.put(model)

    def load_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = np.stack([img] * 3, axis=2)  # 复制成3通道
        img = self.preprocess(img)
        return img

    def predict(self, image_path):
        img_tensor = self.load_image(image_path)

        probs = []

        fold5 = self.get_model()
        for fold in fold5:
            with torch.no_grad():
                output = fold(img_tensor.unsqueeze(0).to(self.device))  # [1,1]
                prob = torch.sigmoid(output).item()
                probs.append(prob)
        self.return_model(fold5)

        print(f"probs: {probs}")
        final_prob = sum(probs) / len(probs)
        final_pred = 1 if final_prob > 0.70 else 0
        print(f"{image_path} -> prob: {prob:.4f}, final_prob: {final_prob:.4f}, final_pred: {final_pred}")
        return final_pred


class Fold5ModelOnnxCpuPool:
    def __init__(self, pool_size=1, model_prefix="2026-03-03-16-15-27"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.model_prefix = model_prefix
        self.pool_size = pool_size  # 池的大小
        self.model_pool = Queue(maxsize=self.pool_size)  # 使用队列管理模型池
        self.lock = Lock()  # 线程安全

        self._initialize_pool()  # 初始化模型池

    def _initialize_pool(self):
        logging.info("Initializing model pool")
        for _ in range(self.pool_size):
            # 初始化模型并放入池中
            model = self._create_model()
            self.model_pool.put(model)

    def _create_model(self):
        logging.info("init models")

        fold5_models = []
        for one in range(5):
            model_path = f"{self.model_prefix}_best_model_fold{one + 1}.onnx"
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            model = {
                "session": session,
                "input_name": input_name,
            }
            fold5_models.append(model)

        return fold5_models

    def get_model(self):
        with self.lock:
            try:
                # 等待直到池中有可用的模型
                return self.model_pool.get(block=True, timeout=5)
            except Exception as e:
                logging.error(f"Error getting model from pool: {e}")
                return None

    def return_model(self, model):
        with self.lock:
            if self.model_pool.full():
                logging.info("Model pool is full, not adding more models.")
            else:
                self.model_pool.put(model)

    def load_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = np.stack([img] * 3, axis=2)

        input_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        input_tensor = input_tensor.unsqueeze(0)  # 增加 batch 维度: (1, 3, 224, 224)


        return input_tensor

    def predict(self, image_path):
        img_tensor = self.load_image(image_path)

        probs = []

        fold5 = self.get_model()
        for fold in fold5:
            onnx_output = fold['session'].run(None, {fold['input_name']: img_tensor.numpy().astype(np.float32)})[0]
            logit = onnx_output[0][0]
            prob = scipy.special.expit(logit)
            probs.append(prob)

        self.return_model(fold5)

        print(f"probs: {probs}")
        final_prob = sum(probs) / len(probs)
        final_pred = 1 if final_prob > 0.70 else 0
        print(f"{image_path} -> prob: {prob:.4f}, final_prob: {final_prob:.4f}, final_pred: {final_pred}")
        return final_pred

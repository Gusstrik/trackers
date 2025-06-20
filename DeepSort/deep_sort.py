import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import cv2
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
import yaml
import json
from ultralytics import YOLO

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Используем предобученный ResNet50 без последнего слоя
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        return self.features(x).squeeze()

class Track:
    def __init__(self, bbox, track_id, max_age=30):
        self.track_id = track_id
        self.max_age = max_age
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.history = []
        
        # Инициализация Kalman Filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # [x, y, s, r, x', y', s']
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Настройка ковариационных матриц
        self.kf.R[2:, 2:] *= 10.  # Увеличение неопределенности для размера
        self.kf.P[4:, 4:] *= 1000.  # Увеличение неопределенности для скорости
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Инициализация состояния
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        # Сохранение признаков
        self.features = []
        
    def _convert_bbox_to_z(self, bbox):
        """Преобразование bbox [x1,y1,x2,y2] в формат [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """Преобразование состояния [x,y,s,r] в bbox [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    
    def predict(self):
        """Предсказание следующего состояния"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def update(self, bbox, features):
        """Обновление трека с новым bbox и признаками"""
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        self.features.append(features)
        if len(self.features) > 100:  # Ограничение истории признаков
            self.features.pop(0)
    
    def get_state(self):
        """Получение текущего состояния трека"""
        return self._convert_x_to_bbox(self.kf.x)

class DeepSORT:
    def __init__(self, max_cosine_distance=0.3, nn_budget=100, max_age=30):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.max_age = max_age
        self.tracks = []
        self.track_id = 0
        self.feature_extractor = FeatureExtractor()
        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
        self.feature_extractor.eval()
        
    def _extract_features(self, frame, bbox):
        """Извлечение признаков из области изображения"""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(2048)
            
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros(2048)
            
        # Изменение размера патча для ResNet
        patch = cv2.resize(patch, (224, 224))
        patch = patch.transpose(2, 0, 1)
        patch = torch.from_numpy(patch).float().unsqueeze(0)
        patch = patch / 255.0
        
        if torch.cuda.is_available():
            patch = patch.cuda()
            
        with torch.no_grad():
            features = self.feature_extractor(patch)
            
        return features.cpu().numpy()
    
    def _cosine_distance(self, a, b):
        """Вычисление косинусного расстояния между признаками"""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return 1 - np.dot(a, b)
    
    def _match_detections_to_tracks(self, detections, tracks):
        """Сопоставление детекций с треками используя Hungarian algorithm"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
            
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = self._cosine_distance(
                    track.features[-1], detection[1])
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > self.max_cosine_distance:
                unmatched_detections.append(j)
                unmatched_tracks.append(i)
            else:
                matches.append((i, j))
                if j in unmatched_detections:
                    unmatched_detections.remove(j)
                if i in unmatched_tracks:
                    unmatched_tracks.remove(i)
                    
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections, frame):
        """Обновление треков с новыми детекциями"""
        # Извлечение признаков для всех детекций
        features = []
        for det in detections:
            bbox = det[:4]
            feat = self._extract_features(frame, bbox)
            features.append((bbox, feat))
            
        # Предсказание новых состояний для всех треков
        for track in self.tracks:
            track.predict()
            
        # Сопоставление детекций с треками
        matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            features, self.tracks)
            
        # Обновление сопоставленных треков
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                features[detection_idx][0], features[detection_idx][1])
            
        # Создание новых треков для несоответствующих детекций
        for detection_idx in unmatched_detections:
            bbox, feat = features[detection_idx]
            self.tracks.append(Track(bbox, self.track_id, self.max_age))
            self.tracks[-1].update(bbox, feat)
            self.track_id += 1
            
        # Удаление старых треков
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        return self.tracks

def load_ground_truth(json_path):
    """Загрузка ground truth аннотаций из JSON файла"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    ground_truth = {}
    for item in data:
        frame_id = item['id'] - 196
        frame_name = f"frame_{frame_id:04d}.jpg"
        
        annotations = []
        for result in item['annotations'][0]['result']:
            value = result['value']
            x = value['x'] * 1920 / 100
            y = value['y'] * 1080 / 100
            width = value['width'] * 1920 / 100
            height = value['height'] * 1080 / 100
            
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            
            annotations.append({
                'bbox': [x1, y1, x2, y2],
                'id': value['rectanglelabels'][0]
            })
        
        ground_truth[frame_name] = annotations
    
    return ground_truth

def calculate_iou(box1, box2):
    """Вычисление IoU между двумя bounding box"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_metrics(tracked_objects, ground_truth, frame_names):
    """Вычисление метрик трекинга"""
    total_frames = len(frame_names)
    
    unique_gt_objects = set()
    for frame in ground_truth.values():
        for gt_obj in frame:
            unique_gt_objects.add(gt_obj['id'])
    
    total_objects = len(unique_gt_objects)
    total_tracks = len(tracked_objects)
    
    track_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
    
    total_false_positives = 0
    total_false_negatives = 0
    total_true_positives = 0
    total_gt_objects = 0
    
    for frame_name in frame_names:
        if frame_name not in ground_truth:
            continue
            
        gt_objects = ground_truth[frame_name]
        frame_tracks = []
        
        for track_id, track_history in tracked_objects.items():
            for track in track_history:
                if track[0] == frame_name:
                    frame_tracks.append((track_id, track[1:5]))
                    break
        
        matched_gt = set()
        matched_tracks = set()
        
        for gt_obj in gt_objects:
            gt_id = gt_obj['id']
            gt_bbox = gt_obj['bbox']
            total_gt_objects += 1
            
            best_iou = 0
            best_track = None
            
            for track_id, track_bbox in frame_tracks:
                if track_id in matched_tracks:
                    continue
                    
                iou = calculate_iou(gt_bbox, track_bbox)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_track = track_id
            
            if best_track is not None:
                track_stats[gt_id]['hits'] += 1
                total_true_positives += 1
                matched_gt.add(gt_id)
                matched_tracks.add(best_track)
            else:
                total_false_negatives += 1
                
            track_stats[gt_id]['total'] += 1
        
        total_false_positives += len(frame_tracks) - len(matched_tracks)
    
    mt = 0
    ml = 0
    
    for gt_id in unique_gt_objects:
        if gt_id in track_stats:
            coverage = track_stats[gt_id]['hits'] / track_stats[gt_id]['total']
            if coverage > 0.8:
                mt += 1
            elif coverage < 0.2:
                ml += 1
    
    mota = 1 - (total_false_positives + total_false_negatives) / total_gt_objects if total_gt_objects > 0 else 0
    motp = total_true_positives / total_gt_objects if total_gt_objects > 0 else 0
    
    return {
        'mt': mt,
        'ml': ml,
        'mota': mota,
        'motp': motp,
        'total_objects': total_objects,
        'total_tracks': total_tracks,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives,
        'true_positives': total_true_positives,
        'total_gt_objects': total_gt_objects,
        'mt_percentage': mt / total_objects if total_objects > 0 else 0,
        'ml_percentage': ml / total_objects if total_objects > 0 else 0
    }

def visualize_tracks(frame, frame_name, tracked_objects, ground_truth, output_dir='track_visualizations'):
    """Визуализация треков и ground truth на кадре"""
    os.makedirs(output_dir, exist_ok=True)
    
    vis_frame = frame.copy()
    
    if frame_name in ground_truth:
        for gt_obj in ground_truth[frame_name]:
            gt_id = gt_obj['id']
            x1, y1, x2, y2 = map(int, gt_obj['bbox'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f'GT: {gt_id}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for track_id, track_history in tracked_objects.items():
        for track in track_history:
            if track[0] == frame_name:
                _, x1, y1, x2, y2, conf, cls = track
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_frame, f'Track {track_id}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break
    
    cv2.putText(vis_frame, f'Frame: {frame_name}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    output_path = os.path.join(output_dir, f'tracks_{frame_name}')
    cv2.imwrite(output_path, vis_frame)
    return output_path

def evaluate_model(model_path=None):
    if model_path is None:
        model_path = 'runs/train/yolov8_car_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена по пути: {model_path}\n"
            "Убедитесь, что обучение было успешно завершено и модель сохранена."
        )
    
    print(f"Загрузка модели из {model_path}")
    model = YOLO(model_path)
    
    dataset_path = os.path.join('./datasets', 'cam_dataset')
    val_path = os.path.join(dataset_path, 'val', 'images')
    
    gt_path = os.path.join(dataset_path, 'val', 'IdentityCars.json')
    ground_truth = load_ground_truth(gt_path)
    
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Валидационный датасет не найден по пути: {val_path}\n"
            "Убедитесь, что датасет находится в правильной директории."
        )
    
    print("Начинаем оценку модели на валидационном наборе данных...")
    
    val_args = {
        'data': os.path.join(dataset_path, 'data.yaml'),
        'imgsz': 640,
        'batch': 16,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'verbose': True,
        'conf': 0.25,
        'iou': 0.45,
    }
    
    results = model.val(**val_args)
    
    try:
        with open(os.path.join(dataset_path, 'data.yaml'), 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
    except UnicodeDecodeError:
        encodings = ['cp1251', 'latin1', 'utf-16']
        for encoding in encodings:
            try:
                with open(os.path.join(dataset_path, 'data.yaml'), 'r', encoding=encoding) as f:
                    data_config = yaml.safe_load(f)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise Exception("Не удалось прочитать файл data.yaml с поддерживаемыми кодировками")
    
    tracker = DeepSORT(max_cosine_distance=0.3, nn_budget=100, max_age=30)
    
    val_images = sorted([f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    track_info = defaultdict(list)
    
    total_time = 0
    frame_times = []
    
    vis_dir = 'track_visualizations_deep_sort'
    os.makedirs(vis_dir, exist_ok=True)
    
    for img_name in val_images:
        img_path = os.path.join(val_path, img_name)
        img = cv2.imread(img_path)
        
        start_time = time.time()
        
        pred_results = model.predict(img_path, conf=0.25, save=False)
        
        detections = pred_results[0].boxes.data.cpu().numpy()
        tracks = tracker.update(detections, img)
        
        for track in tracks:
            bbox = track.get_state()
            track_info[track.track_id].append((
                img_name,
                int(bbox[0][0]), int(bbox[0][1]),
                int(bbox[0][2]), int(bbox[0][3]),
                float(track.hits / max(track.age, 1)),
                track.track_id
            ))
        
        vis_path = visualize_tracks(img, img_name, track_info, ground_truth, vis_dir)
        print(f"Сохранена визуализация: {vis_path}")
        
        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time
        frame_times.append(frame_time)
    
    metrics = calculate_metrics(track_info, ground_truth, val_images)
    
    print("\nРезультаты отслеживания:")
    print(f"Среднее время обработки кадра: {total_time/len(val_images):.4f} секунд")
    print(f"MT (Mostly Tracked): {metrics['mt']}")
    print(f"ML (Mostly Lost): {metrics['ml']}")
    print(f"MOTA: {metrics['mota']:.4f}")
    print(f"MOTP: {metrics['motp']:.4f}")
    print(f"Всего объектов: {metrics['total_objects']}")
    print(f"Всего треков: {metrics['total_tracks']}")
    
    results_file = 'validation_results_deep_sort.txt'
    with open(results_file, 'w') as f:
        f.write(f"Результаты валидации модели YOLOv8 с DeepSORT:\n")
        f.write(f"Путь к модели: {model_path}\n")
        f.write(f"Среднее время обработки кадра: {total_time/len(val_images):.4f} секунд\n")
        f.write(f"MT (Mostly Tracked): {metrics['mt']}\n")
        f.write(f"ML (Mostly Lost): {metrics['ml']}\n")
        f.write(f"MOTA: {metrics['mota']:.4f}\n")
        f.write(f"MOTP: {metrics['motp']:.4f}\n")
        f.write(f"Всего объектов: {metrics['total_objects']}\n")
        f.write(f"Всего треков: {metrics['total_tracks']}\n")
        
        f.write("\nМетрики YOLO:\n")
        f.write(f"mAP50: {results.box.map50:.4f}\n")
        f.write(f"mAP50-95: {results.box.map:.4f}\n")
    
    print(f"\nРезультаты сохранены в файл: {results_file}")
    print(f"Визуализации сохранены в директории: {vis_dir}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка модели YOLOv8 с DeepSORT')
    parser.add_argument('--model', type=str, help='Путь к обученной модели (.pt файл)')
    args = parser.parse_args()
    
    evaluate_model(args.model)

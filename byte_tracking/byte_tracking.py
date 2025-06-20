from ultralytics import YOLO
import os
import torch
from pathlib import Path
import cv2
import numpy as np
import yaml
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple

class ByteTracker:
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.tracked_objects: Dict[int, List[Tuple[str, int, int, int, int, float, int]]] = defaultdict(list)
        self.current_id = 0
        self.frame_count = 0
        self.active_tracks = {}  # Словарь активных треков
        self.track_history = {}  # История треков

    def update(self, detections: np.ndarray, frame: np.ndarray, frame_name: str) -> Tuple[np.ndarray, Dict[int, List[Tuple[str, int, int, int, int, float, int]]]]:
        """
        Update tracks with new detections
        detections: [x1, y1, x2, y2, conf, cls]
        """
        self.frame_count += 1
        
        if len(detections) > 0:
            # Обновляем существующие треки
            for track_id, track_info in list(self.active_tracks.items()):
                last_frame, last_box = track_info
                last_x1, last_y1, last_x2, last_y2 = last_box
                
                # Ищем ближайшее обнаружение
                best_iou = 0
                best_det = None
                
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if conf > self.track_thresh:
                        # Вычисляем IoU
                        intersection_x1 = max(last_x1, x1)
                        intersection_y1 = max(last_y1, y1)
                        intersection_x2 = min(last_x2, x2)
                        intersection_y2 = min(last_y2, y2)
                        
                        if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                            last_area = (last_x2 - last_x1) * (last_y2 - last_y1)
                            det_area = (x2 - x1) * (y2 - y1)
                            union_area = last_area + det_area - intersection_area
                            iou = intersection_area / union_area if union_area > 0 else 0
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_det = det
                
                # Если нашли подходящее обнаружение, обновляем трек
                if best_iou > 0.3:  # Порог IoU для обновления трека
                    x1, y1, x2, y2, conf, cls = best_det
                    self.active_tracks[track_id] = (frame_name, (int(x1), int(y1), int(x2), int(y2)))
                    self.tracked_objects[track_id].append((frame_name, int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
                else:
                    # Удаляем трек, если не нашли подходящее обнаружение
                    del self.active_tracks[track_id]
            
            # Создаем новые треки для несоответствующих обнаружений
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf > self.track_thresh:
                    # Проверяем, не соответствует ли обнаружение существующему треку
                    is_new_track = True
                    for track_info in self.active_tracks.values():
                        last_frame, last_box = track_info
                        last_x1, last_y1, last_x2, last_y2 = last_box
                        
                        # Вычисляем IoU
                        intersection_x1 = max(last_x1, x1)
                        intersection_y1 = max(last_y1, y1)
                        intersection_x2 = min(last_x2, x2)
                        intersection_y2 = min(last_y2, y2)
                        
                        if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                            last_area = (last_x2 - last_x1) * (last_y2 - last_y1)
                            det_area = (x2 - x1) * (y2 - y1)
                            union_area = last_area + det_area - intersection_area
                            iou = intersection_area / union_area if union_area > 0 else 0
                            
                            if iou > 0.3:
                                is_new_track = False
                                break
                    
                    if is_new_track:
                        self.active_tracks[self.current_id] = (frame_name, (int(x1), int(y1), int(x2), int(y2)))
                        self.tracked_objects[self.current_id].append((frame_name, int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
                        self.current_id += 1

        return frame, self.tracked_objects

def load_ground_truth(json_path):
    """Load ground truth annotations from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    ground_truth = {}
    for item in data:
        # Convert task ID to frame number
        frame_id = item['id'] - 196
        frame_name = f"frame_{frame_id:04d}.jpg"
        
        # Extract annotations
        annotations = []
        for result in item['annotations'][0]['result']:
            value = result['value']
            # Convert percentage coordinates to pixel coordinates
            x = value['x'] * 1920 / 100
            y = value['y'] * 1080 / 100
            width = value['width'] * 1920 / 100
            height = value['height'] * 1080 / 100
            
            # Convert to [x1, y1, x2, y2] format
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            
            annotations.append({
                'bbox': [x1, y1, x2, y2],
                'id': value['rectanglelabels'][0]  # Car_1, Car_2, etc.
            })
        
        ground_truth[frame_name] = annotations
    
    # Print example of a car object
    print("\nExample of a car object from ground truth:")
    for frame_name, frame_annotations in ground_truth.items():
        for ann in frame_annotations:
            if ann['id'].startswith('Car'):
                print(f"Frame: {frame_name}")
                print(f"ID: {ann['id']}")
                print(f"Bounding box: {ann['bbox']}")
                print(f"Original coordinates: x={ann['bbox'][0]/1920*100:.2f}%, y={ann['bbox'][1]/1080*100:.2f}%, "
                      f"width={ann['bbox'][2]/1920*100:.2f}%, height={ann['bbox'][3]/1080*100:.2f}%")
                return ground_truth
    
    return ground_truth

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
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
    """Calculate tracking metrics"""
    # Initialize metrics
    total_frames = len(frame_names)
    
    # Get unique ground truth object IDs
    unique_gt_objects = set()
    for frame in ground_truth.values():
        for gt_obj in frame:
            unique_gt_objects.add(gt_obj['id'])
    
    total_objects = len(unique_gt_objects)
    total_tracks = len(tracked_objects)
    
    print(f"\nDebug information:")
    print(f"Total unique ground truth objects: {total_objects}")
    print(f"Total tracks: {total_tracks}")
    
    # Calculate tracking statistics
    track_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
    
    # Initialize counters for MOTA calculation
    total_false_positives = 0
    total_false_negatives = 0
    total_true_positives = 0
    total_gt_objects = 0
    
    for frame_name in frame_names:
        if frame_name not in ground_truth:
            continue
            
        gt_objects = ground_truth[frame_name]
        frame_tracks = []
        
        # Get tracks for this frame
        for track_id, track_history in tracked_objects.items():
            for track in track_history:
                if track[0] == frame_name:
                    frame_tracks.append((track_id, track[1:5]))  # [x1, y1, x2, y2]
                    break
        
        # Match tracks with ground truth
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
                if iou > best_iou and iou > 0.5:  # IoU threshold
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
        
        # Count false positives (unmatched tracks)
        total_false_positives += len(frame_tracks) - len(matched_tracks)
    
    # Print tracking statistics for each object
    print("\nTracking statistics for each object:")
    for gt_id in unique_gt_objects:
        if gt_id in track_stats:
            stats = track_stats[gt_id]
            coverage = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
            print(f"Object {gt_id}: hits={stats['hits']}, total={stats['total']}, coverage={coverage:.2f}")
    
    # Calculate MT and ML
    mt = 0
    ml = 0
    
    for gt_id in unique_gt_objects:
        if gt_id in track_stats:
            coverage = track_stats[gt_id]['hits'] / track_stats[gt_id]['total']
            if coverage > 0.8:
                mt += 1
            elif coverage < 0.2:
                ml += 1
    
    print(f"\nMT calculation: {mt} objects with coverage > 0.8")
    print(f"ML calculation: {ml} objects with coverage < 0.2")
    
    # Calculate MOTA
    mota = 1 - (total_false_positives + total_false_negatives) / total_gt_objects if total_gt_objects > 0 else 0
    
    # Calculate MOTP
    motp = total_true_positives / total_gt_objects if total_gt_objects > 0 else 0
    
    print(f"\nFinal metrics:")
    print(f"Total ground truth objects: {total_gt_objects}")
    print(f"True positives: {total_true_positives}")
    print(f"False positives: {total_false_positives}")
    print(f"False negatives: {total_false_negatives}")
    
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
    """Visualize tracks and ground truth on the frame"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Draw ground truth boxes in green
    if frame_name in ground_truth:
        for gt_obj in ground_truth[frame_name]:
            gt_id = gt_obj['id']
            x1, y1, x2, y2 = map(int, gt_obj['bbox'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f'GT: {gt_id}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw tracked objects in red
    for track_id, track_history in tracked_objects.items():
        for track in track_history:
            if track[0] == frame_name:
                _, x1, y1, x2, y2, conf, cls = track
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_frame, f'Track {track_id}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break
    
    # Add frame information
    cv2.putText(vis_frame, f'Frame: {frame_name}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the visualization
    output_path = os.path.join(output_dir, f'tracks_{frame_name}')
    cv2.imwrite(output_path, vis_frame)
    return output_path

def evaluate_model(model_path=None):
    # Если путь к модели не указан, используем путь по умолчанию
    if model_path is None:
        model_path = 'runs/train/yolov8_car_detection/weights/best.pt'
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена по пути: {model_path}\n"
            "Убедитесь, что обучение было успешно завершено и модель сохранена."
        )
    
    print(f"Загрузка модели из {model_path}")
    # Загружаем модель
    model = YOLO(model_path)
    
    # Путь к валидационному датасету
    dataset_path = os.path.join('./datasets', 'cam_dataset')
    val_path = os.path.join(dataset_path, 'val', 'images')
    
    # Загружаем ground truth
    gt_path = os.path.join(dataset_path, 'val', 'IdentityCars.json')
    ground_truth = load_ground_truth(gt_path)
    
    # Проверяем наличие валидационных данных
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Валидационный датасет не найден по пути: {val_path}\n"
            "Убедитесь, что датасет находится в правильной директории."
        )
    
    print("Начинаем оценку модели на валидационном наборе данных...")
    # Параметры валидации
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
    
    # Запуск валидации
    results = model.val(**val_args)
    
    # Загружаем аннотации из data.yaml
    try:
        with open(os.path.join(dataset_path, 'data.yaml'), 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
    except UnicodeDecodeError:
        # Если UTF-8 не работает, пробуем другие кодировки
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
    
    # Инициализация трекера
    tracker = ByteTracker()
    
    # Получаем все изображения из валидационного набора
    val_images = sorted([f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Словарь для хранения информации о треках
    track_info = defaultdict(list)
    
    # Измерение времени обработки
    total_time = 0
    frame_times = []
    
    # Создаем директорию для визуализаций
    vis_dir = 'track_visualizations_byte_track'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Обработка всех изображений
    for img_name in val_images:
        img_path = os.path.join(val_path, img_name)
        img = cv2.imread(img_path)
        
        # Измеряем время обработки
        start_time = time.time()
        
        # Получаем предсказания
        pred_results = model.predict(img_path, conf=0.25, save=False)
        
        # Обновляем трекер
        detections = pred_results[0].boxes.data.cpu().numpy()
        frame, tracked_objects = tracker.update(detections, img, img_name)
        
        # Сохраняем информацию о треках
        for track_id, track_history in tracked_objects.items():
            track_info[track_id].extend(track_history)
        
        # Визуализируем треки
        vis_path = visualize_tracks(img, img_name, tracked_objects, ground_truth, vis_dir)
        print(f"Сохранена визуализация: {vis_path}")
        
        # Записываем время обработки
        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time
        frame_times.append(frame_time)
    
    # Рассчитываем метрики
    metrics = calculate_metrics(track_info, ground_truth, val_images)
    
    # Выводим результаты
    print("\nРезультаты отслеживания:")
    print(f"Среднее время обработки кадра: {total_time/len(val_images):.4f} секунд")
    print(f"MT (Mostly Tracked): {metrics['mt']}")
    print(f"ML (Mostly Lost): {metrics['ml']}")
    print(f"MOTA: {metrics['mota']:.4f}")
    print(f"MOTP: {metrics['motp']:.4f}")
    print(f"Всего объектов: {metrics['total_objects']}")
    print(f"Всего треков: {metrics['total_tracks']}")
    
    # Сохранение результатов в файл
    results_file = 'validation_results_byte_track.txt'
    with open(results_file, 'w') as f:
        f.write(f"Результаты валидации модели YOLOv8:\n")
        f.write(f"Путь к модели: {model_path}\n")
        f.write(f"Среднее время обработки кадра: {total_time/len(val_images):.4f} секунд\n")
        f.write(f"MT (Mostly Tracked): {metrics['mt']}\n")
        f.write(f"ML (Mostly Lost): {metrics['ml']}\n")
        f.write(f"MOTA: {metrics['mota']:.4f}\n")
        f.write(f"MOTP: {metrics['motp']:.4f}\n")
        f.write(f"Всего объектов: {metrics['total_objects']}\n")
        f.write(f"Всего треков: {metrics['total_tracks']}\n")
        
        # Записываем все доступные метрики YOLO
        f.write("\nМетрики YOLO:\n")
        f.write(f"mAP50: {results.box.map50:.4f}\n")
        f.write(f"mAP50-95: {results.box.map:.4f}\n")
    
    print(f"\nРезультаты сохранены в файл: {results_file}")
    print(f"Визуализации сохранены в директории: {vis_dir}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка модели YOLOv8 с ByteTrack')
    parser.add_argument('--model', type=str, help='Путь к обученной модели (.pt файл)')
    args = parser.parse_args()
    
    evaluate_model(args.model) 
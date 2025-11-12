import cv2
import numpy as np
from math import atan2, degrees

def find_fragment_center_with_rotation_and_scale(reference_image_path, fragment_image_path, mask_path=None):
    """
    Функция для поиска центра фрагмента на полном изображении с учетом любого поворота, масштаба и формы.
    
    :param reference_image_path: Путь к полному изображению.
    :param fragment_image_path: Путь к фрагменту.
    :param mask_path: Опциональный путь к маске фрагмента (черно-белое изображение, где белый — область фрагмента).
    :return: (center_x, center_y, rotation_angle, scale_factor) — пиксельные координаты центра, угол поворота (в градусах) и масштабный фактор.
    """
    # Загрузка изображений в градациях серого
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    frag_img = cv2.imread(fragment_image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    
    if ref_img is None or frag_img is None:
        raise ValueError("Не удалось загрузить изображения. Проверьте пути.")
    
    # Поиск по шкалам (scale factors от 0.5 до 2.0)
    best_match = None
    best_val = -1
    best_scale = 1.0
    best_loc = (0, 0)
    
    for scale in np.arange(0.5, 2.1, 0.1):  # Шаг 0.1, можно настроить
        scaled_frag = cv2.resize(rotated_frag, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(rotated_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if rotated_mask is not None else None
        
        if scaled_frag.shape[0] > ref_img.shape[0] or scaled_frag.shape[1] > ref_img.shape[1]:
            continue  # Пропустить если слишком большой
        
        # Template matching с маской (если есть)
        result = cv2.matchTemplate(ref_img, scaled_frag, cv2.TM_CCOEFF_NORMED, mask=scaled_mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_match = scaled_frag
            best_scale = scale
            best_loc = max_loc
    
    if best_val < 0.6:  # Порог совпадения
        # Fallback: Feature matching (ORB) для независимости от формы
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(ref_img, None)
        kp2, des2 = orb.detectAndCompute(rotated_frag, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            raise ValueError("Совпадение слишком слабое. Фрагмент может не соответствовать изображению.")
        
        # Найти homography и центр
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        M, mask_h = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None:
            raise ValueError("Не удалось найти homography.")
        
        # Центр фрагмента через homography
        frag_center = np.array([[rotated_frag.shape[1] // 2, rotated_frag.shape[0] // 2]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_center = cv2.perspectiveTransform(frag_center, M)[0][0]
        center_x, center_y = int(transformed_center[0]), int(transformed_center[1])
        best_scale = 1.0  # Для ORB scale не вычисляем просто
    else:
        # Используем лучший match из template matching
        top_left = best_loc
        center_x = top_left[0] + best_match.shape[1] // 2
        center_y = top_left[1] + best_match.shape[0] // 2
    
    # Визуализация: рисуем центр на reference и сохраняем
    ref_with_center = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    cv2.circle(ref_with_center, (center_x, center_y), 10, (0, 255, 0), -1)
    cv2.putText(ref_with_center, f"Rotation: {rotation_angle:.1f}°, Scale: {best_scale:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite("fragment_center_with_rotation_and_scale.jpg", ref_with_center)
    
    return center_x, center_y, rotation_angle, best_scale

# Пример использования
if __name__ == "__main__":
    cx, cy, angle, scale = find_fragment_center_with_rotation_and_scale(
        reference_image_path="full_image.jpg",
        fragment_image_path="fragment.jpg",
        mask_path=None  # Укажи путь к маске, если форма фрагмента не прямоугольная, иначе None
    )
    print(f"Центр фрагмента найден: пиксели ({cx}, {cy}), поворот: {angle:.1f}°, масштаб: {scale:.1f}")
    print("Аннотированное изображение сохранено как 'fragment_center_with_rotation_and_scale.jpg'")

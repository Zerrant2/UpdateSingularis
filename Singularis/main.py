import cv2
import numpy as np
from tqdm import tqdm

def auto_perspective_points(frame, frame_height, frame_width):
    """
    Автоматически определяет точки перспективного преобразования на основе линий разметки.
    """
    # Преобразуем кадр в оттенки серого и применяем размытие
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Детекция краев с помощью Canny
    edges = cv2.Canny(blur, 50, 150)

    # Маска для нужной зоны
    mask = np.zeros_like(edges)
    corners = np.array([[
        (0, frame_height),
        (frame_width * 0.1, frame_height * 0.65),
        (frame_width * 0.9, frame_height * 0.65),
        (frame_width, frame_height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, corners, 255)
    masked = cv2.bitwise_and(edges, mask)
    #cv2.imshow("mask",masked)
    #cv2.waitKey()

    # Детекция линий с помощью Hough Transform
    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        raise ValueError("Не удалось обнаружить линии разметки.")

    # Фильтрация линий по углу наклона (предполагаем, что разметка имеет наклон от 10 до 60 градусов)
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if 10 < abs(angle) < 60:  # Фильтруем линии по углу
            cv2.line(gray, (x1, y1), (x2, y2), (0, 255, 0), 2)
            filtered_lines.append(line[0])
    if len(filtered_lines) < 2:
        raise ValueError("Недостаточно линий для определения перспективы.")

    # Находим точки пересечения линий для определения точки схода
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return px, py

    # Собираем точки пересечения
    vanishing_points = []
    for i in range(len(filtered_lines)):
        for j in range(i + 1, len(filtered_lines)):
            intersection = line_intersection(filtered_lines[i], filtered_lines[j])
            if intersection and 0 < intersection[0] < frame_width and 0 < intersection[1] < frame_height:
                vanishing_points.append(intersection)

    if not vanishing_points:
        raise ValueError("Не удалось определить точку схода.")

    # Вычисляем среднюю точку схода
    vanishing_point = np.mean(vanishing_points, axis=0)
    x, y = map(int, vanishing_point)
    cv2.circle(gray, (x, y), 5, (0, 0, 255), -1)  # Красные точки


    # Определяем трапециевидную область дороги
    # Верхние точки (ближе к точке схода) и нижние точки (ближе к нижней границе кадра)
    src_points = np.float32([
        [vanishing_point[0] - 550, vanishing_point[1] + 200],  # Левый верхний
        [vanishing_point[0] + 220, vanishing_point[1] + 200],  # Правый верхний
        [50, frame_height - 50],  # Левый нижний
        [frame_width - 620, frame_height - 50]  # Правый нижний
    ])

    return src_points

def compute_offset(prev_frame, curr_frame):
    """
    Вычисляем вертикальное смещение.
    """

    # Преобразуем кадры в оттенки серого для фазовой корреляции
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Вычисляем смещение с помощью фазовой корреляции
    shift, _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(curr_gray))

    # Возвращаем только вертикальное смещение (по оси Y)
    return shift[1]

def process_video(video_path, output_path, dst_width, dst_height, src_points=None):
    """
    Основная функция обработки видеопотока.
    """

    dst_points = np.float32([
        [0, 0],
        [dst_width, 0],
        [0, dst_height],
        [dst_width, dst_height]
    ])

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеофайл.")
        return

    # Получаем свойства видео
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Если src_points не переданы, определяем их автоматически
    if src_points is None:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось прочитать первый кадр.")
            cap.release()
            return
        src_points = auto_perspective_points(frame, frame_height, frame_width)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Определяем перспективное преобразование
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Создаём выходное полотно
    canvas_height = frame_count * dst_height // 2
    canvas = np.zeros((canvas_height, dst_width, 3), dtype=np.uint8)

    #Первоначальное смещение
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.")
        cap.release()
        return

    prev_warped = cv2.warpPerspective(frame, M, (dst_width, dst_height))

    canvas[canvas_height-dst_height:canvas_height, :, :] = prev_warped
    # Обрабатываем каждый кадр

    canvas_y = canvas_height - dst_height

    for _ in tqdm(range(frame_count)):

        ret, frame = cap.read()
        if not ret:
            break

        # Применяем перспективное преобразование
        curr_warped = cv2.warpPerspective(frame, M, (dst_width, dst_height))

        # Считаем смещение
        pixel_offset = int(compute_offset(prev_warped,curr_warped))

        if pixel_offset > 0:
            new_part = curr_warped[:pixel_offset, :]
            canvas_y -= pixel_offset
            canvas[canvas_y:canvas_y+pixel_offset,:,:] = new_part
            #cv2.imshow("part",new_part)
            #cv2.waitKey()
        prev_warped = curr_warped

    # Обрезаем ненужный фон
    canvas = canvas[canvas_y:canvas_height, :, :]

    # Сохраняем результат
    cv2.imwrite(output_path, canvas)
    print(f"Вывод сохранен в {output_path}")

    # Освобождаем ресурсы
    cap.release()
def main():
    # Настройки генерации текстуры
    dst_width = 500  # Настройте в зависимости от желаемого масштаба вывода
    dst_height = 150  # Высота одного кадра в виде сверху

    #Ручная установка точек перспективы, для использования передать в process_video вместо None
    #Точки перспективы
    src_points = np.float32([
        [530, 850],  #левый-верхний
        [1080, 850], #правый-верхний
        [50, 1000],  #левый-нижний
        [1210, 1000]  #правый-нижний
    ])

    # Пример использования
    video_path = "video/название_видео.mp4"  # Путь к видео
    output_path = "road_texture.png"
    process_video(video_path, output_path, dst_width,dst_height,None)


if __name__ == "__main__":
    main()


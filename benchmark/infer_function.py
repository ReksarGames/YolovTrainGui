import cv2
import numpy as np

try:
    import numba as nb

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
if NUMBA_AVAILABLE:

    @nb.njit(parallel=True, fastmath=True)
    def numba_convert_new_array(src):
        """Numba优化转换,返回新数组"""
        dst = np.empty_like(src, dtype=np.float32)
        for i in nb.prange(src.shape[0]):
            for j in nb.prange(src.shape[1]):
                for k in range(src.shape[2]):
                    dst[i, j, k] = src[i, j, k] * 0.00392156862745098
        return dst

    @nb.njit(parallel=True, fastmath=True)
    def numba_resize_and_normalize(src, target_h, target_w):
        """Numba优化的resize和归一化"""
        dst = np.empty((target_h, target_w, 3), dtype=np.float32)
        scale_h = src.shape[0] / target_h
        scale_w = src.shape[1] / target_w
        for i in nb.prange(target_h):
            for j in nb.prange(target_w):
                src_i = int(i * scale_h)
                src_j = int(j * scale_w)
                if src_i >= src.shape[0]:
                    src_i = src.shape[0] - 1
                if src_j >= src.shape[1]:
                    src_j = src.shape[1] - 1
                dst[i, j, 0] = src[src_i, src_j, 2] * 0.00392156862745098
                dst[i, j, 1] = src[src_i, src_j, 1] * 0.00392156862745098
                dst[i, j, 2] = src[src_i, src_j, 0] * 0.00392156862745098
        return dst


def draw_boxes(image, boxes, scores, classes):
    for box, score, classe in zip(boxes, scores, classes):
        box = box[:4]
        class_id = np.argmax(classe)
        c_x, c_y, w, h = box.astype(np.int32)
        x_min, y_min, x_max, y_max = convert_box_coordinates(c_x, c_y, w, h)
        color = get_color(class_id)
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        text = f"{class_id} {score:.2f}"
        cv2.putText(
            image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness
        )
    return image


def draw_boxes_v8(image, boxes, scores, classes, class_names=None):
    """
    在图像上绘制检测框
    Args:
        image: 原始图像
        boxes: 边界框坐标 [x1, y1, x2, y2]
        scores: 置信度分数
        classes: 类别ID
        class_names: 类别名称列表(可选)
    Returns:
        image: 绘制了检测框的图像
    """
    image = image.copy()
    height, width = image.shape[:2]
    for box, score, class_id in zip(boxes, scores, classes):
        class_id = int(class_id)
        box = box[:4]
        c_x, c_y, w, h = box.astype(np.int32)
        x1, y1, x2, y2 = convert_box_coordinates(c_x, c_y, w, h)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        color = get_color(class_id)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text = f"{class_id} {score:.2f}"
        cv2.putText(
            image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness
        )
    return image


def get_color(class_id):
    """
    为每个类别生成唯一且易于区分的颜色
    """
    predefined_colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 255, 128),
        (255, 0, 128),
    ]
    if class_id < len(predefined_colors):
        return predefined_colors[class_id]
    np.random.seed(class_id)
    color = tuple(map(int, np.random.randint(0, 255, size=3)))
    return color


def draw_fps(image, fps):
    text = f"FPS: {fps:.2f}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    return image


def convert_box_coordinates(center_x, center_y, width, height):
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)
    return (x_min, y_min, x_max, y_max)


def convert_box_coordinates_float(center_x, center_y, width, height):
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    return (x_min, y_min, x_max, y_max)


def postprocess(pred, conf_thres=0.3, iou_thres=0.45, num_classes=1):
    """
    Universal YOLO postprocess - handles all formats automatically.

    Formats (auto-detected):
        - [N, 6]: x1, y1, x2, y2, conf, cls_id (V10, converted V5)
        - [C, N]: cx, cy, w, h, class_scores... (V8)

    Args:
        pred: Model output
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        num_classes: Number of classes (for V8 format)

    Returns:
        boxes: [cx, cy, w, h] format
        scores: Confidence scores
        classes: Class IDs
    """
    pred_sq = np.squeeze(pred)

    if pred_sq.ndim != 2:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

    rows, cols = pred_sq.shape
    detections = []  # list of (x1, y1, x2, y2, conf, cls)

    if cols == 6:
        # Detect format: xyxy (x2>x1, y2>y1) vs cxcywh (w,h smaller than cx,cy)
        top_idx = np.argmax(pred_sq[:, 4])
        c0, c1, c2, c3 = pred_sq[top_idx, :4]
        is_cxcywh = (c2 < c0) or (c3 < c1)

        conf = pred_sq[:, 4]
        mask = conf > conf_thres
        if mask.any():
            filtered = pred_sq[mask]
            if is_cxcywh:
                # Format [N, 6] = cx, cy, w, h, conf, cls
                cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
                half_w, half_h = w / 2, h / 2
                x1, y1 = cx - half_w, cy - half_h
                x2, y2 = cx + half_w, cy + half_h
            else:
                # Format [N, 6] = x1, y1, x2, y2, conf, cls_id (V10)
                x1, y1, x2, y2 = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            conf_f = filtered[:, 4]
            cls_f = filtered[:, 5].astype(int)
            detections = np.column_stack([x1, y1, x2, y2, conf_f, cls_f])
    else:
        # Format [C, N] transposed - V8 style (vectorized)
        cx = pred_sq[0, :]
        cy = pred_sq[1, :]
        w = pred_sq[2, :]
        h = pred_sq[3, :]

        # Get max class scores vectorized
        class_scores = pred_sq[4:4+num_classes, :]  # [num_classes, N]
        max_scores = np.max(class_scores, axis=0)   # [N]
        max_cls = np.argmax(class_scores, axis=0)   # [N]

        # Filter by confidence
        mask = max_scores > conf_thres
        if mask.any():
            cx_f, cy_f, w_f, h_f = cx[mask], cy[mask], w[mask], h[mask]
            scores_f, cls_f = max_scores[mask], max_cls[mask]

            # Convert cxcywh to xyxy
            half_w, half_h = w_f * 0.5, h_f * 0.5
            x1 = cx_f - half_w
            y1 = cy_f - half_h
            x2 = cx_f + half_w
            y2 = cy_f + half_h
            detections = np.column_stack([x1, y1, x2, y2, scores_f, cls_f])

    if isinstance(detections, list):
        if not detections:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
        detections = np.array(detections)
    elif not isinstance(detections, np.ndarray) or len(detections) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

    # NMS (vectorized with adaptive threshold for small targets)
    if iou_thres > 0 and len(detections) > 1:
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        areas = (x2 - x1) * (y2 - y1)

        # Adaptive NMS: softer threshold for small targets
        median_area = np.median(areas)
        small_threshold = median_area * 0.5

        order = np.argsort(scores)  # ascending, we pop from end
        keep = []

        while order.size > 0:
            i = order[-1]
            keep.append(i)

            if order.size == 1:
                break

            # Adaptive IoU threshold based on current box size
            if areas[i] < small_threshold:
                current_iou_thres = min(iou_thres * 1.5, 0.8)
            else:
                current_iou_thres = iou_thres

            # Vectorized IoU with remaining
            remaining = order[:-1]
            xx1 = np.maximum(x1[i], x1[remaining])
            yy1 = np.maximum(y1[i], y1[remaining])
            xx2 = np.minimum(x2[i], x2[remaining])
            yy2 = np.minimum(y2[i], y2[remaining])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            union = areas[i] + areas[remaining] - intersection
            iou = intersection / union

            # Keep only boxes with IoU <= threshold
            order = remaining[iou <= current_iou_thres]

        detections = detections[keep]

    # Convert xyxy to cxcywh for output
    x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    boxes = np.stack([cx, cy, w, h], axis=1)
    scores = detections[:, 4]
    classes = detections[:, 5].astype(int)

    return boxes, scores, classes


def read_img(img_data, size=(320, 320)):
    target_w, target_h = size
    # if NUMBA_AVAILABLE and target_w == 640 and (target_h == 640):
    #     try:
    #         resized_img = cv2.resize(img_data, (target_w, target_h))
    #         normalized_img = numba_convert_new_array(resized_img)
    #         processed_img = normalized_img[:, :, [2, 1, 0]]
    #         blob = np.transpose(processed_img, (2, 0, 1))
    #         blob = np.expand_dims(blob, axis=0)
    #         return blob
    #     except Exception as e:
    #         pass
    blob = cv2.dnn.blobFromImage(
        image=img_data,
        scalefactor=0.00392156862745098,
        size=(target_w, target_h),
        mean=(0.0, 0.0, 0.0),
        swapRB=True,
        crop=False,
    )
    return blob

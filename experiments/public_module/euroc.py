from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .schedule import frame_pair_start_indices


class EuRoCDataError(RuntimeError):
    """EuRoC 文件、标定或图像内容无效。"""


@dataclass(frozen=True)
class CameraCalibration:
    camera: np.ndarray
    distortion: np.ndarray
    body_from_sensor: np.ndarray
    resolution: tuple[int, int]


def read_camera_calibration(path: Path) -> CameraCalibration:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
        distortion = np.asarray(payload["distortion_coefficients"], dtype=np.float64)
        resolution_values = np.asarray(payload["resolution"], dtype=int)
        transform_node = payload["T_BS"]
        transform = np.asarray(transform_node["data"], dtype=np.float64).reshape(
            int(transform_node["rows"]), int(transform_node["cols"])
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise EuRoCDataError(f"无法解析标定文件：{path}") from exc
    if intrinsics.size != 4 or distortion.size not in (4, 5):
        raise EuRoCDataError(f"相机参数维度不正确：{path}")
    if resolution_values.size != 2 or transform is None or transform.shape != (4, 4):
        raise EuRoCDataError(f"分辨率或 T_BS 无效：{path}")
    fx, fy, cx, cy = intrinsics
    camera = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraCalibration(
        camera=camera,
        distortion=distortion,
        body_from_sensor=np.asarray(transform, dtype=np.float64),
        resolution=(int(resolution_values[0]), int(resolution_values[1])),
    )


@dataclass
class StereoRectifier:
    map0_x: np.ndarray
    map0_y: np.ndarray
    map1_x: np.ndarray
    map1_y: np.ndarray
    rectified_camera: np.ndarray
    reprojection: np.ndarray
    matcher: cv2.StereoSGBM

    @classmethod
    def from_sequence(cls, sequence_root: Path, stereo_config: dict[str, Any]) -> "StereoRectifier":
        cam0 = read_camera_calibration(sequence_root / "mav0" / "cam0" / "sensor.yaml")
        cam1 = read_camera_calibration(sequence_root / "mav0" / "cam1" / "sensor.yaml")
        if cam0.resolution != cam1.resolution:
            raise EuRoCDataError("双目相机分辨率不一致")

        sensor1_from_sensor0 = np.linalg.inv(cam1.body_from_sensor) @ cam0.body_from_sensor
        rotation = sensor1_from_sensor0[:3, :3]
        translation = sensor1_from_sensor0[:3, 3]
        if not np.isfinite(np.linalg.det(rotation)) or abs(np.linalg.det(rotation) - 1.0) > 1e-5:
            raise EuRoCDataError("双目外参旋转无效")
        baseline = float(np.linalg.norm(translation))
        if not 0.05 <= baseline <= 0.2:
            raise EuRoCDataError(f"双目基线不合理：{baseline}")

        image_size = cam0.resolution
        rect0, rect1, projection0, projection1, reprojection, _, _ = cv2.stereoRectify(
            cam0.camera,
            cam0.distortion,
            cam1.camera,
            cam1.distortion,
            image_size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )
        map0_x, map0_y = cv2.initUndistortRectifyMap(
            cam0.camera,
            cam0.distortion,
            rect0,
            projection0[:, :3],
            image_size,
            cv2.CV_32FC1,
        )
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            cam1.camera,
            cam1.distortion,
            rect1,
            projection1[:, :3],
            image_size,
            cv2.CV_32FC1,
        )
        block_size = int(stereo_config["block_size"])
        matcher = cv2.StereoSGBM_create(
            minDisparity=int(stereo_config["min_disparity"]),
            numDisparities=int(stereo_config["num_disparities"]),
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            preFilterCap=31,
            uniquenessRatio=int(stereo_config["uniqueness_ratio"]),
            speckleWindowSize=int(stereo_config["speckle_window_size"]),
            speckleRange=int(stereo_config["speckle_range"]),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        return cls(
            map0_x,
            map0_y,
            map1_x,
            map1_y,
            np.asarray(projection0[:, :3], dtype=np.float64),
            np.asarray(reprojection, dtype=np.float64),
            matcher,
        )

    def rectify(self, image0: np.ndarray, image1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = cv2.remap(image0, self.map0_x, self.map0_y, cv2.INTER_LINEAR)
        right = cv2.remap(image1, self.map1_x, self.map1_y, cv2.INTER_LINEAR)
        return left, right


@dataclass(frozen=True)
class ImageRecord:
    timestamp_ns: int
    filename: str


def read_image_index(sequence_root: Path) -> list[ImageRecord]:
    csv_path = sequence_root / "mav0" / "cam0" / "data.csv"
    if not csv_path.is_file():
        raise EuRoCDataError(f"图像索引不存在：{csv_path}")
    records: list[ImageRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(line for line in stream if not line.startswith("#")):
            if len(row) < 2:
                continue
            records.append(ImageRecord(int(row[0]), row[1].strip()))
    if len(records) < 300:
        raise EuRoCDataError("图像索引数量异常")
    timestamps = np.asarray([record.timestamp_ns for record in records], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise EuRoCDataError("图像时间戳不是严格递增")
    return records


def locate_sequence(dataset_root: Path, sequence: str) -> Path:
    direct = dataset_root / sequence
    if (direct / "mav0" / "cam0" / "data.csv").is_file():
        return direct
    matches = [
        candidate.parent.parent.parent
        for candidate in dataset_root.rglob("cam0/data.csv")
        if candidate.parent.parent.parent.name == sequence
    ]
    unique = sorted(set(path.resolve() for path in matches))
    if len(unique) != 1:
        raise EuRoCDataError(f"序列定位结果不是唯一值：{sequence} -> {unique}")
    return unique[0]


@dataclass(frozen=True)
class TrackSet:
    points_3d: np.ndarray
    target_points_2d: np.ndarray
    detected_count: int
    valid_disparity_count: int
    tracked_count: int


def extract_tracks(
    sequence_root: Path,
    records: list[ImageRecord],
    start_index: int,
    temporal_gap: int,
    rectifier: StereoRectifier,
    feature_config: dict[str, Any],
    stereo_config: dict[str, Any],
) -> TrackSet:
    first = records[start_index]
    second = records[start_index + temporal_gap]
    left0_raw = cv2.imread(str(sequence_root / "mav0" / "cam0" / "data" / first.filename), cv2.IMREAD_GRAYSCALE)
    right0_raw = cv2.imread(str(sequence_root / "mav0" / "cam1" / "data" / first.filename), cv2.IMREAD_GRAYSCALE)
    left1_raw = cv2.imread(str(sequence_root / "mav0" / "cam0" / "data" / second.filename), cv2.IMREAD_GRAYSCALE)
    if left0_raw is None or right0_raw is None or left1_raw is None:
        raise EuRoCDataError("帧对图像缺失或不可读取")

    left0, right0 = rectifier.rectify(left0_raw, right0_raw)
    left1 = cv2.remap(left1_raw, rectifier.map0_x, rectifier.map0_y, cv2.INTER_LINEAR)
    corners = cv2.goodFeaturesToTrack(
        left0,
        maxCorners=int(feature_config["max_corners"]),
        qualityLevel=float(feature_config["quality_level"]),
        minDistance=float(feature_config["min_distance_px"]),
        blockSize=int(feature_config["block_size"]),
        useHarrisDetector=False,
    )
    if corners is None:
        return TrackSet(np.empty((0, 3)), np.empty((0, 2)), 0, 0, 0)
    points0 = corners.reshape(-1, 2).astype(np.float32)
    detected_count = len(points0)

    disparity = rectifier.matcher.compute(left0, right0).astype(np.float32) / 16.0
    rounded = np.rint(points0).astype(int)
    height, width = left0.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    sampled_disparity = np.full(len(points0), np.nan, dtype=np.float64)
    sampled_disparity[inside] = disparity[rounded[inside, 1], rounded[inside, 0]]
    disparity_limits = stereo_config["valid_disparity_open_interval_px"]
    disparity_valid = (
        inside
        & (sampled_disparity > float(disparity_limits[0]))
        & (sampled_disparity < float(disparity_limits[1]))
    )
    valid_disparity_count = int(np.sum(disparity_valid))

    lk_parameters = {
        "winSize": (int(feature_config["lk_window_px"]), int(feature_config["lk_window_px"])),
        "maxLevel": int(feature_config["lk_max_level"]),
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    tracked1, status_forward, _ = cv2.calcOpticalFlowPyrLK(left0, left1, points0, None, **lk_parameters)
    if tracked1 is None:
        return TrackSet(np.empty((0, 3)), np.empty((0, 2)), detected_count, valid_disparity_count, 0)
    tracked0, status_backward, _ = cv2.calcOpticalFlowPyrLK(left1, left0, tracked1, None, **lk_parameters)
    if tracked0 is None:
        return TrackSet(np.empty((0, 3)), np.empty((0, 2)), detected_count, valid_disparity_count, 0)
    forward_ok = status_forward.reshape(-1).astype(bool)
    backward_ok = status_backward.reshape(-1).astype(bool)
    forward_backward = np.linalg.norm(tracked0 - points0, axis=1)
    border = float(feature_config["image_border_px"])
    target_inside = (
        (tracked1[:, 0] >= border)
        & (tracked1[:, 0] < width - border)
        & (tracked1[:, 1] >= border)
        & (tracked1[:, 1] < height - border)
    )
    tracked_valid = (
        disparity_valid
        & forward_ok
        & backward_ok
        & np.isfinite(forward_backward)
        & (forward_backward <= float(feature_config["forward_backward_max_px"]))
        & target_inside
    )

    point_rows = points0[tracked_valid]
    target_rows = tracked1[tracked_valid].astype(np.float64)
    disparity_rows = sampled_disparity[tracked_valid]
    homogeneous = np.column_stack(
        [point_rows[:, 0], point_rows[:, 1], disparity_rows, np.ones(len(point_rows))]
    )
    reprojected = (rectifier.reprojection @ homogeneous.T).T
    points_3d = reprojected[:, :3] / reprojected[:, 3:4]
    depth_limits = stereo_config["valid_depth_closed_interval_m"]
    geometry_valid = (
        np.all(np.isfinite(points_3d), axis=1)
        & (points_3d[:, 2] >= float(depth_limits[0]))
        & (points_3d[:, 2] <= float(depth_limits[1]))
    )
    points_3d = points_3d[geometry_valid]
    target_rows = target_rows[geometry_valid]
    return TrackSet(
        points_3d=np.asarray(points_3d, dtype=np.float64),
        target_points_2d=np.asarray(target_rows, dtype=np.float64),
        detected_count=detected_count,
        valid_disparity_count=valid_disparity_count,
        tracked_count=len(points_3d),
    )

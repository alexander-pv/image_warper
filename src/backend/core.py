import dataclasses

import io
import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial import distance_matrix as dist_measure

"""
The implementation was based on: 
https://stackoverflow.com/questions/71443071/opencv-transform-image-shape-transformation-into-a-given-contour
"""


@dataclasses.dataclass
class ContourArea:
    left: np.ndarray
    bottom: np.ndarray
    right: np.ndarray
    top: np.ndarray


class ImgTransformer:

    def __init__(self, verbose: bool = False):
        """
        :param verbose:
        """
        self.verbose = verbose
        self.white_color = (255, 255, 255)
        self.bbox_sides = ('top', 'bottom', 'left', 'right')

    @staticmethod
    def _lsa_contour_sort(contour1: np.ndarray, contour2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param contour1:
        :param contour2:
        :return:
        """
        dist_mat = dist_measure(contour1, contour2)
        _, cntr2_order = lsa(dist_mat)
        contour2 = contour2[cntr2_order, :]
        return contour1, contour2

    @staticmethod
    def _find_closest_point(contour: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        :param contour:
        :param point:
        :return:
        """
        cntr_pnts = contour[:, 0, :]
        min_dst_idx = np.argmin(np.sqrt(np.sum((cntr_pnts - point) ** 2, axis=1)))
        closest_point = cntr_pnts[min_dst_idx, :]
        return closest_point

    @staticmethod
    def _combine_contour_areas(areas: ContourArea) -> np.ndarray:
        """
        :param areas:
        :return:
        """
        return np.concatenate([areas.top, areas.right, areas.bottom, areas.left])

    def _sample_points(self, points: np.ndarray, num: int) -> np.ndarray:
        """
        :param points:
        :param num:
        :return:
        """
        assert len(points.shape) == 2
        sampled_ids = np.linspace(0, points.shape[0] - 1, num, dtype=int)
        sampled_points = points[sampled_ids, :]
        assert sampled_points.shape[0] == num
        return sampled_points

    def _sample_areas(self, areas: ContourArea, num: int) -> ContourArea:
        """
        :param areas:
        :param num:
        :return:
        """
        for side in self.bbox_sides:
            areas.__dict__[side] = self._sample_points(areas.__dict__[side], num)
        return areas

    def _extract_sampled_contour_points(self, img: np.ndarray, sample_size: int) -> np.ndarray:
        """
        :param img:
        :param sample_size:
        :return:
        """
        mask = get_mask(img, verbose=self.verbose)
        contour = get_contour(mask)[:, 0, :]
        sampled_contour = self._sample_points(contour, sample_size)
        if self.verbose:
            zero_img = np.zeros(shape=img.shape[:2], dtype=img.dtype)
            cv2.drawContours(zero_img, [contour], 0, self.white_color, 1)
            show_img(zero_img, 'contour')
        return sampled_contour

    def _match_contour_bbox_vertices(self, contour: np.ndarray) -> list:
        """
        :param contour:
        :return:
        """
        x, y, w, h = cv2.boundingRect(contour)
        top_left = np.array([x, y])
        top_right = np.array([x + w, y])
        bot_left = np.array([x, y + h])
        bot_right = np.array([x + w, y + h])
        pnts = [top_left, top_right, bot_left, bot_right]
        return [self._find_closest_point(contour, pnt) for pnt in pnts]

    def _split_contour_to_areas(self, contour: np.ndarray) -> ContourArea:
        """
        Find contour points matched to normalized bbox and search for the contour array split.
        Searching is carried out with the following matched points order:
            top_left     -> bottom_left
            bottom_left  -> bottom_right
            bottom_right -> top_right
        :param contour: np.ndarray
        :return: ContourArea
        """

        top_left, top_right, bot_left, bot_right = self._match_contour_bbox_vertices(contour)
        cntr_pnts = contour[:, 0, :]
        idx0 = int(np.where((cntr_pnts[:, 0] == top_left[0]) & (cntr_pnts[:, 1] == top_left[1]))[0])
        idx1 = int(np.where((cntr_pnts[:, 0] == bot_left[0]) & (cntr_pnts[:, 1] == bot_left[1]))[0])
        left_area = cntr_pnts[idx0: idx1 + 1, :]
        idx2 = int(np.where((cntr_pnts[:, 0] == bot_right[0]) & (cntr_pnts[:, 1] == bot_right[1]))[0])
        bottom_area = cntr_pnts[idx1 + 1: idx2 + 1, :]
        idx3 = int(np.where((cntr_pnts[:, 0] == top_right[0]) & (cntr_pnts[:, 1] == top_right[1]))[0])
        right_arera = cntr_pnts[idx2 + 1: idx3 + 1, :]
        top_area = np.concatenate([cntr_pnts[idx3 + 1:, :], cntr_pnts[:idx0, :]])
        area = ContourArea(left_area, bottom_area, right_arera, top_area)
        return area

    def _extract_sampled_contour_areas(self, img: np.ndarray, sample_size: int) -> ContourArea:
        mask = get_mask(img, verbose=self.verbose)
        contour = get_contour(mask)
        areas = self._split_contour_to_areas(contour)
        areas = self._sample_areas(areas, sample_size)
        return areas

    def contour_points_sampling(self, primary_img: np.ndarray,
                                secondary_img: np.ndarray, sample_size: int = 150) -> np.ndarray:
        """
        Transform image shape into a contour with following steps:
            1. Get images of the equal size
            2. Extract object mask
            3. Extract object contour
            4. Sample contour for each image
            5. Calculate distance matrix & perform linear sum alignment to get the proper points pairs
            6. Split images to triangles and
               perform affine transformations and warping to fit triangles to the primary image.
        :param primary_img:
        :param secondary_img:
        :param sample_size:
        :return:
        """
        primary_img = cv2.resize(primary_img, secondary_img.shape[:2])
        contour1 = self._extract_sampled_contour_points(primary_img, sample_size)
        contour2 = self._extract_sampled_contour_points(secondary_img, sample_size)
        contour1, contour2 = self._lsa_contour_sort(contour1, contour2)
        warped_img = warp(secondary_img, primary_img, contour2, contour1)
        return warped_img

    def contour_areas_stratification(self, primary_img: np.ndarray,
                                     secondary_img: np.ndarray, sample_size: int = 150) -> np.ndarray:
        """
        Transform image shape into a contour with following steps:
            1. Get images of the equal size
            2. Extract object mask
            3. Extract object contour
            4. Extract bbox of a contour
            4. Find closest contour points to bbox vertices
            5. Split contour points based on closest vertices to areas: top, bottom, left, right. Combine sorted areas.
            6. Split images to triangles and
               perform affine transformations and warping to fit triangles to the primary image.
        :param primary_img:
        :param secondary_img:
        :param sample_size:
        :return:
        """
        primary_img = cv2.resize(primary_img, secondary_img.shape[:2])
        area1 = self._extract_sampled_contour_areas(primary_img, sample_size)
        area2 = self._extract_sampled_contour_areas(secondary_img, sample_size)
        contour1, contour2 = self._combine_contour_areas(area1), self._combine_contour_areas(area2)
        warped_img = warp(secondary_img, primary_img, contour2, contour1)
        return warped_img


def dry_run(img_path: str, method: str = 'cps', encode: bool = False) -> np.ndarray or io.BytesIO:
    transformer = ImgTransformer()
    primary_image = cv2.imread(os.path.join(img_path, 'bulb.png'), cv2.IMREAD_UNCHANGED)
    secondary_image = cv2.imread(os.path.join(img_path, 'fox.png'), cv2.IMREAD_UNCHANGED)
    if method == 'cps':
        output = transformer.contour_points_sampling(primary_image, secondary_image)
    elif method == 'cas':
        output = transformer.contour_areas_stratification(primary_image, secondary_image)
    else:
        raise ValueError(f'Unknown warping method: {method}')

    if encode:
        io_buf = encode_img(output)
        return io_buf
    else:
        return output


def get_grayscale(img: np.ndarray) -> np.ndarray:
    """
    :param img:
    :return:
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_mask(img: np.ndarray, min_val: int = 1, max_val: int = 255, verbose: bool = False) -> np.ndarray:
    """
    :param img:
    :param min_val:
    :param max_val:
    :param verbose:
    :return:
    """
    if img.shape[2] == 4:
        img = img[:, :, 3]
    else:
        img = get_grayscale(img)
    ret, threshold = cv2.threshold(img, min_val, max_val, cv2.THRESH_BINARY)

    if verbose:
        show_img(threshold, 'image_threshold')
    return threshold


def get_contour(img: np.ndarray, approx=cv2.CHAIN_APPROX_NONE) -> np.ndarray:
    """
    :param img:
    :param approx:
    :return:
    """
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, approx)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    largest_cntr = sorted(contours, key=cv2.contourArea)[-1]
    return largest_cntr


def get_corrected_mask(img: np.ndarray, contour: np.ndarray,
                       fill_color: tuple or list = (255, 255, 255), verbose: bool = False):
    """
    :param img:
    :param contour:
    :param fill_color:
    :param verbose:
    :return:
    """
    cntr_mask = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(cntr_mask, [contour], fill_color)
    if verbose:
        show_img(cntr_mask, 'cntr_mask')
    return cntr_mask


def crop(img: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    :param img:
    :param pts:
    :return:
    """
    x, y, w, h = cv2.boundingRect(pts)
    img_cropped = img[y: y + h, x: x + w]
    pts[:, 0] -= x
    pts[:, 1] -= y
    return img_cropped, pts


def triangles(points: np.ndarray) -> list:
    """
    :param points:
    :return:
    """
    points = np.where(points, points, 1)
    subdiv = cv2.Subdiv2D((*points.min(0), *points.max(0)))
    for pt in points:
        subdiv.insert(tuple(map(int, pt)))
    for pts in subdiv.getTriangleList().reshape(-1, 3, 2):
        yield [np.where(np.all(points == pt, 1))[0][0] for pt in pts]


def warp(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    :param img1:
    :param img2:
    :param pts1:
    :param pts2:
    :return:
    """
    img2 = img2.copy()
    for indices in triangles(pts1):
        img1_cropped, triangle1 = crop(img1, pts1[indices])
        img2_cropped, triangle2 = crop(img2, pts2[indices])
        transform = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
        img2_warped = cv2.warpAffine(img1_cropped, transform, img2_cropped.shape[:2][::-1],
                                     None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        mask = np.zeros_like(img2_cropped)
        cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 16, 0)
        img2_cropped *= 1 - mask
        img2_cropped += img2_warped * mask
    return img2


def encode_img(img: np.ndarray) -> io.BytesIO:
    """
    :param img:
    :return:
    """
    is_success, buffer = cv2.imencode(".png", img)
    io_buf = io.BytesIO(buffer)
    return io_buf


def decode_img(buf: bytes) -> np.ndarray:
    """
    :param buf:
    :return:
    """
    array = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    return img


def show_img(img: np.ndarray, name: str, destroy: bool = False) -> None:
    """
    :param img:
    :param name:
    :param destroy:
    :return:
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyWindow(name)

import io
import logging
from abc import abstractmethod, ABCMeta

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from fastapi import Response


class ImgPage(metaclass=ABCMeta):
    def __init__(self, title: str, loglevel: int = logging.DEBUG):
        """
        Abstract class for streamlit pages
        :param title:         str, page title
        """
        self.backend = 'localhost'
        self.title = title
        self.display_width = 300
        self.timeout = 100
        self.loglevel = loglevel
        logging.basicConfig(level=self.loglevel,
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def set_title(self) -> None:
        """
        Set title for a streamlit page
        :return: None
        """
        st.write(f"{self.title}")

    def display_img(self, img, caption):
        st.image(img, caption=caption, width=self.display_width)

    @staticmethod
    def read_img_content(content: str, img_format: str = 'bgra') -> np.ndarray:
        """
        :param content:
        :param img_format:
        :return:
        """
        array = np.fromstring(content, np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if img_format == 'bgra':
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        logging.debug('Done read_img_content')
        return img

    @staticmethod
    def img_to_bytes(img: np.ndarray) -> io.BytesIO:
        """
        :param img:
        :return:
        """
        is_success, buffer = cv2.imencode(".png", img)
        io_buf = io.BytesIO(buffer)
        return io_buf

    @staticmethod
    def hide_style() -> None:
        """
        Hide streamlit style
        :return: None
        """
        hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    @abstractmethod
    def write(self) -> None:
        """
        Write page
        :return:
        """
        return

    def _write(self, img1: np.ndarray, img2: np.ndarray) -> None:
        """
        :return: None
        """
        st.write("Inputs: ")
        self.display_img([img1, img2], ['primary', 'secondary'])
        with st.spinner(text="contour_points_sampling..."):
            res1 = self.post_warp(img1, img2, method='cps')
            warped_img1 = self.read_img_content(res1.content, 'rgba')
        with st.spinner(text="contour_areas_stratification..."):
            res2 = self.post_warp(img1, img2, method='cas')
            warped_img2 = self.read_img_content(res2.content, 'rgba')
        st.write("Outputs: ")
        self.display_img([warped_img1, warped_img2], ['Method: contour_points_sampling (cps)',
                                                      'Method: contour_areas_stratification (cas)'])
        self.hide_style()

    def post_warp(self, img1: np.ndarray, img2: np.ndarray, method: str) -> Response:
        return requests.post(f"http://{self.backend}/warp_imgs/{method}",
                             files={"method": (None, method),
                                    "file1": (f"file1.png;type=image/png", self.img_to_bytes(img1)),
                                    "file2": (f"file2.png;type=image/png", self.img_to_bytes(img2)),

                                    }, timeout=self.timeout)


class TestImgPage(ImgPage):

    def __init__(self, backend: str, *args, **kwargs):
        """
        Streamlit page with test images
        :param backend:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.backend = backend

    def write(self) -> None:
        self.set_title()
        bulb = np.array(Image.open(io.BytesIO(requests.get("https://i.stack.imgur.com/kyB2q.png").content)))
        fox = np.array(Image.open(io.BytesIO(requests.get("https://i.stack.imgur.com/so8PX.png").content)))
        self._write(bulb, fox)


class RandomImgPage(ImgPage):

    def __init__(self, backend: str, *args, **kwargs):
        """
        Streamlit page with random images
        :param backend:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.display_width = 100

    def fetch_random_img(self) -> Response:
        return requests.post(f"http://{self.backend}/fetch_random_img")

    def generate(self):
        with st.spinner(text="Parsing images from Emojipedia..."):
            rnd_res1 = self.fetch_random_img()
            rnd_res2 = self.fetch_random_img()
            img1 = np.array(Image.open(io.BytesIO(rnd_res1.content)))
            img2 = np.array(Image.open(io.BytesIO(rnd_res2.content)))
            self._write(img1, img2)

    def write(self) -> None:
        self.set_title()
        if st.button('Generate random pair'):
            self.generate()


class SelectImgPage(ImgPage):

    def __init__(self, backend: str, *args, **kwargs):
        """
        Streamlit page with random images
        :param backend:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.display_width = 100

    def write(self) -> None:
        self.set_title()
        file1 = st.file_uploader('Primary image...', type=["png"])
        file2 = st.file_uploader('Secondary image...', type=["png"])
        if file1 and file2:
            img1 = np.array(Image.open(file1))
            img2 = np.array(Image.open(file2))
            self._write(img1, img2)

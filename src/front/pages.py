import io
import logging
from abc import abstractmethod, ABCMeta

import os
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from fastapi import Response

from itertools import permutations


class ImgPage(metaclass=ABCMeta):
    def __init__(self, title: str, loglevel: int = logging.DEBUG):
        """
        Abstract class for streamlit pages
        :param title:         str, page title
        """
        self.core_backend = 'localhost:10000'
        self.style_backend = 'localhost:15000'
        self.title = title
        self.display_width = 300
        self.timeout = 999999
        self.loglevel = loglevel
        self.root = os.path.dirname(os.path.abspath(__file__))
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
        with st.spinner(text="contour_areas_stratification...v1"):
            res2 = self.post_warp(img1, img2, method='cas_v1')
            warped_img2 = self.read_img_content(res2.content, 'rgba')
        with st.spinner(text="contour_areas_stratification...v2"):
            res3 = self.post_warp(img1, img2, method='cas_v2')
            warped_img3 = self.read_img_content(res3.content, 'rgba')
        st.write("Outputs: ")
        self.display_img([warped_img1, warped_img2, warped_img3],
                         ['Method: contour_points_sampling (cps)',
                          'Method: contour_areas_stratification (cas_v1)',
                          'Method: contour_areas_stratification (cas_v2)'])
        self.hide_style()

    def post_warp(self, img1: np.ndarray, img2: np.ndarray, method: str) -> Response:
        return requests.post(f"http://{self.core_backend}/warp_imgs/{method}",
                             files={"method": (None, method),
                                    "file1": (f"file1.png;type=image/png", self.img_to_bytes(img1)),
                                    "file2": (f"file2.png;type=image/png", self.img_to_bytes(img2)),

                                    }, timeout=self.timeout)

    def post_stylize(self, content_img: np.ndarray, style_img: np.ndarray, steps: int) -> Response:
        return requests.post(f"http://{self.style_backend}/style_img/{steps}",
                             files={"steps": (None, steps),
                                    "file_content": (
                                    f"file_content.png;type=image/png", self.img_to_bytes(content_img)),
                                    "file_style": (f"file_style.png;type=image/png", self.img_to_bytes(style_img)),

                                    }, timeout=self.timeout)


class TestImgPage(ImgPage):

    def __init__(self, *args, **kwargs):
        """
        Streamlit page with test images
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def write(self) -> None:
        self.set_title()
        bulb = np.array(Image.open(io.BytesIO(requests.get("https://i.stack.imgur.com/kyB2q.png").content)))
        fox = np.array(Image.open(io.BytesIO(requests.get("https://i.stack.imgur.com/so8PX.png").content)))
        self._write(bulb, fox)


class RandomImgPage(ImgPage):

    def __init__(self, *args, **kwargs):
        """
        Streamlit page with random images
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.display_width = 100

    def fetch_random_img(self) -> Response:
        return requests.post(f"http://{self.core_backend}/fetch_random_img")

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

    def __init__(self, *args, **kwargs):
        """
        Streamlit page with selected images
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.display_width = 100

    def write(self) -> None:
        self.set_title()
        file1 = st.file_uploader('Primary image...', type=["png"])
        file2 = st.file_uploader('Secondary image...', type=["png"])
        if file1 and file2:
            img1 = np.array(Image.open(file1))
            img2 = np.array(Image.open(file2))
            self._write(img1, img2)


class TestThreeImgPage(ImgPage):

    def __init__(self, *args, **kwargs):
        """
        Streamlit page with 3 predefined images
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.display_width = 100

    def write(self) -> None:
        self.set_title()
        names = None

        if st.button('Test castle-dragon-fire'):
            names = ('castle.png', 'dragon.png', 'fire.png')
        if st.button('Test joy-duck-pistol'):
            names = ('joy.png', 'duck.png', 'pistol.png')

        if names:
            files = [np.array(Image.open(os.path.join(self.root, '..', 'tests', 'pics', name))) for name in names]
            perms = permutations(files)

            if perms:
                for triple in perms:
                    self._write(triple[0], triple[1])


class SelectThreeImgPage(ImgPage):

    def __init__(self, *args, **kwargs):
        """
        Streamlit page with 3 selected images
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.display_width = 100

    def write(self) -> None:
        self.set_title()

        files = [st.file_uploader(f'Image {i}...', type=["png"]) for i in range(1, 4)]

        if files[0] and files[1] and files[2]:
            files = [np.array(Image.open(f)) for f in files]
            perms = permutations(files)
            if perms:
                for triple in perms:
                    self._write(triple[0], triple[1])


class StyleTransferPage(ImgPage):

    def __init__(self, *args, **kwargs):
        """
        Streamlit page with 3 selected images and style transfer
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.display_width = 100
        self.style_steps = (50, 100, 150)

    def write(self) -> None:
        self.set_title()

        files = [st.file_uploader(f'Image {i}...', type=["png"]) for i in range(1, 4)]

        if files[0] and files[1] and files[2]:
            files = [np.array(Image.open(f)) for f in files]
            perms = permutations(files)
            if perms:
                for triple in perms:
                    self._write(triple[0], triple[1])

                    res_cas_v2 = self.post_warp(triple[0], triple[1], method='cas_v2')
                    warped_img = self.read_img_content(res_cas_v2.content, 'rgba')

                    to_display = []
                    for s in self.style_steps:
                        with st.spinner(text=f"cas_v2 + style transfer with steps: {s}..."):
                            res_stylized = self.post_stylize(warped_img, triple[2], s)
                            stylized_img = self.read_img_content(res_stylized.content, 'rgb')
                            to_display.append(stylized_img)
                    self.display_img(to_display, [f'cas_v2 + style transfer, steps: {s}' for s in self.style_steps])

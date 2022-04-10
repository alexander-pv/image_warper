import io
from abc import abstractmethod, ABCMeta

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from fastapi import Response


class ImgPage(metaclass=ABCMeta):
    def __init__(self, title: str):
        """
        Abstract class for streamlit pages
        :param title:         str, page title
        """
        self.backend = 'localhost'
        self.title = title
        self.display_width = 300

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
        array = np.fromstring(content, np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if img_format == 'bgra':
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img

    @staticmethod
    def img_to_bytes(img: np.ndarray) -> io.BytesIO:
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
        :return: None
        """
        return

    def post_warp(self, img1: np.ndarray, img2: np.ndarray, method: str) -> Response:
        return requests.post(f"http://{self.backend}/warp_imgs/{method}",
                             files={"method": (None, method),
                                    "file1": (f"file1.png;type=image/png", self.img_to_bytes(img1)),
                                    "file2": (f"file2.png;type=image/png", self.img_to_bytes(img2)),

                                    })


class TestImgPage(ImgPage):

    def __init__(self, backend: str, *args, **kwargs):
        """
        Streamlit page with images
        :param backend:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.display_width = 300

    def write(self) -> None:
        self.set_title()
        st.write("Inputs: ")
        bulb = np.array(Image.open(io.BytesIO(requests.get("https://i.stack.imgur.com/kyB2q.png").content)))
        fox = np.array(Image.open(io.BytesIO(requests.get("https://i.stack.imgur.com/so8PX.png").content)))
        self.display_img([bulb, fox], ['bulb', 'fox'])

        res1 = self.post_warp(bulb, fox, method='cps')
        img1 = self.read_img_content(res1.content, 'rgba')
        res2 = self.post_warp(bulb, fox, method='cas')
        img2 = self.read_img_content(res2.content, 'rgba')

        st.write("Outputs: ")
        self.display_img([img1, img2], ['Method: contour_points_sampling (cps)',
                                        'Method: contour_areas_stratification (cas)'])
        self.hide_style()


class RandomImgPage(ImgPage):

    def __init__(self, backend: str, *args, **kwargs):
        """
        Streamlit page with images
        :param backend:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.display_width = 100

    def fetch_random_img(self) -> Response:
        return requests.post(f"http://{self.backend}/fetch_random_img")

    def write(self) -> None:
        self.set_title()

        if st.button('Generate'):
            with st.spinner(text="Parsing images from Emojipedia..."):
                rnd_res1 = self.fetch_random_img()
                rnd_res2 = self.fetch_random_img()
                img1 = np.array(Image.open(io.BytesIO(rnd_res1.content)))
                img2 = np.array(Image.open(io.BytesIO(rnd_res2.content)))

            st.write("Inputs: ")
            self.display_img([img1, img2], ['img1', 'img2'])

            with st.spinner(text="contour_points_sampling..."):
                wrap_res1 = self.post_warp(img1, img2, 'cps')
                warped1 = self.read_img_content(wrap_res1.content, 'rgba')
            with st.spinner(text="contour_areas_stratification..."):
                wrap_res2 = self.post_warp(img1, img2, 'cas')
                warped2 = self.read_img_content(wrap_res2.content, 'rgba').copy()
            st.write("Outputs: ")
            self.display_img([warped1, warped2], ['Method: contour_points_sampling (cps)',
                                                  'Method: contour_areas_stratification (cas)'])
        self.hide_style()

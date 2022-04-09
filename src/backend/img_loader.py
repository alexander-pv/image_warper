import logging
import urllib

import cv2
import emojipedia as emp
import numpy as np


class EmojipediaParser:

    def __init__(self, img_size: int = 120, loglevel: int = logging.DEBUG):
        self.loglevel = loglevel
        self.img_size = img_size
        logging.basicConfig(level=self.loglevel)

    @staticmethod
    def _read_url_img(url: str) -> np.ndarray:
        """
        Read emoji with OpenCV via url
        :param url: str, image address
        :return: np.ndarray
        """
        req = urllib.request.urlopen(url)
        array = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def _fetch_emoji_urls(self, name: str) -> list:
        """
        Prepare all possible emoji urls
        :param name: str, emoji name
        :return: list
        """
        page = emp.Emojipedia._get_emoji_page(name)
        imgs_info = page.find_all('img')
        img_srcs = [x.attrs['src'] for x in imgs_info if int(x.attrs['width']) == self.img_size]
        return img_srcs

    def fetch_by_name(self, name: str) -> np.ndarray or int:
        """
        Fetch random emoji image
        :return: np.ndarray or int
        """
        urls = self._fetch_emoji_urls(name=name)
        for img_url in urls:
            try:
                logging.debug(f'Reading {img_url}\n')
                img = self._read_url_img(img_url)
                return img
            except Exception as e:
                logging.debug(f'Exception occurred:\n{e}')
                continue
        logging.debug(f'{self.__class__.__name__}: Failed to read any img.')
        return -1

    def fetch_random(self):
        return self.fetch_by_name(name='random')

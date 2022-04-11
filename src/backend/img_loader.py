import logging
import threading
import time
import urllib

import cv2
import emojipedia as emp
import numpy as np


class EmojipediaParser:

    def __init__(self, img_size: int, caching: bool, cache_limit: int, verbose: bool, loglevel: int):
        self.img_size = img_size
        self.caching = caching
        self.cache_limit = cache_limit
        self.verbose = verbose
        self.loglevel = loglevel

        if self.caching:
            self.cache_daemon = ImgCacheDaemon(parser=self, cache_limit=self.cache_limit, verbose=self.verbose)
            self.cache_daemon.start()

    @staticmethod
    def _read_url_img(url: str) -> np.ndarray:
        """
        Read emoji with OpenCV via url
        :param url: str, image address
        :return: np.ndarray
        """
        req = urllib.request.urlopen(url)
        array = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

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

    def destroy(self) -> None:
        if self.verbose:
            logging.debug('Destroying parser...')
        if self.caching:
            self.cache_daemon.alive = False
            self.cache_daemon.join()

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

    def fetch_random(self) -> np.ndarray:
        if self.caching and self.cache_daemon.cache_size > 0:
            if self.verbose:
                logging.debug('Taking data from cache...')
            value = self.cache_daemon.pop_value()
        else:
            value = self.fetch_by_name(name='random')
        return value

    async def fetch_random_async(self) -> np.ndarray:
        return self.fetch_random()


class ImgCacheDaemon(threading.Thread):

    def __init__(self, parser: EmojipediaParser, cache_limit: int, verbose: bool = False):
        """
        :param cache_limit:
        """
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.parser = parser
        self.cache_limit = cache_limit
        self.cache = []
        self.alive = True
        self.sleep_interval = 5
        self.verbose = verbose

    @property
    def cache_size(self) -> int:
        return len(self.cache)

    def is_cache_full(self) -> bool:
        return self.cache_size >= self.cache_limit

    def update(self, value: np.ndarray) -> None:
        """
        :param value:
        :return:
        """
        th_name = threading.current_thread().name
        if self.verbose:
            logging.debug(f'Waiting for {th_name} to be blocked')

        with self.lock:
            if self.verbose:
                logging.debug(f'{th_name} is blocked')
            self.cache.append(value)
            if self.verbose:
                logging.debug(f'{th_name} is released')

    def pop_value(self) -> np.ndarray or int:
        if self.verbose:
            logging.debug(f'Cache size is full: {self.cache_size}')

        popped = False
        while not popped:
            if not self.lock.locked():
                with self.lock:
                    value = self.cache.pop(0)
                popped = True

        return value

    def run(self) -> None:
        """
        :param parser:
        :return:
        """
        while self.alive:
            try:
                if not self.is_cache_full():
                    value = self.parser.fetch_by_name(name='random')
                    if self.verbose:
                        logging.debug(f'Cache size: {self.cache_size}')
                        logging.debug(f'Value type: {type(value)}')
                    if isinstance(value, np.ndarray):
                        self.update(value)
                else:
                    time.sleep(self.sleep_interval)
            except KeyboardInterrupt as e:
                self.alive = False

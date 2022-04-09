import argparse
import os
import sys

import cv2

import core
from img_loader import EmojipediaParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument parser for emoji baker')
    parser.add_argument('-cps', action='store_true', help='bool, use contour_points_sampling')
    parser.add_argument('-cas', action='store_true', help='bool, use contour_areas_stratification')
    parser.add_argument('-random', action='store_true', help='bool, use random images parsed by EmojipediaParser')
    parser.add_argument('-v', action='store_true', help='bool, verbosity')
    parser.add_argument('-save', action='store_true', help='bool, save images')
    parser.add_argument('-show', action='store_true', help='bool, show images')
    parser.add_argument('--random_tries', metavar='random_tries', default=10, type=int,
                        help='int, number of random images to fetch with EmojipediaParser')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transformer = core.ImgTransformer(verbose=args.v)

    if args.random:
        parser = EmojipediaParser()
        for i in range(args.random_tries):
            primary_image = parser.fetch_random()
            secondary_image = parser.fetch_random()

            if args.cps:
                warped_cps = transformer.contour_points_sampling(primary_image, secondary_image)
                if args.save:
                    cv2.imwrite(f'warped_cps_{i}.png', warped_cps)
                if args.show:
                    core.show_img(warped_cps, f'warped_cps_{i}')

            if args.cas:
                warped_cas = transformer.contour_areas_stratification(primary_image, secondary_image)
                if args.save:
                    cv2.imwrite(f'warped_cas_{i}.png', warped_cas)
                if args.show:
                    core.show_img(warped_cas, f'warped_cas_{i}')
            cv2.destroyAllWindows()
    else:
        bulb_img = cv2.imread(os.path.join('..', 'tests', 'pics', 'bulb.png'), cv2.IMREAD_UNCHANGED)
        fox_img = cv2.imread(os.path.join('..', 'tests', 'pics', 'fox.png'), cv2.IMREAD_UNCHANGED)
        primary_image = bulb_img
        secondary_image = fox_img

        if args.cps:
            warped_cps = transformer.contour_points_sampling(primary_image, secondary_image)
            if args.save:
                cv2.imwrite(f'warped_cps_test.png', warped_cps)
            if args.show:
                core.show_img(warped_cps, 'warped_cps_test')
        if args.cas:
            warped_cas = transformer.contour_areas_stratification(primary_image, secondary_image)
            if args.save:
                cv2.imwrite(f'warped_cas_test.png', warped_cas)
            if args.show:
                core.show_img(warped_cas, 'warped_cas_test')


if __name__ == '__main__':
    sys.exit(main())

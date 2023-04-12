import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='dynamic', help='method to quantify model, dynamic or static')
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')

    opt = parser.parse_args()


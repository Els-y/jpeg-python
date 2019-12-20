from collections import namedtuple
import numpy as np

RLEObject = namedtuple('RLEObject', 'num_zero value')

class BitObject(object):

    def __init__(self, value, length):
        self.value = value
        self.length = length

    def bitstring(self):
        value = self.value & ((1 << self.length) - 1)
        return bin(value)[2:].zfill(self.length)

    def __repr__(self):
        return 'BitObject(value={}, length={}, bitstring={})'.format(
            self.value, self.length, self.bitstring())

    def __eq__(self, obj):
        return obj.value == self.value and obj.length == self.length


BIT_MASK = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# Quantization Table for: Photoshop - (Save As 10)
# http://www.impulseadventure.com/photo/jpeg-quantization.html
# https://www.impulseadventure.com/photo/jpeg-quantization-lookup.html?src1=10323

LUMINANCE_QUANTIZATION_TABLE = np.array([
    [2, 2, 3, 4, 5, 6, 8, 11],
    [2, 2, 2, 4, 5, 7, 9, 11],
    [3, 2, 3, 5, 7, 9, 11, 12],
    [4, 4, 5, 7, 9, 11, 12, 12],
    [5, 5, 7, 9, 11, 12, 12, 12],
    [6, 7, 9, 11, 12, 12, 12, 12],
    [8, 9, 11, 12, 12, 12, 12, 12],
    [11, 11, 12, 12, 12, 12, 12, 12]
])

CHROMINANCE_QUANTIZATION_TABLE = np.array([
    [3, 3, 7, 13, 15, 15, 15, 15],
    [3, 4, 7, 13, 14, 12, 12, 12],
    [7, 7, 13, 14, 12, 12, 12, 12],
    [13, 13, 14, 12, 12, 12, 12, 12],
    [15, 14, 12, 12, 12, 12, 12, 12],
    [15, 12, 12, 12, 12, 12, 12, 12],
    [15, 12, 12, 12, 12, 12, 12, 12],
    [15, 12, 12, 12, 12, 12, 12, 12]
])

ZIGZAG_TABLES = [
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]

LUMINANCE_DC_NRCODES = [0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
LUMINANCE_DC_VALUES = [4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11]

CHROMINANCE_DC_NRCODES = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
CHROMINANCE_DC_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

LUMINANCE_AC_NOCODES = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
LUMINANCE_AC_VALUES = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa]

CHROMINANCE_AC_NRCODES = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
CHROMINANCE_AC_VALUES = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
]


def build_huffman_table(nr_codes, values):
    huffman_table = {}
    inverse_huffman_table = {}
    pos_in_table = 0
    code_value = 0
    for k in range(1, 17):
        for j in range(1, nr_codes[k - 1] + 1):
            bit_object = BitObject(value=code_value, length=k)
            huffman_table[values[pos_in_table]] = bit_object
            inverse_huffman_table[bit_object.bitstring()] = values[pos_in_table]
            pos_in_table += 1
            code_value += 1
        code_value <<= 1

    return huffman_table, inverse_huffman_table


LUMINANCE_DC_HUFFMAN_TABLE, LUMINANCE_DC_INVERSE_HUFFMAN_TABLE = build_huffman_table(LUMINANCE_DC_NRCODES, LUMINANCE_DC_VALUES)
LUMINANCE_AC_HUFFMAN_TABLE, LUMINANCE_AC_INVERSE_HUFFMAN_TABLE = build_huffman_table(LUMINANCE_AC_NOCODES, LUMINANCE_AC_VALUES)
CHROMINANCE_DC_HUFFMAN_TABLE, CHROMINANCE_DC_INVERSE_HUFFMAN_TABLE = build_huffman_table(CHROMINANCE_DC_NRCODES, CHROMINANCE_DC_VALUES)
CHROMINANCE_AC_HUFFMAN_TABLE, CHROMINANCE_AC_INVERSE_HUFFMAN_TABLE = build_huffman_table(CHROMINANCE_AC_NRCODES, CHROMINANCE_AC_VALUES)


def get_quantization_tables():
    return [LUMINANCE_QUANTIZATION_TABLE,
            CHROMINANCE_QUANTIZATION_TABLE,
            CHROMINANCE_QUANTIZATION_TABLE]


def get_huffman_tables():
    tables = {
        'lum_dc': LUMINANCE_DC_HUFFMAN_TABLE,
        'lum_ac': LUMINANCE_AC_HUFFMAN_TABLE,
        'chrom_dc': CHROMINANCE_DC_HUFFMAN_TABLE,
        'chrom_ac': CHROMINANCE_AC_HUFFMAN_TABLE
    }
    return tables


def get_inverse_huffman_tables():
    tables = {
        'lum_dc': LUMINANCE_DC_INVERSE_HUFFMAN_TABLE,
        'lum_ac': LUMINANCE_AC_INVERSE_HUFFMAN_TABLE,
        'chrom_dc': CHROMINANCE_DC_INVERSE_HUFFMAN_TABLE,
        'chrom_ac': CHROMINANCE_AC_INVERSE_HUFFMAN_TABLE
    }
    return tables


def zigzag_points(rows, cols):
    # constants for directions
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # move the point in different directions
    def move(direction, point):
        return {
            UP: lambda point: (point[0] - 1, point[1]),
            DOWN: lambda point: (point[0] + 1, point[1]),
            LEFT: lambda point: (point[0], point[1] - 1),
            RIGHT: lambda point: (point[0], point[1] + 1),
            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
        }[direction](point)

    # return true if point is inside the block bounds
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # start in the top-left cell
    point = (0, 0)

    # True when moving up-right, False when moving down-left
    move_up = True

    for i in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)
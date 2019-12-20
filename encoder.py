import math
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import BitObject, RLEObject, zigzag_points, get_quantization_tables, get_huffman_tables, BIT_MASK


class JpegEncoder(object):

    def __init__(self):
        self.quantization_tables = get_quantization_tables()
        self.huffman_tables = get_huffman_tables()
        self.total_bit_objects = []


    def encode(self, image_path):
        ycbcr = self.preprocess(image_path)
        height, width, _ = ycbcr.shape
        progress_bar = tqdm(total=height // 8 * width // 8, desc='encode')

        self.total_bit_objects = []
        for row in range(0, height, 8):
            for col in range(0, width, 8):
                progress_bar.set_postfix({'row': row, 'col': col})
                progress_bar.update()
                for component in range(3):
                    block = ycbcr[row:row+8, col:col+8, component]
                    dct_matrix = self.dct(block)
                    quantized_matrix = self.quantize(dct_matrix, component)
                    zigzag = self.matrix_to_zigzag(quantized_matrix)
                    bit_objects = self.entropy_encode(zigzag, component)
                    self.total_bit_objects.extend(bit_objects)
        progress_bar.close()
        return self.convert_bitstream(), height, width


    def preprocess(self, image_path):
        image = Image.open(image_path)
        ycbcr = image.convert('YCbCr')
        return np.array(ycbcr) - 128


    def dct(self, block):
        transform = np.zeros([8, 8], dtype=np.float32)
        for i in range(8):
            for j in range(8):
                c = np.sqrt(0.125) if i == 0 else 0.5
                transform[i, j] = c * np.cos((j + 0.5) * np.pi * i / 8.0)

        uv = np.matmul(np.matmul(transform, block), transform.T)
        return uv


    def quantize(self, block, component):
        return (block / self.quantization_tables[component]).round().astype(np.int32)


    def matrix_to_zigzag(self, matrix):
        return np.array([matrix[point] for point in zigzag_points(*matrix.shape)])


    def entropy_encode(self, zigzag, component):
        if component == 0:
            dc_ht = self.huffman_tables['lum_dc']
            ac_ht = self.huffman_tables['lum_ac']
        else:
            dc_ht = self.huffman_tables['chrom_dc']
            ac_ht = self.huffman_tables['chrom_ac']

        rle_objects = self.run_length_code(zigzag)
        bit_objects = []

        # dc
        dc_bit_object = self.get_bit_code(rle_objects[0].value)
        bit_objects.append(dc_ht[dc_bit_object.length])
        bit_objects.append(dc_bit_object)

        # ac
        for rle_obj in rle_objects[1:]:
            ac_bit_object = self.get_bit_code(rle_obj.value)
            bit_objects.append(ac_ht[rle_obj.num_zero << 4 | ac_bit_object.length])
            bit_objects.append(ac_bit_object)

        if bit_objects[-1] == BitObject(0, 0):
            bit_objects.pop(-1)

        return bit_objects


    def run_length_code(self, zigzag):
        rle_objects = []

        end_idx = 63
        while end_idx > 0 and zigzag[end_idx] == 0:
            end_idx -= 1

        zero_cnt = 0
        for idx, val in enumerate(zigzag):
            if idx > end_idx:
                rle_objects.append(RLEObject(num_zero=0, value=0))
                break
            elif val == 0 and zero_cnt < 15:
                zero_cnt += 1
            else:
                rle_objects.append(RLEObject(num_zero=zero_cnt, value=val))
                zero_cnt = 0
        return rle_objects


    def get_bit_code(self, value):
        tmp = value if value > 0 else -value
        length = 0
        while tmp > 0:
            tmp >>= 1
            length += 1
        bit_value = value if value > 0 else (BIT_MASK[length] + value - 1)
        return BitObject(length=length, value=bit_value)


    def convert_bitstream(self):
        bitstream = ''
        for bit_obj in self.total_bit_objects:
            value = bit_obj.value & (BIT_MASK[bit_obj.length] - 1)
            bitstream += bin(value)[2:].zfill(bit_obj.length)
        return bitstream

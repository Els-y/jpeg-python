import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import BitObject, RLEObject, zigzag_points, get_quantization_tables, get_inverse_huffman_tables, BIT_MASK


class JpegDecoder(object):

    def __init__(self):
        self.quantization_tables = get_quantization_tables()
        self.inverse_huffman_tables = get_inverse_huffman_tables()


    def decode(self, bitstream, height, width):
        ycbcr = np.zeros([height, width, 3], dtype=np.int32)

        rle_objects_group = self.parse_bitstream(bitstream)
        progress_bar = tqdm(total=height // 8 * width // 8, desc='decode')

        block_idx = 0
        for row in range(0, height, 8):
            for col in range(0, width, 8):
                progress_bar.set_postfix({'row': row, 'col': col})
                progress_bar.update()
                for component in range(3):
                    zigzag = self.entropy_decode(rle_objects_group[block_idx])
                    quantized_matrix = self.zigzag_to_matrix(zigzag)
                    dct_matrix = self.dequantize(quantized_matrix, component)
                    block = self.idct(dct_matrix)
                    ycbcr[row:row+8, col:col+8, component] = block
                    block_idx += 1

        progress_bar.close()

        image = self.postprocess(ycbcr)
        return image


    def parse_bitstream(self, bitstream):
        rle_objects_group = []
        component = 0
        bit_idx = 0
        while bit_idx < len(bitstream):
            block_rle_objects = []
            dc_inv_ht = self.inverse_huffman_tables['lum_dc'] if component == 0 else self.inverse_huffman_tables['chrom_dc']
            ac_inv_ht = self.inverse_huffman_tables['lum_ac'] if component == 0 else self.inverse_huffman_tables['chrom_ac']

            # dc
            bitstring = ''
            while bit_idx < len(bitstream):
                bitstring += bitstream[bit_idx]
                bit_idx += 1

                ht_value = dc_inv_ht.get(bitstring)
                if ht_value is not None:
                    num_zero = (ht_value >> 4) & 0xF
                    length = ht_value & 0xF
                    value_bit_string = bitstream[bit_idx:bit_idx+length]
                    bit_idx += length

                    value = int(value_bit_string, 2)
                    if value_bit_string[0] == '0':
                        value = value + 1 - (1 << length)
                    rle_obj = RLEObject(num_zero=num_zero, value=value)
                    block_rle_objects.append(rle_obj)
                    break

            # ac
            eob = False
            cnt = 0
            while cnt < 63:
                bitstring = ''
                while bit_idx < len(bitstream):
                    bitstring += bitstream[bit_idx]
                    bit_idx += 1
                    ht_value = ac_inv_ht.get(bitstring)
                    if ht_value is not None:
                        if ht_value == 0:
                            eob = True
                            break

                        num_zero = (ht_value >> 4) & 0xF
                        length = ht_value & 0xF
                        cnt += num_zero + 1

                        if num_zero == 0xF:
                            rle_obj = RLEObject(num_zero=num_zero, value=0)
                            bit_idx += 1
                        else:
                            value_bit_string = bitstream[bit_idx:bit_idx+length]
                            value = int(value_bit_string, 2)
                            bit_idx += length
                            if value_bit_string[0] == '0':
                                value = value + 1 - (1 << length)
                            rle_obj = RLEObject(num_zero=num_zero, value=value)

                        block_rle_objects.append(rle_obj)
                        break
                if eob:
                    break

            rle_objects_group.append(block_rle_objects)
            component = (component + 1) % 3

        return rle_objects_group


    def entropy_decode(self, rle_objects):
        zigzag = []

        for rle_obj in rle_objects:
            zigzag.extend([0 for _ in range(rle_obj.num_zero)] + [rle_obj.value])

        diff = 64 - len(zigzag)
        if diff > 0:
            zigzag.extend([0 for _ in range(diff)])

        return zigzag


    def get_rle_obj(self, bit_objects):
        value_first_bit = bit_objects[1].value & BIT_MASK[bit_objects[1].length - 1]
        if value_first_bit == 1:
            value = bit_objects[1].value
        else:
            value = bit_objects[1].value + 1 - (1 << bbit_objects[1].length)

        num_zero = 0
        return RLEObject(num_zero=num_zero, value=value)


    def zigzag_to_matrix(self, zigzag):
        matrix = np.zeros([8, 8], dtype=np.int32)
        for idx, point in enumerate(zigzag_points(*matrix.shape)):
            matrix[point] = zigzag[idx]
        return matrix


    def dequantize(self, matrix, component):
        return matrix * self.quantization_tables[component]


    def idct(self, matrix):
        transform = np.zeros([8, 8], dtype=np.float32)
        for i in range(8):
            for j in range(8):
                c = np.sqrt(0.125) if i == 0 else 0.5
                transform[i, j] = c * np.cos((j + 0.5) * np.pi * i / 8.0)

        f = np.matmul(np.matmul(transform.T, matrix), transform)
        return f


    def postprocess(self, ycbcr):
        ycbcr = (ycbcr + 128).astype(np.uint8)
        rgb = self.ycbcr2rgb(ycbcr)
        image = Image.fromarray(rgb, 'RGB')
        return image


    def ycbcr2rgb(self, ycbcr):
        # R  = Y +                       + (Cr - 128) *  1.40200
        # G  = Y + (Cb - 128) * -0.34414 + (Cr - 128) * -0.71414
        # B  = Y + (Cb - 128) *  1.77200
        matrix = np.array([[1.0, 0.0, 1.40200],
                           [1.0, -0.34414, -0.71414],
                           [1.0, 1.77200, 0.0]])
        ycbcr = np.array(ycbcr, dtype=np.float)
        ycbcr[:, :, 1:] -= 128
        rgb = np.dot(ycbcr, matrix.T)
        rgb = np.clip(rgb, a_min=0, a_max=255)
        return rgb.astype(np.uint8)

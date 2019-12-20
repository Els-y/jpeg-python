from encoder import JpegEncoder
from decoder import JpegDecoder


def main():
    input_file = 'lena.bmp'

    encoder = JpegEncoder()
    decoder = JpegDecoder()

    bitstream, height, width = encoder.encode(input_file)
    print('bitstream len:', len(bitstream))
    restore_image = decoder.decode(bitstream, height=height, width=width)
    restore_image.show()


if __name__ == "__main__":
    main()

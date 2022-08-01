#
# Reference arithmetic coding
# Copyright (c) Project Nayuki
#
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
#

import arithmeticcoding
import contextlib
import numpy as np
import torch

import os
from quantresnet import QuantResNet
from quantization import Quantizer

class ArithmeticCoder:

    def __init__(self, outputfile, adaptive=False):
        self.outputfile = outputfile
        self.adaptive = adaptive
        self.indevice = None
        self.inshape = None


    def encode(self, inp):
        if isinstance(inp, torch.Tensor):
            self.indevice = inp.device
            inp = inp.detach().cpu().numpy()
        self.inshape = inp.shape
        inp = np.ravel(inp).astype(np.uint8)
        self._encode(inp)


    def decode(self):
        out = self._decode()
        out = np.reshape(out, self.inshape)
        out = torch.from_numpy(out).to(self.indevice)
        return out


    def _encode(self, inp):
        if self.adaptive:
            with contextlib.closing(arithmeticcoding.BitOutputStream(open(self.outputfile, "wb"))) as bitout:
                adaptive_compress(inp, bitout)
        else:
            # Read input once to compute symbol frequencies
            freqs = get_frequencies(inp)
            freqs.increment(256)  # EOF symbol gets a frequency of 1
            # Read input again, compress with arithmetic coding, and write output file
            with contextlib.closing(arithmeticcoding.BitOutputStream(open(self.outputfile, "wb"))) as bitout:
                write_frequencies(bitout, freqs)
                compress(freqs, inp, bitout)


    def _decode(self):
        if self.adaptive:
            with open(self.outputfile, "rb") as out:
                bitin = arithmeticcoding.BitInputStream(out)
                return adaptive_decompress(bitin)
        else:
            with open(self.outputfile, "rb") as out:
                bitin = arithmeticcoding.BitInputStream(out)
                freqs = read_frequencies(bitin)
                return decompress(freqs, bitin)



def get_frequencies(inp):
    freqs = arithmeticcoding.SimpleFrequencyTable([0] * 257)
    for b in np.array(inp):
        freqs.increment(b)
    return freqs



def write_frequencies(bitout, freqs):
    for i in range(256):
        write_int(bitout, 32, freqs.get(i))



def read_frequencies(bitin):
    def read_int(n):
        result = 0
        for _ in range(n):
            result = (result << 1) | bitin.read_no_eof()  # Big endian
        return result

    freqs = [read_int(32) for _ in range(256)]
    freqs.append(1)  # EOF symbol
    return arithmeticcoding.SimpleFrequencyTable(freqs)



def write_int(bitout, numbits, value):
    for i in reversed(range(numbits)):
        bitout.write((value >> i) & 1)



def compress(freqs, inp, bitout):
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    for b in np.array(inp):
        enc.write(freqs, b)
    enc.write(freqs, 256)  # EOF
    enc.finish()  # Flush remaining code bits



def decompress(freqs, bitin):
    bitout = []
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
    while True:
        symbol = dec.read(freqs)
        if symbol == 256:  # EOF symbol
            break
        bitout.append(symbol)
    return np.array(bitout, dtype=np.uint8)



def adaptive_compress(inp, bitout):
    initfreqs = arithmeticcoding.FlatFrequencyTable(257)
    freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    for b in np.array(inp):
        enc.write(freqs, b)
        freqs.increment(b)
    enc.write(freqs, 256)  # EOF
    enc.finish()  # Flush remaining code bits



def adaptive_decompress(bitin):
    bitout = []
    initfreqs = arithmeticcoding.FlatFrequencyTable(257)
    freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
    while True:
        # Decode and write one byte
        symbol = dec.read(freqs)
        if symbol == 256:  # EOF symbol
            break
        bitout.append(symbol)
        freqs.increment(symbol)
    return np.array(bitout, dtype=np.uint8)



if __name__ == "__main__":
    q = Quantizer(4)
    p = QuantResNet(2, None, q, pretrained=True).cuda()
    ac = ArithmeticCoder("out.code", adaptive=True)

    p.eval()
    inputs = torch.rand([1, 3, 32, 32]).cuda()

    print(f"Original size: {inputs.element_size() * inputs.nelement()} bytes")

    out = p.analysis(inputs)
    print(f"Analysis size: {out.element_size() * out.nelement()} bytes")

    ac.encode(out)
    print(f"Encoded size: {os.path.getsize('out.code')} bytes")

    out = ac.decode()
    print(f"Decoded size: {out.element_size() * out.nelement()} bytes")

    out = p.synthesis(out)
    print(f"Final size: {out.element_size() * out.nelement()} bytes")

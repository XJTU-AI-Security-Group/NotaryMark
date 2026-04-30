import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Set

import numpy as np
import pandas as pd


PRIMITIVE_POLY = {
    2: 0b111,         # x^2 + x + 1
    3: 0b1011,        # x^3 + x + 1
    4: 0b10011,       # x^4 + x + 1
    5: 0b100101,      # x^5 + x^2 + 1
    6: 0b1000011,     # x^6 + x + 1
    7: 0b10000011,    # x^7 + x + 1
    8: 0b100011101,   # x^8 + x^4 + x^3 + x^2 + 1
    9: 0b1000010001,  # x^9 + x^4 + 1
    10: 0b10000001001,  # x^10 + x^3 + 1
}

VALID_PARENT_N = [2 ** m - 1 for m in range(3, 11)]  # 7..1023


def bits_to_str(bits: List[int]) -> str:
    """Convert a list of binary integers into a compact bitstring."""
    return ''.join(str(int(b)) for b in bits)


def int_to_bits(num: int, length: int) -> List[int]:
    """Convert an integer into a fixed-length bit list."""
    return [int(b) for b in bin(num)[2:].zfill(length)]


def bitstr_to_list(bitstr: str) -> List[int]:
    """Parse a bitstring into a list of 0/1 integers."""
    bitstr = str(bitstr).strip()
    if any(ch not in '01' for ch in bitstr):
        raise ValueError(f"Invalid bitstring: {bitstr}")
    return [int(ch) for ch in bitstr]


class GF2m:
    """Finite-field helper for GF(2^m) arithmetic used by BCH construction."""

    def __init__(self, m: int):
        if m not in PRIMITIVE_POLY:
            raise ValueError(f"Unsupported m={m}. Supported: {sorted(PRIMITIVE_POLY)}")
        self.m = m
        self.n = (1 << m) - 1
        self.prim_poly = PRIMITIVE_POLY[m]
        self.exp = [0] * (2 * self.n)
        self.log = [None] * (self.n + 1)
        self._build_tables()

    def _build_tables(self):
        x = 1
        for i in range(self.n):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & (1 << self.m):
                x ^= self.prim_poly
        for i in range(self.n, 2 * self.n):
            self.exp[i] = self.exp[i - self.n]

    def add(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[(self.log[a] + self.log[b]) % self.n]

    def pow_alpha(self, power: int) -> int:
        return self.exp[power % self.n]


def gf_poly_mul(field: GF2m, p: List[int], q: List[int]) -> List[int]:
    """Multiply two polynomials whose coefficients live in GF(2^m)."""
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b == 0:
                continue
            out[i + j] ^= field.mul(a, b)
    return out


def binary_poly_mul(p: List[int], q: List[int]) -> List[int]:
    """Multiply two binary polynomials represented in low-to-high order."""
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b == 0:
                continue
            out[i + j] ^= 1
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def cyclotomic_coset(i: int, n: int) -> List[int]:
    """Return the cyclotomic coset of exponent i modulo n under doubling."""
    seen = []
    x = i % n
    while x not in seen:
        seen.append(x)
        x = (x * 2) % n
    return seen


def minimal_polynomial_binary(field: GF2m, exponent: int) -> List[int]:
    """Build the minimal polynomial over GF(2) for alpha^exponent."""
    coset = cyclotomic_coset(exponent, field.n)
    poly = [1]  # low-to-high
    for e in coset:
        poly = gf_poly_mul(field, poly, [field.pow_alpha(e), 1])
    binary = []
    for coeff in poly:
        if coeff == 0:
            binary.append(0)
        elif coeff == 1:
            binary.append(1)
        else:
            raise RuntimeError(
                f"Minimal polynomial coefficient not in GF(2): coeff={coeff}, exponent={exponent}, coset={coset}"
            )
    while len(binary) > 1 and binary[-1] == 0:
        binary.pop()
    return binary


def generator_polynomial(field: GF2m, t: int) -> List[int]:
    """Build the BCH generator polynomial for correcting up to t errors."""
    used_cosets: Set[int] = set()
    g = [1]
    for i in range(1, 2 * t + 1):
        coset = cyclotomic_coset(i, field.n)
        leader = min(coset)
        if leader in used_cosets:
            continue
        used_cosets.add(leader)
        mpoly = minimal_polynomial_binary(field, leader)
        g = binary_poly_mul(g, mpoly)
    while len(g) > 1 and g[-1] == 0:
        g.pop()
    return g


def poly_mod2_div(dividend: List[int], divisor: List[int]) -> List[int]:
    """Return the binary polynomial remainder of dividend / divisor."""
    rem = dividend[:]
    m = len(divisor)
    for i in range(len(dividend) - m + 1):
        if rem[i] == 1:
            for j in range(m):
                rem[i + j] ^= divisor[j]
    return rem[-(m - 1):] if m > 1 else []



def binomial_t_for_target_success(length: int, bit_acc: float, target_success: float = 0.99) -> int:
    """Estimate the minimum error-correction budget t from bit accuracy."""
    if not (0.0 <= bit_acc <= 1.0):
        raise ValueError("bit_acc must be in [0, 1]")
    p = 1.0 - bit_acc
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return length

    q = 1.0 - p
    pmf = q ** length  # P(E=0)
    cdf = pmf
    if cdf >= target_success:
        return 0

    for k in range(0, length):
        pmf = pmf * (length - k) / (k + 1) * (p / q)
        cdf += pmf
        if cdf >= target_success:
            return k + 1
    return length


def max_feasible_t_for_length(total_length: int) -> int:
    """Search the largest feasible t for a given shortened code length."""
    max_t = 0
    for t in range(1, total_length + 1):
        try:
            ShortenedBCHEncoder(total_length=total_length, t=t)
            max_t = t
        except ValueError as exc:
            if "gives k=" in str(exc):
                break
            raise
    return max_t


@dataclass
class BCHDesign:
    """Serializable summary of one BCH design choice."""

    target_length: int
    t: int
    m: int
    parent_n: int
    parent_k: int
    shortened_n: int
    shortened_k: int
    parity_bits: int
    designed_distance_lower_bound: int
    generator_poly_low_to_high: List[int]


class ShortenedBCHEncoder:
    """Construct and encode shortened BCH codewords for a target total length."""

    def __init__(self, total_length: int, t: int):
        if total_length < 7 or total_length > 1023:
            raise ValueError("Current implementation supports total_length in [7, 1023].")
        if t < 1:
            raise ValueError("t must be >= 1")

        self.total_length = total_length
        self.t = t
        self.m = min(m for m in range(3, 11) if (2 ** m - 1) >= total_length)
        self.field = GF2m(self.m)
        self.parent_n = self.field.n
        self.g_low = generator_polynomial(self.field, t)
        self.parity_bits = len(self.g_low) - 1
        self.parent_k = self.parent_n - self.parity_bits
        self.shorten_by = self.parent_n - total_length
        self.k = self.parent_k - self.shorten_by
        if self.k <= 0:
            raise ValueError(
                f"Invalid design: total_length={total_length}, t={t} gives k={self.k} <= 0. "
                f"Try smaller t or larger total_length."
            )
        self.g_high = list(reversed(self.g_low))

    def design_summary(self) -> BCHDesign:
        """Return a compact design summary for reporting and metadata export."""
        return BCHDesign(
            target_length=self.total_length,
            t=self.t,
            m=self.m,
            parent_n=self.parent_n,
            parent_k=self.parent_k,
            shortened_n=self.total_length,
            shortened_k=self.k,
            parity_bits=self.parity_bits,
            designed_distance_lower_bound=2 * self.t + 1,
            generator_poly_low_to_high=self.g_low[:],
        )

    def encode_bits(self, message_bits: List[int]) -> List[int]:
        """Encode one message bit vector into a shortened BCH codeword."""
        if len(message_bits) != self.k:
            raise ValueError(f"message_bits length must be {self.k}, got {len(message_bits)}")
        full_message = [0] * self.shorten_by + [int(b) & 1 for b in message_bits]
        shifted = full_message + [0] * self.parity_bits
        remainder = poly_mod2_div(shifted, self.g_high)
        full_codeword = full_message + remainder
        shortened_codeword = full_codeword[self.shorten_by:]
        assert len(shortened_codeword) == self.total_length
        return shortened_codeword

    def encode_batch(self, messages: List[List[int]]) -> List[List[int]]:
        """Encode a batch of messages."""
        return [self.encode_bits(msg) for msg in messages]

    def random_messages(self, num_codewords: int, seed: int = 42) -> List[List[int]]:
        """Generate random message vectors compatible with the current code design."""
        rng = np.random.default_rng(seed)
        messages = rng.integers(0, 2, size=(num_codewords, self.k), dtype=np.uint8)
        return messages.tolist()

    def build_codebook_table(
        self,
        num_codewords: int = 16,
        save_csv_path: str | None = None,
        use_random: bool = False,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Build and optionally save a codebook table of encoded messages."""
        rows = []
        if use_random:
            messages = self.random_messages(num_codewords, seed=seed)
        else:
            if self.k > 62:
                raise ValueError("Sequential mode supports k <= 62 for safe integer enumeration. Use use_random=True.")
            if num_codewords > (1 << self.k):
                raise ValueError(f"num_codewords={num_codewords} exceeds 2^k={1 << self.k}")
            messages = [int_to_bits(i, self.k) for i in range(num_codewords)]

        for user_id, message_bits in enumerate(messages):
            codeword_bits = self.encode_bits(message_bits)
            rows.append({
                "user_id": user_id,
                "message_bits": bits_to_str(message_bits),
                "codeword_bits": bits_to_str(codeword_bits),
            })

        df = pd.DataFrame(rows)
        if save_csv_path is not None:
            save_dir = os.path.dirname(save_csv_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        return df


def metadata_dict(design: BCHDesign, bit_acc: float | None = None, target_success: float | None = None) -> Dict:
    """Convert a BCH design summary into JSON-serializable metadata."""
    data = {
        "target_length": design.target_length,
        "t": design.t,
        "m": design.m,
        "parent_n": design.parent_n,
        "parent_k": design.parent_k,
        "shortened_n": design.shortened_n,
        "shortened_k": design.shortened_k,
        "parity_bits": design.parity_bits,
        "designed_distance_lower_bound": design.designed_distance_lower_bound,
        "generator_polynomial_binary_low_to_high": bits_to_str(design.generator_poly_low_to_high),
    }
    if bit_acc is not None:
        data["bit_acc"] = bit_acc
    if target_success is not None:
        data["target_success"] = target_success
    return data


def main():
    """Interactively or programmatically design a BCH codebook and save it."""
    parser = argparse.ArgumentParser(description="Design a shortened BCH code from total length and bit accuracy, then generate encoded messages.")
    parser.add_argument("--length", type=int, help="Target total codeword length L (7..1023)")
    parser.add_argument("--bit_acc", type=float, help="Bit accuracy, e.g. 0.94")
    parser.add_argument("--num_codewords", type=int, default=32, help="Number of encoded messages to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_random", action="store_true", help="Use random messages instead of sequential messages")
    parser.add_argument("--target_success", type=float, default=0.99)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--metadata_json", type=str, default=None)
    parser.add_argument("--non_interactive", action="store_true", help="Skip the Enter-to-confirm step")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve the target design parameters
    # ------------------------------------------------------------------
    total_length = args.length if args.length is not None else int(input("Input total codeword length L (7..1023): ").strip())
    bit_acc = args.bit_acc if args.bit_acc is not None else float(input("Input bit accuracy (e.g. 0.94): ").strip())
    if total_length < 7 or total_length > 1023:
        raise ValueError("Current implementation supports L in [7, 1023].")
    if not (0.0 <= bit_acc <= 1.0):
        raise ValueError("bit_acc must be in [0, 1]")

    t_req = binomial_t_for_target_success(total_length, bit_acc, args.target_success)
    max_feasible_t = max_feasible_t_for_length(total_length)
    if max_feasible_t < 1:
        raise ValueError(f"No feasible BCH design found for total_length={total_length}.")
    suggested_t = min(t_req, max_feasible_t)

    print("=" * 68)
    print("BCH design suggestion")
    print("=" * 68)
    print(f"Target total length (L) : {total_length}")
    print(f"Bit accuracy            : {bit_acc:.6f}")
    print(f"Target success rate     : {args.target_success:.2%}")
    print(f"Suggested required t    : {suggested_t}")
    print(f"Required d_min >=       : {2 * suggested_t + 1}")
    print("Design rule             : need P(Binomial(L, 1-bit_acc) <= t) >= target_success")
    print(f"Max feasible t for L    : {max_feasible_t}")
    if t_req > max_feasible_t:
        print(f"Raw required t          : {t_req}  (not feasible for L={total_length}, auto-capped)")
    print("=" * 68)

    # ------------------------------------------------------------------
    # Confirm or override the suggested correction budget
    # ------------------------------------------------------------------
    chosen_t = suggested_t
    if not args.non_interactive:
        reply = input("Press Enter to accept suggested t, input another t to override, or type q to quit: ").strip()
        if reply.lower() == 'q':
            print("Quit without generating codewords.")
            return
        if reply:
            chosen_t = int(reply)

    if chosen_t > max_feasible_t:
        raise ValueError(
            f"Chosen t={chosen_t} is not feasible for total_length={total_length}. "
            f"Maximum feasible t is {max_feasible_t}."
        )

    encoder = ShortenedBCHEncoder(total_length=total_length, t=chosen_t)
    design = encoder.design_summary()

    output_csv = args.output_csv or f"bch_codebook_L{design.shortened_n}_t{design.t}.csv"
    metadata_json = args.metadata_json or f"bch_codebook_L{design.shortened_n}_t{design.t}_metadata.json"

    # ------------------------------------------------------------------
    # Generate and save the codebook
    # ------------------------------------------------------------------
    df = encoder.build_codebook_table(
        num_codewords=args.num_codewords,
        save_csv_path=output_csv,
        use_random=args.use_random,
        seed=args.seed,
    )

    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata_dict(design, bit_acc=bit_acc, target_success=args.target_success), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print("Generated BCH codebook")
    print("=" * 68)
    print(f"Parent BCH length       : {design.parent_n}")
    print(f"Shortened code length   : {design.shortened_n}")
    print(f"Message length          : {design.shortened_k}")
    print(f"Parity bits             : {design.parity_bits}")
    print(f"Chosen t                : {design.t}")
    print(f"Designed d_min >=       : {design.designed_distance_lower_bound}")
    print(f"Generator poly degree   : {len(design.generator_poly_low_to_high) - 1}")
    print(f"Num codewords generated : {len(df)}")
    print(f"CSV saved to            : {output_csv}")
    print(f"Metadata saved to       : {metadata_json}")
    print("Sample rows:")
    print(df.head(min(5, len(df))).to_string(index=False))
    print("=" * 68)


if __name__ == "__main__":
    main()

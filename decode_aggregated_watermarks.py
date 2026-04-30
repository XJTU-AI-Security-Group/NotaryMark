import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from generate_bch_codebook import GF2m, ShortenedBCHEncoder


def bits_to_str(bits: np.ndarray) -> str:
    """Convert a NumPy bit array into a compact string."""
    return "".join(map(str, bits.tolist()))


def str_to_bits(bitstr: str) -> np.ndarray:
    """Parse a bitstring into a uint8 NumPy array."""
    bitstr = str(bitstr).strip()
    if bitstr == "":
        raise ValueError("Input bitstring is empty.")
    if any(ch not in "01" for ch in bitstr):
        raise ValueError(f"Found a non-binary character in bitstring: {bitstr}")
    return np.array([int(ch) for ch in bitstr], dtype=np.uint8)


class CodebookNearestDecoder:
    """Decode by nearest-neighbor matching against a precomputed codebook."""

    def __init__(self, codebook_csv: str, metadata_json: Optional[str] = None):
        self.codebook_csv = codebook_csv
        self.metadata_json = metadata_json
        self.df = pd.read_csv(codebook_csv, encoding="utf-8-sig")

        required_cols = {"message_bits", "codeword_bits"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Codebook is missing required columns: {sorted(missing)}")

        self.user_id_col = "user_id" if "user_id" in self.df.columns else None
        self.message_bits = self.df["message_bits"].astype(str).tolist()
        self.codeword_bits = self.df["codeword_bits"].astype(str).tolist()

        if len(self.codeword_bits) == 0:
            raise ValueError("Codebook is empty.")

        self.n = len(self.codeword_bits[0])
        for idx, codeword in enumerate(self.codeword_bits):
            if len(codeword) != self.n:
                raise ValueError(f"Row {idx} in the codebook has an inconsistent codeword_bits length.")
            if any(ch not in "01" for ch in codeword):
                raise ValueError(f"Row {idx} in the codebook contains a non-binary codeword_bits value.")

        self.codeword_arr = np.array(
            [[int(ch) for ch in codeword] for codeword in self.codeword_bits],
            dtype=np.uint8,
        )

        self.t = None
        self.designed_distance_lower_bound = None
        if metadata_json is not None and os.path.exists(metadata_json):
            with open(metadata_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.t = meta.get("t")
            self.designed_distance_lower_bound = meta.get("designed_distance_lower_bound")

    def decode_str(self, received_str: str) -> dict:
        """Return the closest codeword entry and related diagnostics."""
        recv = str_to_bits(received_str)
        if len(recv) != self.n:
            raise ValueError(f"Received bitstring length is {len(recv)}, but the codebook length is {self.n}.")

        distances = np.sum(self.codeword_arr != recv, axis=1)
        min_dist = int(np.min(distances))
        best_indices = np.where(distances == min_dist)[0]
        best_idx = int(best_indices[0])

        decoded_user_id = None
        if self.user_id_col is not None:
            decoded_user_id = self.df.iloc[best_idx][self.user_id_col]

        is_unique_best = len(best_indices) == 1
        is_recoverable = None
        if self.t is not None:
            is_recoverable = bool(min_dist <= self.t and is_unique_best)

        return {
            "decoded_user_id": decoded_user_id,
            "decoded_message_bits": self.message_bits[best_idx],
            "corrected_codeword_bits": self.codeword_bits[best_idx],
            "hamming_distance": min_dist,
            "num_tied_candidates": int(len(best_indices)),
            "is_unique_best": bool(is_unique_best),
            "is_recoverable": is_recoverable,
        }


class BCHMessageMatchingDecoder:
    """Decode with BCH error correction and then map the message to the codebook."""

    def __init__(self, codebook_csv: str, metadata_json: str):
        self.codebook_csv = codebook_csv
        self.metadata_json = metadata_json
        self.df = pd.read_csv(codebook_csv, encoding="utf-8-sig")
        with open(metadata_json, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        required_cols = {"message_bits", "codeword_bits"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Codebook is missing required columns: {sorted(missing)}")

        self.user_id_col = "user_id" if "user_id" in self.df.columns else None
        self.message_to_user_id = {}
        if self.user_id_col is not None:
            for _, row in self.df.iterrows():
                self.message_to_user_id[str(row["message_bits"])] = row[self.user_id_col]

        self.n = int(self.meta["target_length"])
        self.t = int(self.meta["t"])
        self.encoder = ShortenedBCHEncoder(total_length=self.n, t=self.t)
        self.parent_n = self.encoder.parent_n
        self.shorten_by = self.encoder.shorten_by
        self.k = self.encoder.k
        self.field = self.encoder.field

    def _compute_syndromes(self, coeff_low: np.ndarray) -> list[int]:
        """Compute BCH syndromes from the low-to-high coefficient view."""
        syndromes = [0] * (2 * self.t + 1)
        for j in range(1, 2 * self.t + 1):
            s = 0
            for i, bit in enumerate(coeff_low):
                if bit:
                    s ^= self.field.pow_alpha(j * i)
            syndromes[j] = s
        return syndromes

    def _berlekamp_massey(self, syndromes: list[int]) -> list[int]:
        """Solve for the BCH error locator polynomial."""
        c = [1] + [0] * (2 * self.t)
        b = [1] + [0] * (2 * self.t)
        l = 0
        m = 1
        bb = 1

        for n_idx in range(0, 2 * self.t):
            d = syndromes[n_idx + 1]
            for i in range(1, l + 1):
                if c[i] != 0 and syndromes[n_idx + 1 - i] != 0:
                    d ^= self.field.mul(c[i], syndromes[n_idx + 1 - i])

            if d == 0:
                m += 1
                continue

            t_poly = c[:]
            factor = self.field.mul(d, self.field.exp[(self.field.n - self.field.log[bb]) % self.field.n])
            for i in range(m, 2 * self.t + 1):
                if b[i - m] != 0:
                    c[i] ^= self.field.mul(factor, b[i - m])

            if 2 * l <= n_idx:
                l = n_idx + 1 - l
                b = t_poly
                bb = d
                m = 1
            else:
                m += 1

        return c[: l + 1]

    def _chien_search(self, sigma: list[int]) -> list[int]:
        """Locate error positions by evaluating the locator polynomial."""
        error_positions = []
        for i in range(self.parent_n):
            x = self.field.pow_alpha((self.parent_n - i) % self.parent_n)
            val = 0
            x_pow = 1
            for coeff in sigma:
                if coeff != 0:
                    val ^= self.field.mul(coeff, x_pow)
                x_pow = self.field.mul(x_pow, x)
            if val == 0:
                error_positions.append(i)
        return error_positions

    def _decode_bch_codeword(self, received_str: str) -> tuple[str, str, int, bool]:
        """Decode one BCH codeword and return corrected outputs plus diagnostics."""
        recv = str_to_bits(received_str)
        if len(recv) != self.n:
            raise ValueError(f"Received bitstring length is {len(recv)}, but the BCH code length is {self.n}.")

        full_high = np.concatenate(
            [np.zeros(self.shorten_by, dtype=np.uint8), recv.astype(np.uint8)]
        )
        coeff_low = full_high[::-1].copy()
        syndromes = self._compute_syndromes(coeff_low)

        if all(s == 0 for s in syndromes[1:]):
            corrected_high = full_high
            num_errors = 0
        else:
            sigma = self._berlekamp_massey(syndromes)
            error_positions = self._chien_search(sigma)
            if len(error_positions) != len(sigma) - 1:
                raise RuntimeError("BCH decoding failed: the number of located roots does not match the locator polynomial degree.")
            if len(error_positions) > self.t:
                raise RuntimeError("BCH decoding failed: the estimated number of errors exceeds t.")

            corrected_low = coeff_low.copy()
            for pos in error_positions:
                corrected_low[pos] ^= 1
            corrected_high = corrected_low[::-1]

            check_syndromes = self._compute_syndromes(corrected_low)
            if any(s != 0 for s in check_syndromes[1:]):
                raise RuntimeError("BCH decoding failed: non-zero syndromes remain after correction.")
            num_errors = len(error_positions)

        shortened_high = corrected_high[self.shorten_by :]
        corrected_codeword_bits = bits_to_str(shortened_high)
        decoded_message_bits = bits_to_str(shortened_high[: self.k])
        return decoded_message_bits, corrected_codeword_bits, num_errors, True

    def decode_str(self, received_str: str) -> dict:
        """Decode one received bitstring and map it back to the codebook user entry."""
        decoded_message_bits, corrected_codeword_bits, hamming_distance, is_recoverable = (
            self._decode_bch_codeword(received_str)
        )
        decoded_user_id = self.message_to_user_id.get(decoded_message_bits)
        return {
            "decoded_user_id": decoded_user_id,
            "decoded_message_bits": decoded_message_bits,
            "corrected_codeword_bits": corrected_codeword_bits,
            "hamming_distance": hamming_distance,
            "num_tied_candidates": 1,
            "is_unique_best": True,
            "is_recoverable": is_recoverable,
        }


def infer_metadata_json_path(codebook_csv: str) -> Optional[str]:
    """Infer the default metadata JSON path that matches one codebook CSV."""
    base, ext = os.path.splitext(codebook_csv)
    candidate = f"{base}_metadata.json"
    if os.path.exists(candidate):
        return candidate
    return None


def decode_csv_minimal(
    input_csv: str,
    output_csv: str,
    codebook_csv: str,
    bit_col: str = "voted_watermark",
    metadata_json: Optional[str] = None,
    keep_input_details: bool = False,
    decoder_mode: str = "auto",
):
    """Decode aggregated watermark trials and save the decoded result table."""
    if metadata_json is None:
        metadata_json = infer_metadata_json_path(codebook_csv)

    if decoder_mode not in {"auto", "bch_then_match", "nearest"}:
        raise ValueError("decoder_mode must be one of: auto, bch_then_match, nearest")

    if decoder_mode == "nearest":
        decoder = CodebookNearestDecoder(codebook_csv=codebook_csv, metadata_json=metadata_json)
        resolved_mode = "nearest"
    elif decoder_mode == "bch_then_match":
        if metadata_json is None:
            raise ValueError("metadata_json is required when decoder_mode=bch_then_match")
        decoder = BCHMessageMatchingDecoder(codebook_csv=codebook_csv, metadata_json=metadata_json)
        resolved_mode = "bch_then_match"
    else:
        if metadata_json is not None:
            with open(metadata_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if {"target_length", "t", "generator_polynomial_binary_low_to_high"}.issubset(meta.keys()):
                decoder = BCHMessageMatchingDecoder(codebook_csv=codebook_csv, metadata_json=metadata_json)
                resolved_mode = "bch_then_match"
            else:
                decoder = CodebookNearestDecoder(codebook_csv=codebook_csv, metadata_json=metadata_json)
                resolved_mode = "nearest"
        else:
            decoder = CodebookNearestDecoder(codebook_csv=codebook_csv, metadata_json=metadata_json)
            resolved_mode = "nearest"

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if bit_col not in df.columns:
        raise ValueError(f"Input CSV does not contain column: {bit_col}")

    # ------------------------------------------------------------------
    # Decode each aggregated bitstring
    # ------------------------------------------------------------------
    results = []
    decode_errors = 0

    for _, row in df.iterrows():
        recv_str = str(row[bit_col]).strip()
        try:
            result = decoder.decode_str(recv_str)
        except Exception:
            result = {
                "decoded_user_id": None,
                "decoded_message_bits": None,
                "corrected_codeword_bits": None,
                "hamming_distance": None,
                "num_tied_candidates": None,
                "is_unique_best": False,
                "is_recoverable": None,
            }
            decode_errors += 1
        results.append(result)

    result_df = pd.DataFrame(results)

    preferred_input_cols = ["trial_id", "sample_size", bit_col]
    if keep_input_details:
        preferred_input_cols.extend(
            ["correct_bits", "total_bits", "agg_accuracy", "gt", "sampled_filenames"]
        )
    kept_input_cols = [c for c in preferred_input_cols if c in df.columns]

    out_df = pd.concat(
        [
            df[kept_input_cols].reset_index(drop=True),
            result_df[
                [
                    "decoded_user_id",
                    "decoded_message_bits",
                    "corrected_codeword_bits",
                    "hamming_distance",
                    "num_tied_candidates",
                    "is_unique_best",
                    "is_recoverable",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    # ------------------------------------------------------------------
    # Save outputs and print a concise summary
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    total = len(out_df)
    decoded_rows = total - decode_errors

    print("=" * 68)
    print("Aggregated watermark decoding summary")
    print("=" * 68)
    print(f"Input file                 : {input_csv}")
    print(f"Output file                : {output_csv}")
    print(f"Codebook file              : {codebook_csv}")
    print(f"Metadata file              : {metadata_json}")
    print(f"Bit column                 : {bit_col}")
    print(f"Decoder mode               : {resolved_mode}")
    print(f"Code length                : {decoder.n}")
    print(f"Total rows                 : {total}")
    print(f"Decoded rows               : {decoded_rows}")
    print(f"Parse/decode errors        : {decode_errors}")

    valid_dist = out_df["hamming_distance"].dropna()
    if len(valid_dist) > 0:
        print(f"Avg hamming distance       : {valid_dist.mean():.2f}")
        print(f"Median hamming distance    : {valid_dist.median():.2f}")

    if "is_unique_best" in out_df.columns:
        unique_best = int(out_df["is_unique_best"].fillna(False).sum())
        print(f"Unique-best decoded rows   : {unique_best}")

    if decoder.t is not None and "is_recoverable" in out_df.columns:
        recoverable = int(out_df["is_recoverable"].fillna(False).sum())
        print(f"Recoverable rows (t={decoder.t}) : {recoverable}")
    print("=" * 68)

    return out_df


def main():
    """Command-line entry point for aggregated watermark decoding."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode aggregated watermark bitstrings produced by aggregate_majority_vote.py "
            "using either nearest-neighbor codebook matching or BCH decoding."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="",
        help="Path to the input aggregation_trials.csv file.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Path to the output decoded_trials.csv file.",
    )
    parser.add_argument(
        "--codebook_csv",
        default="",
        help="Path to the codebook CSV. The file must contain message_bits and codeword_bits columns.",
    )
    parser.add_argument(
        "--metadata_json",
        default=None,
        help="Optional metadata JSON for the codebook. If omitted, the script tries to infer it automatically.",
    )
    parser.add_argument(
        "--bit_col",
        default="voted_watermark",
        help="Column name containing the aggregated bitstring to decode.",
    )
    parser.add_argument(
        "--keep_input_details",
        action="store_true",
        help="Keep additional input columns from aggregation_trials.csv in the output.",
    )
    parser.add_argument(
        "--decoder_mode",
        default="bch_then_match",
        choices=["auto", "bch_then_match", "nearest"],
        help="auto: infer the decoder type; bch_then_match: BCH decode first and then map message_bits to user_id; nearest: nearest-neighbor codebook matching.",
    )
    args = parser.parse_args()

    decode_csv_minimal(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        codebook_csv=args.codebook_csv,
        bit_col=args.bit_col,
        metadata_json=args.metadata_json,
        keep_input_details=args.keep_input_details,
        decoder_mode=args.decoder_mode,
    )


if __name__ == "__main__":
    main()

import argparse
import fnmatch
import io
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download


warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")


CHANNEL_FILTER = [
    "@duongfg",
    "@meGAME_Official",
    "@daylaphegame",
    "@ducisreal",
    "@VuTruNguyenThuy",
    "@ThanhPahm",
    "@Spiderum",
    "@BoringPPL",
    "@tamhonanuong",
    "@KienThucQuanSu",
    "@nguoithanhcong1991",
    "@baihoc10phut",
    "@betterversionvn",
    "@caikinhdi_vn",
    "@CuThongThai",
    "@SpiderumBooks",
    "@CoBaBinhDuong",
    "@HocvienBovaGau",
    "@CDTeam-Why",
    "@toansam",
    "@AnhThamTu",
    "@Web5Ngay",
    "@W2Whorror",
    "@FonosVietnam",
    "@gc.gamelab",
    "@PhanTichGame",
    "@ThePresentWriter",
    "@AnimeRewind.Official",
]


SUPPORTED_EXTENSIONS = {".arrow", ".feather", ".parquet", ".pq"}


def _read_table_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".arrow", ".feather"}:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.feather as feather
        try:
            with pa.memory_map(path, "r") as source:
                try:
                    reader = ipc.open_file(source)
                except Exception:
                    source.seek(0)
                    reader = ipc.open_stream(source)
                table = reader.read_all()
            return table.to_pandas()
        except Exception:
            table = feather.read_table(path)
            return table.to_pandas()
    raise ValueError(f"Unsupported file type: {ext}")


def _resolve_audio_path(path_value, audio_root):
    if not isinstance(path_value, str):
        return None
    if os.path.isabs(path_value) and os.path.exists(path_value):
        return path_value
    if os.path.exists(path_value):
        return path_value
    if audio_root:
        p = os.path.join(audio_root, path_value)
        if os.path.exists(p):
            return p
        p = os.path.join(audio_root, "audio", path_value)
        if os.path.exists(p):
            return p
    return None


def _decode_audio_cell(cell, default_sr=None, audio_root=None):
    if isinstance(cell, dict):
        if "array" in cell and "sampling_rate" in cell:
            wav = np.asarray(cell["array"], dtype=np.float32)
            sr = int(cell["sampling_rate"])
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav, sr
        if "bytes" in cell:
            buf = io.BytesIO(cell["bytes"])
            wav, sr = sf.read(buf)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32), int(sr)
        if "path" in cell:
            path = _resolve_audio_path(cell["path"], audio_root=audio_root)
            if path is None:
                raise FileNotFoundError(f"Audio path not found: {cell['path']}")
            wav, sr = sf.read(path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32), int(sr)
    if isinstance(cell, (bytes, bytearray)):
        buf = io.BytesIO(cell)
        wav, sr = sf.read(buf)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)
    if isinstance(cell, np.ndarray):
        if default_sr is None:
            raise ValueError("Audio array missing sampling rate")
        wav = cell.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav, int(default_sr)
    if isinstance(cell, str):
        path = _resolve_audio_path(cell, audio_root=audio_root)
        if path is None:
            raise FileNotFoundError(f"Audio path not found: {cell}")
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)
    raise TypeError("Unsupported audio format in input file")


def _match_patterns(path, patterns):
    for pat in patterns:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(os.path.basename(path), pat):
            return True
    return False


def _list_data_files(repo_id, pattern, split):
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    if pattern in {"", "auto"}:
        files = [f for f in files if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    else:
        patterns = [p.strip() for p in pattern.split(",") if p.strip()]
        files = [f for f in files if _match_patterns(f, patterns)]

    if split:
        files = [f for f in files if split in f]
    if not files:
        raise FileNotFoundError(
            "No matching files found in dataset repo. "
            "Try --pattern auto or --pattern '*.parquet' and verify --split."
        )
    return sorted(files)


def export_audio_and_metadata(
    repo_id,
    out_dir,
    pattern,
    split,
    audio_col,
    sample_rate_col,
    start_index,
    audio_root,
    keep_cols,
    channel_filter,
):
    os.makedirs(out_dir, exist_ok=True)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.csv")

    files = _list_data_files(repo_id, pattern, split)
    write_header = True
    index_now = start_index
    total_rows = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in files:
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=tmpdir,
                local_dir=tmpdir,
                local_dir_use_symlinks=False,
            )

            df = _read_table_any(local_path)
            if audio_col not in df.columns:
                raise ValueError(f"Column '{audio_col}' not found in {filename}")
            if channel_filter is not None:
                if "channel" not in df.columns:
                    raise ValueError("Column 'channel' not found for filtering")
                df = df[df["channel"].isin(channel_filter)].reset_index(drop=True)
            if keep_cols:
                missing = [c for c in keep_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns in {filename}: {missing}")
                df = df[keep_cols]

            audio_filenames = []
            for _, row in df.iterrows():
                default_sr = None
                if sample_rate_col and sample_rate_col in df.columns:
                    default_sr = row[sample_rate_col]
                try:
                    wav, sr = _decode_audio_cell(
                        row[audio_col],
                        default_sr=default_sr,
                        audio_root=audio_root,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Decode failed at index {index_now}: {row[audio_col]!r}"
                    ) from e
                wav_name = f"{index_now:08d}.wav"
                sf.write(os.path.join(audio_dir, wav_name), wav, sr)
                audio_filenames.append(wav_name)
                index_now += 1

            df_out = df.copy()
            df_out[audio_col] = audio_filenames
            df_out.to_csv(metadata_path, mode="a", header=write_header, index=False)
            write_header = False
            total_rows += len(df_out)

            try:
                os.remove(local_path)
            except OSError:
                pass

    print(f"Processed {len(files)} files from {repo_id}")
    print(f"Saved {total_rows} rows -> {metadata_path}")
    print(f"Audio folder -> {audio_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset shards and export audio + metadata without keeping shards",
    )
    parser.add_argument("dataset", help="Dataset name, e.g. vuhoanhuy/viVoice-v1-p3")
    parser.add_argument("--out_dir", type=str, default="output_data")
    parser.add_argument("--pattern", type=str, default="auto")
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Optional substring filter on filenames (leave empty to use all)",
    )
    parser.add_argument("--audio_col", type=str, default="audio")
    parser.add_argument("--sample_rate_col", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--audio_root", type=str, default=None)
    parser.add_argument("--no_channel_filter", action="store_true")
    args = parser.parse_args()

    keep_cols = ["channel", "text", "audio", "id"]
    channel_filter = None if args.no_channel_filter else CHANNEL_FILTER

    export_audio_and_metadata(
        repo_id=args.dataset,
        out_dir=args.out_dir,
        pattern=args.pattern,
        split=args.split,
        audio_col=args.audio_col,
        sample_rate_col=args.sample_rate_col,
        start_index=args.start_index,
        audio_root=args.audio_root,
        keep_cols=keep_cols,
        channel_filter=channel_filter,
    )


if __name__ == "__main__":
    main()

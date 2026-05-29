# RUST UNIVERSAL EXTRACTION CRATES
## The krampus-ingest extraction layer — no more garbage UTF-8 decodes

Magic bytes → correct crate → raw text/metadata → Needle swarm → GO-25 payload → graph.

| Format | Crate | Why |
|---|---|---|
| `.zip .tar .gz .7z` | `exarch` / `archive` | Zero-unsafe, zip-bomb safe, path-traversal safe, 2000MB/s |
| `.sqlite .db` | `sqlx` (async) | Drop a .db file, dump tables to text, no ORM needed |
| `.parquet .csv` | `polars` | 2GB Parquet sliced in milliseconds, multithreaded |
| `.jpg .png .webp .tiff` | `image` (decode) + `ocrs` (text) | ocrs = pure Rust OCR, RTen neural net, CPU, no cloud needed |
| `.pdf` | `pdfium-render` / `lopdf` | Google PDFium binding, text streams + page-to-image for OCR |
| `.mp4 .mov .wav .mp3` | `ffmpeg-light` | Type-safe wrapper over system ffmpeg, no C bloat in binary |
| `.docx .odt` | `dotext` / `rdocx` | Unzips DOCX, returns pure text, no XML garbage |
| `.xlsx .xls` | `calamine` | Native Excel reader, what Python Pandas uses under the hood |
| `.md .txt` | stdlib read_to_string | Just read it |

## The actual flow

```
file arrives
  → read magic bytes (first 16 bytes)
  → match to crate above
  → extract raw text / metadata
  → hand to Needle swarm (already running ×6)
  → GO-25 typed JSON out
  → ABSURD queue → graph staging
```

No HumanReview dead-ends for format reasons.
No garbage UTF-8 decodes on binary files.
Everything handled deterministically before any model is called.

## Add to krampus-ingest Cargo.toml when building extraction layer
exarch, sqlx, polars, image, ocrs, pdfium-render, ffmpeg-light, dotext, calamine

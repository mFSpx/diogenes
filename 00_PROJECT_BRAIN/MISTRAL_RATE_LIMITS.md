# Mistral La Plateforme — Rate Limits Reference

Key in secrets.env: `MISTRAL_API_KEY`  
Base URL: `https://api.mistral.ai/v1`  
CLI: `vibe` (binary at `.venv/bin/vibe`, package `mistral-vibe`)

## High-throughput models (no monthly cap)

| Model | TPM | RPS |
|---|---|---|
| codestral-2508 | 625,000 | 2.08 |
| ministral-3b-2512 | 1,300,000 | 12.50 |
| ministral-8b-2512 | 625,000 | 3.13 |
| ministral-14b-2512 | 937,500 | 0.50 |
| mistral-small-2506 | 2,250,000 | 5.00 |
| mistral-medium-2505 | 375,000 | 0.42 |
| mistral-medium-2508 | 356,250 | 0.38 |
| open-mistral-nemo | 937,500 | 0.50 |
| labs-leanstral-2603 | 5,000,000 | 0.63 |
| mistral-embed-2312 | 20,000,000 | 1.00 |

## Capped models (4M token/month default)

| Model | TPM | Monthly | RPS |
|---|---|---|---|
| devstral-2512 | 50,000 | 4M | 1.00 |
| devstral-medium-2507 | 50,000 | 4M | 1.00 |
| devstral-small-2507 | 50,000 | 4M | 1.00 |
| mistral-large-2512 | 50,000 | 4M | 1.00 |
| mistral-medium-3-5 | 50,000 | 4M | 1.00 |
| mistral-small-2603 | 50,000 | — | 0.83 |
| pixtral-large-2411 | 50,000 | 4M | 1.00 |
| mistral-ocr-2505 | 50,000 | 4M | 1.00 |
| mistral-ocr-2512 | 50,000 | 4M | 1.00 |
| mistral-moderation-2411/2603 | 50,000 | 4M | 1.67 |
| voxtral-* | 50,000 | varies | 1.00 |
| magistral-medium-2509 | 75,000 | 1B | 0.08 |
| magistral-small-2509 | 25,000 | 1B | 0.03 |
| mistral-large-2411 | 600,000 | 200B | 0.43 |

## Decision guide

- **Code work, no monthly budget pressure**: `codestral-2508` (625k TPM, no cap)
- **Cheapest fast inference**: `ministral-3b-2512` (1.3M TPM, 12.5 RPS)
- **Embedding**: `mistral-embed-2312` (20M TPM — firehose)
- **Heavy reasoning**: `magistral-medium-2509` (careful: 0.08 RPS)
- **Groq alternative**: use existing GROQ_API_KEY + llama models for speed

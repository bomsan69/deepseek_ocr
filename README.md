# deepseek-ocr

DeepSeek OCR 모델을 이용해 PDF를 Markdown으로 변환하는 CLI 도구

## Project Info

| 항목 | 값 |
|---|---|
| **Path** | `/Users/jeahyungchung/Project/AI/ocr_project/deepseek_ocr` |

## 개요

[DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) 모델을 사용해 PDF 파일의 각 페이지를 이미지로 변환한 뒤 OCR을 수행하고, 결과를 Markdown 파일로 저장합니다.

- PDF 페이지를 PNG 이미지로 렌더링 (PyMuPDF)
- DeepSeek-OCR 모델로 텍스트 추출
- grounding 메타데이터 자동 제거
- 결과를 `{날짜}_{파일명}_ocr.md` 형식으로 저장
- 다중 GPU 병렬 처리 지원 (`NUM_GPUS` 설정)

## 요구 사항

- Python 3.13+
- CUDA 지원 GPU (CUDA 12.6 이상 권장)
- [uv](https://docs.astral.sh/uv/)

## 설치

```bash
uv sync
```

## 환경 변수

`.env.example`을 복사해 `.env`를 생성하고 필요에 따라 수정합니다.

```bash
cp .env.example .env
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_NAME` | `deepseek-ai/DeepSeek-OCR` | HuggingFace 모델명 |
| `NUM_GPUS` | `1` | 사용할 GPU 수 (병렬 처리) |
| `DPI` | `200` | PDF 렌더링 해상도 |
| `BASE_SIZE` | `1024` | 모델 base_size 파라미터 |
| `IMAGE_SIZE` | `640` | 모델 image_size 파라미터 |
| `CROP_MODE` | `true` | crop 모드 활성화 여부 |
| `OUTPUT_DIR` | `./output` | 결과 파일 저장 경로 |

## 다중 GPU 병렬 처리

`.env`에서 `NUM_GPUS`를 GPU 수에 맞게 설정하면 페이지를 병렬로 처리합니다.

```env
NUM_GPUS=2
```

**동작 방식 (NUM_GPUS=2):**

```
GPU 0: 페이지 0, 2, 4, 6, ...
GPU 1: 페이지 1, 3, 5, 7, ...
         ↓ 처리 완료 후 페이지 순서대로 재정렬
       output/{날짜}_{파일명}_ocr.md
```

- 각 GPU에 모델이 독립적으로 로드됩니다 (GPU당 모델 1벌)
- RTX 3090 24GB 기준 GPU당 약 14GB 사용 (bfloat16)

## 실행

```bash
python pdf.py <PDF_파일_경로>
```

**예시:**

```bash
python pdf.py tiger_kr.pdf
```

결과 파일은 `OUTPUT_DIR`에 `{YYYYMMDD}_{파일명}_ocr.md` 형식으로 저장됩니다.

## 주요 파일

| 파일 | 설명 |
|---|---|
| `pdf.py` | PDF OCR 처리 메인 스크립트 |
| `.env` | 환경 변수 설정 (git 제외) |
| `.env.example` | 환경 변수 예시 템플릿 |
| `pyproject.toml` | 프로젝트 의존성 및 설정 |

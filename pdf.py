import argparse
import logging
import os
import re
from datetime import datetime

import fitz  # PyMuPDF
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

from notify import send_mail

NOTIFY_EMAIL = "bomsan69@gmail.com"
LOG_DIR = "logs"

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
DPI = int(os.getenv("DPI", "200"))
BASE_SIZE = int(os.getenv("BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("CROP_MODE", "true").lower() == "true"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")


def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d')}_ocr.log")

    logger = logging.getLogger("deepseek_ocr")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def clean_ocr_result(text):
    pattern = r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>\n?'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    cleaned = cleaned.replace('\\coloneqq', ':=')
    cleaned = cleaned.replace('\\eqqcolon', '=:')
    cleaned = re.sub(r'\n{4,}', '\n\n', cleaned)
    return cleaned.strip()


def process_pdf(pdf_path: str, logger: logging.Logger) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf = fitz.open(pdf_path)
    total_pages = len(pdf)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

    logger.info(f"PDF 처리 시작: {pdf_basename} ({total_pages} 페이지)")
    logger.info(f"모델 로딩 중: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    logger.info("모델 로딩 완료, 페이지 처리 시작")

    results = []
    for page_num in range(total_pages):
        logger.info(f"페이지 {page_num + 1}/{total_pages} 처리 중...")

        pix = pdf[page_num].get_pixmap(dpi=DPI)
        img_path = f"page_{page_num}.png"
        pix.save(img_path)

        try:
            result = model.infer(
                tokenizer,
                prompt="<image>\n<|grounding|>Convert the document to markdown.",
                image_file=img_path,
                output_path=OUTPUT_DIR,
                base_size=BASE_SIZE,
                image_size=IMAGE_SIZE,
                crop_mode=CROP_MODE,
                eval_mode=True,
                save_results=False,
            )

            if result:
                results.append((page_num, clean_ocr_result(result)))
                logger.debug(f"페이지 {page_num + 1} 완료")
            else:
                logger.warning(f"페이지 {page_num + 1} OCR 결과 없음")
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    pdf.close()

    logger.info("모든 페이지 처리 완료. 결과 파일 생성 중...")

    results.sort(key=lambda x: x[0])
    texts = [text for _, text in results]

    today = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(OUTPUT_DIR, f"{today}_{pdf_basename}_ocr.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(texts))

    logger.info(f"완료! 결과 파일: {output_path} (총 {len(results)}/{total_pages} 페이지)")

    send_mail(
        to=NOTIFY_EMAIL,
        title=f"[DeepSeek OCR] {pdf_basename} 처리 완료",
        message=(
            f"OCR 작업이 완료되었습니다.\n\n"
            f"파일: {pdf_basename}\n"
            f"처리 페이지: {len(results)} / {total_pages}\n"
            f"결과 파일: {output_path}\n"
            f"완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="DeepSeek OCR - PDF to Markdown")
    parser.add_argument("pdf_path", help="OCR 처리할 PDF 파일 경로")
    args = parser.parse_args()

    logger = setup_logging()

    if not os.path.exists(args.pdf_path):
        logger.error(f"파일을 찾을 수 없습니다: {args.pdf_path}")
        raise SystemExit(1)

    try:
        process_pdf(args.pdf_path, logger)
    except Exception as e:
        logger.exception(f"OCR 처리 중 오류 발생: {e}")
        send_mail(
            to=NOTIFY_EMAIL,
            title=f"[DeepSeek OCR] {os.path.basename(args.pdf_path)} 처리 실패",
            message=(
                f"OCR 작업 중 오류가 발생했습니다.\n\n"
                f"파일: {args.pdf_path}\n"
                f"오류: {e}\n"
                f"발생 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
        )
        raise


if __name__ == "__main__":
    main()

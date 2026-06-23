import argparse
import logging
import os
import re
from datetime import datetime

import fitz  # PyMuPDF
import torch
import torch.multiprocessing as mp
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
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))


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


def gpu_worker(gpu_id: int, page_nums: list, pdf_path: str, result_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 워커 프로세스는 별도 프로세스라 logger를 새로 설정
    logger = setup_logging()
    logger.info(f"[GPU {gpu_id}] 모델 로딩 중: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    logger.info(f"[GPU {gpu_id}] 모델 로딩 완료, {len(page_nums)}개 페이지 처리 시작")

    pdf = fitz.open(pdf_path)

    for page_num in page_nums:
        logger.info(f"[GPU {gpu_id}] 페이지 {page_num + 1} 처리 중...")

        pix = pdf[page_num].get_pixmap(dpi=DPI)
        img_path = f"page_gpu{gpu_id}_{page_num}.png"
        pix.save(img_path)

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
            result_queue.put((page_num, clean_ocr_result(result)))
            logger.debug(f"[GPU {gpu_id}] 페이지 {page_num + 1} 완료")
        else:
            logger.warning(f"[GPU {gpu_id}] 페이지 {page_num + 1} OCR 결과 없음")

        if os.path.exists(img_path):
            os.remove(img_path)

    pdf.close()
    logger.info(f"[GPU {gpu_id}] 모든 페이지 처리 완료")


def process_pdf(pdf_path: str, logger: logging.Logger) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf = fitz.open(pdf_path)
    total_pages = len(pdf)
    pdf.close()

    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info(f"PDF 처리 시작: {pdf_basename} ({total_pages} 페이지, GPU {NUM_GPUS}개 사용)")

    result_queue = mp.Queue()

    if NUM_GPUS == 1:
        gpu_worker(0, list(range(total_pages)), pdf_path, result_queue)
    else:
        page_groups = [list(range(i, total_pages, NUM_GPUS)) for i in range(NUM_GPUS)]

        processes = [
            mp.Process(target=gpu_worker, args=(gpu_id, pages, pdf_path, result_queue))
            for gpu_id, pages in enumerate(page_groups)
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    raw = {}
    while not result_queue.empty():
        page_num, text = result_queue.get()
        raw[page_num] = text

    results = [raw[i] for i in range(total_pages) if i in raw]

    logger.info("모든 페이지 처리 완료. 결과 파일 생성 중...")
    today = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(OUTPUT_DIR, f"{today}_{pdf_basename}_ocr.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(results))

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
    mp.set_start_method("spawn")
    main()

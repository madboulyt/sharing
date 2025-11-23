import datetime
import logging
import os
from uuid import uuid4
import aiofiles
import httpx
from fastapi import Body, FastAPI, File, UploadFile, status
from schemas.pipeline import SubmitRequestRequest, SubmitRequestResponse
from worker.worker import broker, run_ocr_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="OCR Pipeline Server",
    description="""
            PDF processing pipeline using existing OCR model endpoints.
            This API provides endpoints for:
            * Processing PDF documents by:
            * Layout detection
            * Table detection and extraction for:
            * - Caption,
            * - Footnote,
            * - Formula,
            * - List-item,
            * - Page-footer,
            * - Page-header,
            * - Picture,
            * - Section-header,
            * - Table,
            * - Text,
            * - Title,
            * Line detection
            * Text recognition
            """,
    version="1.0.0",
    docs_url="/docs",
)

PDFS_PATH = os.getenv("PDFS_PATH", "/home/moath/osos-ingest-serving/output/pdfs")


@app.on_event("startup")
async def startup_event():
    await broker.startup()


@app.on_event("shutdown")
async def shutdown_event():
    await broker.shutdown()


@app.post(
    "/process-pdf",
    response_model=SubmitRequestResponse,
    status_code=status.HTTP_200_OK,
    tags=["PDF Processing"],
    summary="Submit a request to process a PDF file through the OCR pipeline",
)
async def submit_request_to_process_pdf(
    file_metadata: SubmitRequestRequest = Body(...),
    file: UploadFile = File(..., description="PDF file to process"),
):
    file_path = f"{PDFS_PATH}/{file.filename}"
    async with aiofiles.open(file_path, "wb") as w_file:
        while content := await file.read(1024):
            await w_file.write(content)
    # Generate unique ID here so we can return it immediately
    unique_id = str(uuid4())
    task = await run_ocr_pipeline.kiq(
        file_path=file_path, page_numbers=file_metadata.page_numbers, unique_id=unique_id
    )

    return {"request_id": task.task_id,"output_data_folder": unique_id}

async def task_exists(task_id: str) -> bool:
    """Check if a task_id exists in the result backend"""
    try:
        result = await broker.result_backend.get_result(task_id)
        return result is not None
    except Exception:
        return False

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    if not await task_exists(task_id):
        return {"task_id": task_id, "is_done": False, "message": "Task not found"}
        
    ready = await broker.result_backend.is_result_ready(task_id)
    if not ready:
        return {"task_id": task_id, "is_done": False}

    result = await broker.result_backend.get_result(task_id)

    return {
        "task_id": task_id,
        "is_done": True,
        "result": result.return_value,
        "execution_time": result.execution_time,
        "is_error": getattr(result, "is_err", False),
        "error": result.serialize_error(result.error),
    }


@app.get(
    "/",
    tags=["System"],
    summary="Get API information",
    response_description="Basic information about the API",
)
async def root():
    models_server_url = os.getenv("MODELS_SERVER_URL")
    return {
        "service": "OCR Pipeline Server",
        "version": "1.0.0",
        "endpoints": {
            "process_pdf": "/process-pdf",
            "task_status": "/task-status/{task_id}",
            "health": "/health",
        },
        "models_server_url": models_server_url,
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    response_description="Health status of the service",
)
async def health():
    models_server_url = os.getenv("MODELS_SERVER_URL")
    try:
        async with httpx.AsyncClient() as client:
            models_health = await client.get(f"{models_server_url}/health")
            models_status = (
                "healthy" if models_health.status_code == 200 else "unhealthy"
            )

        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "models_server": {"url": models_server_url, "status": models_status},
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e),
            "models_server": {"url": models_server_url, "status": "unreachable"},
        }

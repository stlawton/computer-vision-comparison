import time
import threading
import psutil

from pathlib import Path
from datetime import datetime, timezone

def cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000


def benchmark(image_paths, load_image, detection_pipeline, warmups=5, method=None):

    image_paths = list(image_paths)

    if not image_paths:
        raise ValueError("No images provided for benchmarking.")

    run_id = (f"{method}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}")

    # Warm up before collecting measurements.
    for _ in range(warmups):
        image = load_image(image_paths[0])

        if image is None:
            raise ValueError(f"Failed to load: {image_paths[0]}.")
        
        detection_pipeline(image)

    process = psutil.Process()

    latency_samples = []
    resource_samples = []
    prediction_samples = []
    monitor_errors = []

    stop = threading.Event()
    ready = threading.Event()
    started = threading.Event()

    sample_interval = 0.1


    def monitor():
        try:
            process.cpu_percent(interval=None)
            ready.set()
            started.wait()

            while not stop.wait(sample_interval):

                timestamp = time.perf_counter_ns()

                resource_samples.append({
                    "run_id": run_id,
                    "sample_id": len(resource_samples) + 1,
                    "timestamp_ns": timestamp,
                    "cpu_percent": process.cpu_percent(interval=None),
                    "memory_rss_bytes": process.memory_info().rss,
                    "temperature_c": cpu_temperature()
                })

        except Exception as exc:
            monitor_errors.append(exc)
            ready.set()

    monitor_thread = threading.Thread(target=monitor, daemon=True)

    monitor_thread.start()
    ready.wait()

    if monitor_errors:
        stop.set()
        started.set()
        monitor_thread.join()
        raise RuntimeError(
            "Resource monitor failed to initialize"
        ) from monitor_errors[0]

    cpu_start = process.cpu_times()
    run_start = time.perf_counter_ns()
    started.set()

    try:
        for execution_id, image_path in enumerate(
            image_paths, start=1
        ):

            load_start = time.perf_counter_ns()

            image = load_image(image_path)

            if image is None:
                raise ValueError(f"Cannot load: {image_path}")

            processing_start = time.perf_counter_ns()

            predictions = detection_pipeline(image)

            processing_end = time.perf_counter_ns()

            if not isinstance(predictions, list):
                raise TypeError(
                    "Detection pipeline must return "
                    "a list of prediction dictionaries."
                )

            image_id = Path(image_path).stem

            # Record latency for this execution
            latency_samples.append({
                "run_id": run_id,
                "execution_id": execution_id,
                "image_id": image_id,
                "image_path": str(image_path),
                "load_start_ns": load_start,
                "processing_start_ns": processing_start,
                "processing_end_ns": processing_end,
                "processing_ms": (
                    processing_end - processing_start
                ) / 1_000_000,
                "stored_image_end_to_end_ms": (
                    processing_end - load_start
                ) / 1_000_000,
                "detection_count": len(predictions)
            })

            # Save every predicted bounding box
            for prediction_id, prediction in enumerate(
                predictions, start=1
            ):

                x1, y1, x2, y2 = prediction["bbox"]

                prediction_samples.append({
                    "run_id": run_id,
                    "execution_id": execution_id,
                    "image_id": image_id,
                    "prediction_id": prediction_id,
                    "pred_x1": float(x1),
                    "pred_y1": float(y1),
                    "pred_x2": float(x2),
                    "pred_y2": float(y2),
                    "class_id": prediction.get("class_id"),
                    "class_name": prediction.get("class_name"),
                    "confidence": prediction.get("confidence")
                })

        run_end = time.perf_counter_ns()
        cpu_end = process.cpu_times()

    finally:
        stop.set()
        started.set()
        monitor_thread.join()

    if monitor_errors:
        raise RuntimeError(
            "Resource monitor failed during benchmark"
        ) from monitor_errors[0]

    run_info = {
        "run_id": run_id,
        "method": method,
        "run_start_ns": run_start,
        "run_end_ns": run_end,
        "warmups": warmups,
        "sample_interval_s": sample_interval,
        "image_count": len(image_paths),
        "cpu_user_start_s": cpu_start.user,
        "cpu_user_end_s": cpu_end.user,
        "cpu_system_start_s": cpu_start.system,
        "cpu_system_end_s": cpu_end.system,
        "run_elapsed_ms": (
            run_end - run_start
        ) / 1_000_000
    }

    return {
        "latency": latency_samples,
        "resources": resource_samples,
        "predictions": prediction_samples,
        "run_info": run_info,
        "frame_count": len(latency_samples),
        "run_elapsed_ms": run_info["run_elapsed_ms"]
    }


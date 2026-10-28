
import time
import threading
import psutil


def cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000


def benchmark(image_paths, load_image, detection_pipeline, warmups=5):

    image_paths = list(image_paths)

    if not image_paths:
        raise ValueError("No images provided for benchmarking.")

    # Warm up before collecting measurements.
    for _ in range(warmups):
        image = load_image(image_paths[0])
        detection_pipeline(image)

    process = psutil.Process()

    latency_samples = []
    resource_samples = []

    stop = threading.Event()
    ready = threading.Event()

    start_reference = time.perf_counter_ns()

    def monitor():
        process.cpu_percent(interval=None)
        ready.set()

        while not stop.wait(0.05):

            resource_samples.append({
                "elapsed_seconds": (time.perf_counter_ns() - start_reference) / 1_000_000_000,
                "cpu_percent": process.cpu_percent(interval=None),
                "memory_mb": process.memory_info().rss / 1_000_000,
                "temperature_c": cpu_temperature()
            })

    monitor_thread = threading.Thread(
        target=monitor, daemon=True
    )

    monitor_thread.start()
    ready.wait()

    try:
        run_start = time.perf_counter_ns()

        for image_path in image_paths:

            start = time.perf_counter_ns()

            image = load_image(image_path)

            processing_start = time.perf_counter_ns()

            detection_pipeline(image)

            processing_end = time.perf_counter_ns()

            latency_samples.append({
                "image": str(image_path),
                "processing_ms": (processing_end - processing_start) / 1_000_000,
                "stored_image_end_to_end_ms": (processing_end - start) / 1_000_000
            })

        run_end = time.perf_counter_ns()

    finally:
        stop.set()
        monitor_thread.join()

    return {
        "latency": latency_samples,
        "resources": resource_samples,
        "frame_count": len(latency_samples),
        "run_elapsed_ms": (run_end - run_start) / 1_000_000
    }

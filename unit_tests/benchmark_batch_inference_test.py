from pathlib import Path

import benchmark_batch_inference


def test_write_csv_writes_benchmark_rows(tmp_path):
    rows = [
        {
            "batch_size": 1,
            "total_samples": 10,
            "total_time_s": 0.1,
            "latency_ms_per_image": 10.0,
            "throughput_images_per_s": 100.0,
        },
        {
            "batch_size": 4,
            "total_samples": 10,
            "total_time_s": 0.04,
            "latency_ms_per_image": 4.0,
            "throughput_images_per_s": 250.0,
        },
    ]
    output_path = tmp_path / "benchmark.csv"

    benchmark_batch_inference.write_csv(rows, output_path)

    csv_text = Path(output_path).read_text(encoding="utf-8")
    assert "batch_size,total_samples,total_time_s,latency_ms_per_image,throughput_images_per_s" in csv_text
    assert "4,10,0.04,4.0,250.0" in csv_text

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

MODEL_ERRORS = Counter(
    "model_errors_total",
    "Total model errors",
    ["type"],
)

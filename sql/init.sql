CREATE TABLE IF NOT EXISTS inference_requests (
    id UUID PRIMARY KEY,
    request_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_path TEXT,
    content_type TEXT,
    input_bytes INTEGER,
    latency_ms INTEGER,
    status TEXT NOT NULL,
    predictions JSONB
);

CREATE INDEX IF NOT EXISTS idx_inference_requests_request_id ON inference_requests (request_id);
CREATE INDEX IF NOT EXISTS idx_inference_requests_created_at ON inference_requests (created_at DESC);

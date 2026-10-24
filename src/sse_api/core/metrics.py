"""
In-process metrics of the API, rendered in the Prometheus text format.

Deliberately dependency-free: the deployment already runs nginx, PostgreSQL
and Milvus, and an external time series database would be a second thing to
operate.  A pull endpoint over a lock-protected registry is enough to answer
the two operational questions that matter -- how fast is the API, and is it
returning errors -- and it degrades gracefully (a restart loses the window,
which is what the scrape interval is for).

Two series are kept:

* ``sse_api_requests_total`` (counter, labelled by method/path/status)
* ``sse_api_request_duration_seconds`` (histogram, unlabelled, so the
  cardinality of the path labels cannot multiply into the buckets)

The number of distinct label combinations is capped: the path comes from the
request, and without a cap a client probing random URLs would grow the
registry (and the body of every later scrape) without end.
"""

import threading
import time

#: Distinct ``(method, path, status)`` combinations kept.
MAX_SERIES = 500

#: Histogram bucket edges, in seconds.  Chosen around the latency of a real
#: search: a vector-only query is tens of milliseconds, a hybrid search with
#: a reranker and a generated answer is tens of seconds.
BUCKET_EDGES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

_LOCK = threading.Lock()

#: {(method, path, status): count}
_series: dict = {}
#: cumulative buckets: {edge: count}, plus "+Inf"
_buckets: dict = {}
_duration_count = 0
_duration_sum = 0.0

#: When the process started, reported so a scrape can compute a rate over the
#: whole lifetime even when the counters were reset.
STARTED_AT = time.time()


def record_request(method: str, path: str, status: int, duration: float) -> None:
    """
    Account for one request.

    ``duration`` is in seconds.  A negative duration (a clock going backwards)
    is kept out of the histogram sum but still counted.
    """
    global _duration_count, _duration_sum

    key = (method or "-", path or "-", int(status or 0))
    with _LOCK:
        if key in _series or len(_series) < MAX_SERIES:
            _series[key] = _series.get(key, 0) + 1
        _duration_count += 1
        if duration is not None and duration >= 0:
            _duration_sum += duration
        for edge in BUCKET_EDGES:
            if duration is not None and duration <= edge:
                _buckets[edge] = _buckets.get(edge, 0) + 1
        _buckets["+Inf"] = _buckets.get("+Inf", 0) + 1


def reset() -> None:
    """Forget every counter (tests)."""
    global _duration_count, _duration_sum
    with _LOCK:
        _series.clear()
        _buckets.clear()
        _duration_count = 0
        _duration_sum = 0.0


def _labelled(name: str, labels: str, value) -> str:
    return f"{name}{{{labels}}} {value}" if labels else f"{name} {value}"


def render() -> str:
    """The whole registry as the Prometheus 0.0.4 text format."""
    lines = []
    with _LOCK:
        series = dict(_series)
        buckets = dict(_buckets)
        count = _duration_count
        total = _duration_sum

    if series:
        lines.append("# TYPE sse_api_requests_total counter")
        for (method, path, status), value in sorted(series.items()):
            labels = f'method="{method}",path="{path}",status="{status}"'
            lines.append(_labelled("sse_api_requests_total", labels, float(value)))

    if count:
        lines.append("# TYPE sse_api_request_duration_seconds histogram")
        running = 0
        for edge in BUCKET_EDGES:
            running = buckets.get(edge, 0)
            lines.append(
                f"sse_api_request_duration_seconds_bucket{{le=\"{_format_le(edge)}\"}} "
                f"{float(running)}"
            )
        lines.append(
            "sse_api_request_duration_seconds_bucket{le=\"+Inf\"} "
            f"{float(buckets.get('+Inf', 0))}"
        )
        lines.append(f"sse_api_request_duration_seconds_count {float(count)}")
        lines.append(f"sse_api_request_duration_seconds_sum {total}")
        lines.append(
            "# TYPE sse_api_process_start_time_seconds gauge"
        )
        lines.append(f"sse_api_process_start_time_seconds {STARTED_AT}")

    return "\n".join(lines) + "\n" if lines else ""


def _format_le(edge: float) -> str:
    return repr(float(edge))


class MetricsMiddleware:
    """
    Records the duration and the status code of every request.

    Installed as a Django middleware; it never touches the response, so a
    failure of the accounting itself cannot break a request.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        started = time.monotonic()
        try:
            response = self.get_response(request)
        except Exception:
            record_request(
                request.method, request.path, 500, time.monotonic() - started
            )
            raise
        record_request(
            request.method, request.path, response.status_code,
            time.monotonic() - started,
        )
        return response

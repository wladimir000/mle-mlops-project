from time import perf_counter

from fastapi import Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

REQUEST_COUNT = Counter(
    "taxi_api_requests_total",
    "Total HTTP requests processed by the Taxi API",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "taxi_api_request_duration_seconds",
    "Taxi API request duration in seconds",
    ["method", "endpoint"],
)


def install_metrics(app):
    """Install Prometheus endpoint and request instrumentation."""

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = perf_counter() - start

            route = request.scope.get("route")
            endpoint = (
                route.path
                if route is not None
                else "unmatched"
            )

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=str(status_code),
            ).inc()

            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

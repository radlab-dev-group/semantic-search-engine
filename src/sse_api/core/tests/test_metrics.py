"""
Tests of the in-process metrics registry and of the ``/metrics`` endpoint.

No external dependency: the registry is a plain, lock-protected set of
counters and histogram buckets rendered in the Prometheus text format.  The
tests cover the accounting (a request is counted once, under the right
labels), the cumulative bucket edges, the cap on distinct series (so a
fuzzer cannot grow the registry unboundedly) and the endpoint itself.
"""

from django.test import SimpleTestCase
from rest_framework.permissions import AllowAny
from rest_framework.test import APIClient

from sse_api.core import metrics as m


class RegistryTests(SimpleTestCase):
    def setUp(self):
        m.reset()

    def test_request_is_counted_once_with_labels(self):
        m.record_request("GET", "/api/search", 200, 0.42)
        m.record_request("GET", "/api/search", 200, 0.11)
        m.record_request("POST", "/api/search", 500, 1.5)

        rendered = m.render()
        self.assertIn(
            'sse_api_requests_total{method="GET",path="/api/search",'
            'status="200"} 2.0',
            rendered,
        )
        self.assertIn(
            'sse_api_requests_total{method="POST",path="/api/search",'
            'status="500"} 1.0',
            rendered,
        )

    def test_histogram_buckets_are_cumulative(self):
        m.record_request("GET", "/api/x", 200, 0.05)
        m.record_request("GET", "/api/x", 200, 0.5)
        m.record_request("GET", "/api/x", 200, 5.0)
        m.record_request("GET", "/api/x", 200, 120.0)

        rendered = m.render()
        self.assertIn('sse_api_request_duration_seconds_count 4.0', rendered)
        self.assertIn('sse_api_request_duration_seconds_sum 125.55', rendered)
        self.assertIn(
            'sse_api_request_duration_seconds_bucket{le="0.1"} 1.0', rendered
        )
        self.assertIn(
            'sse_api_request_duration_seconds_bucket{le="1.0"} 2.0', rendered
        )
        self.assertIn(
            'sse_api_request_duration_seconds_bucket{le="10.0"} 3.0', rendered
        )
        self.assertIn(
            'sse_api_request_duration_seconds_bucket{le="+Inf"} 4.0', rendered
        )

    def test_distinct_series_are_capped(self):
        for i in range(m.MAX_SERIES + 10):
            m.record_request("GET", f"/fuzz/{i}", 200, 0.01)
        self.assertLessEqual(len(m._series), m.MAX_SERIES)

    def test_reset_clears_everything(self):
        m.record_request("GET", "/api/x", 200, 0.1)
        m.reset()
        self.assertEqual(len(m._series), 0)
        self.assertNotIn("sse_api_requests_total", m.render())


class MetricsEndpointTests(SimpleTestCase):
    def setUp(self):
        m.reset()
        self.client = APIClient()

    def test_metrics_endpoint_is_public_and_prometheus(self):
        m.record_request("GET", "/api/anything", 200, 0.01)
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response["Content-Type"].startswith("text/plain")
        )
        body = response.content.decode()
        self.assertIn("# TYPE sse_api_requests_total counter", body)
        self.assertIn(
            "# TYPE sse_api_request_duration_seconds histogram", body
        )

    def test_metrics_view_opts_out_of_auth_and_throttling(self):
        from sse_api.system.api import SSEMetrics

        self.assertEqual(SSEMetrics.authentication_classes, [])
        self.assertEqual(SSEMetrics.permission_classes, [AllowAny])
        self.assertEqual(SSEMetrics.throttle_classes, [])

"""
Tests of the SSRF guard of the WWW content grabber.

The content supervisor follows every URL it finds in a user-supplied chat
message, so the guard is the only thing standing between a user and the
internal network / the cloud metadata endpoint.  These tests check the pure
policy (``is_safe_url``) without touching the network.
"""

from urllib.parse import urlsplit

from django.test import SimpleTestCase

from sse_api.content_supervisor.executors.www_content_grabber import (
    WWWContentGrabberExecutor,
    UnsafeUrl,
    is_safe_url,
)


class IsSafeUrlTests(SimpleTestCase):
    def test_public_http_urls_are_allowed(self):
        self.assertTrue(is_safe_url("https://example.com/page"))
        self.assertTrue(is_safe_url("http://example.com:8080/a?b=c"))
        self.assertTrue(is_safe_url("http://example.com:443/x.html"))

    def test_public_ip_literals_are_allowed(self):
        self.assertTrue(is_safe_url("http://93.184.216.34/"))
        self.assertTrue(is_safe_url("https://[2606:2800:220:1:248:1893:25c8:1946]/"))

    def test_cloud_metadata_address_is_rejected(self):
        for host in ("169.254.169.254", "fd00:ec2::254"):
            self.assertFalse(is_safe_url(f"http://{host}/latest/meta-data/"))

    def test_link_local_and_loopback_are_rejected(self):
        for host in (
            "127.0.0.1",
            "127.0.0.53",
            "10.0.0.7",
            "192.168.1.10",
            "172.16.0.1",
            "172.31.255.255",
            "100.64.0.1",
            "100.127.255.255",
            "198.18.0.1",
            "198.51.100.9",
            "192.0.2.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fc00::1",
            "fe00::1",
        ):
            self.assertFalse(is_safe_url(f"http://{host}/"), host)

    def test_non_http_schemes_are_rejected(self):
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/file",
            "gopher://example.com/",
            "dict://example.com:2628/",
            "javascript:alert(1)",
        ):
            self.assertFalse(is_safe_url(url), url)

    def test_unparsable_or_empty_urls_are_rejected(self):
        self.assertFalse(is_safe_url("not a url"))
        self.assertFalse(is_safe_url("https://"))
        self.assertFalse(is_safe_url(""))

    def test_credentials_are_stripped_from_the_url(self):
        cleaned = WWWContentGrabberExecutor.sanitize_url(
            "https://user:secret@example.com/path"
        )
        parts = urlsplit(cleaned)
        self.assertIsNone(parts.password)
        self.assertEqual(parts.hostname, "example.com")

    def test_trailing_punctuation_is_stripped(self):
        cleaned = WWWContentGrabberExecutor.sanitize_url(
            "https://example.com/page."
        )
        self.assertEqual(cleaned, "https://example.com/page")

    def test_rejection_raises_unsafe_url(self):
        with self.assertRaises(UnsafeUrl):
            WWWContentGrabberExecutor.prepare_download(
                "http://169.254.169.254/latest/meta-data/"
            )
        with self.assertRaises(UnsafeUrl):
            WWWContentGrabberExecutor.prepare_download("file:///etc/passwd")

    def test_preparation_passes_safe_url_through(self):
        prepared = WWWContentGrabberExecutor.prepare_download(
            "https://example.com/page?"
        )
        self.assertEqual(prepared, "https://example.com/page")

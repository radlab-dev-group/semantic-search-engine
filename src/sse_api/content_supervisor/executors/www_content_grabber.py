import ipaddress
import socket
import urllib.error
import urllib.request
from typing import Dict, Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from sse_api.content_supervisor.executors.executor import ExecutorI


class UnsafeUrl(ValueError):
    """
    Raised when a URL fails the SSRF guard: a disallowed scheme, an
    unresolvable host, or an address in a private / reserved range.
    """


ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 10.0
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024


def _address_is_unsafe(address: Any) -> bool:
    """
    True for every address which must never be reached from a user-supplied
    URL: loopback, private, CGNAT, link-local (cloud metadata), reserved,
    documentation, multicast and unspecified ranges.
    """
    try:
        if isinstance(address, str):
            address = ipaddress.ip_address(address)
        if isinstance(address, tuple):
            address = ipaddress.ip_address(address[0])
    except ValueError:
        return True
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        or _is_cgnat(address)
        or _is_documentation_range(address)
    )


def _is_cgnat(address) -> bool:
    # 100.64.0.0/10 (carrier-grade NAT) is not flagged by the stdlib.
    return address.version == 4 and ipaddress.ip_address(address) in ipaddress.ip_network(
        "100.64.0.0/10"
    )


def _is_documentation_range(address) -> bool:
    # 192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24 (TEST-NET-1..3).
    if address.version != 4:
        return False
    return address in ipaddress.ip_network("192.0.2.0/24") or address in ipaddress.ip_network(
        "198.51.100.0/24"
    ) or address in ipaddress.ip_network("203.0.113.0/24")


def _resolved_addresses(hostname: str) -> list:
    """
    Every address the hostname resolves to; an unresolvable host is unsafe
    (fail closed) instead of being passed to urlopen with a generic error.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, OSError, UnicodeError):
        return []
    return [info[4][0] for info in infos]


def _url_addresses(url: str) -> list:
    """
    The addresses a URL would be sent to: the literal address when the host
    is one, the resolved addresses otherwise.  Checking the resolved set is
    what defeats a DNS name pointing at 169.254.169.254 or a LAN host.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not host:
        return []
    try:
        ipaddress.ip_address(host)
        return [host]
    except ValueError:
        return _resolved_addresses(host)


def is_safe_url(url: str) -> bool:
    """
    SSRF policy of the content grabber.

    A URL is safe when its scheme is http/https and every address it could be
    sent to is a public one.  Anything else (file://, ftp://, an unresolvable
    host, loopback, RFC1918, CGNAT, link-local metadata, ...) is rejected.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ALLOWED_SCHEMES or not parts.hostname:
        return False
    addresses = _url_addresses(url)
    if not addresses:
        return False
    return not any(_address_is_unsafe(address) for address in addresses)


class WWWContentGrabberExecutor(ExecutorI):
    EXECUTOR_NAME = "ContentGrabberExecutor"
    MOZILLA_HEADER = {
        "User-Agent": " Mozilla/5.0 (Windows NT 6.1; WOW64; rv:12.0) Gecko/20100101 Firefox/12.0"
    }

    def __init__(
        self,
        options: Dict | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
    ):
        super(WWWContentGrabberExecutor, self).__init__(
            name_str=self.EXECUTOR_NAME, options=options
        )
        self._connect_timeout = float(connect_timeout)
        self._read_timeout = float(read_timeout)
        self._max_bytes = int(max_bytes)

    def apply(self, body) -> Dict[Any, Any]:
        urls_content = {}
        if type(body) is not list:
            body = [body]
        for url_to_grab_content in body:
            urls_content[url_to_grab_content] = self.__grab_url_content(
                url=url_to_grab_content
            )
        return urls_content

    def __grab_url_content(self, url: str) -> str:
        try:
            safe_url = self.prepare_download(url)
        except UnsafeUrl:
            self._logger().warning("Refusing to fetch unsafe URL: %r", url)
            return ""

        try:
            req = urllib.request.Request(
                url=safe_url, headers=WWWContentGrabberExecutor.MOZILLA_HEADER
            )
            with urllib.request.urlopen(
                req, timeout=self._read_timeout
            ) as response:
                return self.__parse_url_content(
                    url_content=response.read(self._max_bytes + 1)
                )
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._logger().error("Error downloading content from %s: %s", safe_url, e)
            return ""

    @staticmethod
    def _logger():
        import logging

        return logging.getLogger("sse_api")

    @classmethod
    def prepare_download(cls, url: str) -> str:
        """
        Validate and normalise a URL before it leaves the process.

        Raises :class:`UnsafeUrl` when the URL must not be fetched; otherwise
        returns the cleaned URL (trailing sentence punctuation removed,
        credentials stripped).
        """
        cleaned = cls.sanitize_url(url)
        if not is_safe_url(cleaned):
            raise UnsafeUrl(f"URL refused by the SSRF guard: {cleaned!r}")
        return cleaned

    @staticmethod
    def sanitize_url(url: str) -> str:
        remove_from_end = [".", ",", ";", "?", "!", " "]
        for r in remove_from_end:
            url = url.strip(r)
        parts = urlsplit(url)
        if parts.username or parts.password:
            netloc = parts.hostname or ""
            if parts.port:
                netloc += f":{parts.port}"
            url = parts._replace(netloc=netloc).geturl()
        return url

    @staticmethod
    def __parse_url_content(url_content: bytes) -> str:
        if not len(url_content.strip()):
            return ""

        raw_text = ""
        soup = BeautifulSoup(url_content, features="lxml")
        for p in soup.find_all("p"):
            p_text = p.text.strip()
            if not len(p_text):
                continue
            raw_text += p_text + "\n"
        return raw_text.strip()

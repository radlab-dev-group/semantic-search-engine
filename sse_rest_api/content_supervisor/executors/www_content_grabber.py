import urllib.request
from typing import Dict, Any
from bs4 import BeautifulSoup

from content_supervisor.executors.executor import ExecutorI


class WWWContentGrabberExecutor(ExecutorI):
    EXECUTOR_NAME = "ContentGrabberExecutor"
    MOZILLA_HEADER = {
        "User-Agent": " Mozilla/5.0 (Windows NT 6.1; WOW64; rv:12.0) Gecko/20100101 Firefox/12.0"
    }

    def __init__(self, options: Dict | None = None):
        super(WWWContentGrabberExecutor, self).__init__(
            name_str=self.EXECUTOR_NAME, options=options
        )

    def apply(self, body) -> Dict[Any, Any]:
        urls_content = {}
        if type(body) is not list:
            body = [body]
        for url_to_grab_content in body:
            url_content = self.__download_url_content(url=url_to_grab_content)
            url_parsed_content = self.__parse_url_content(url_content=url_content)

            urls_content[url_to_grab_content] = url_parsed_content
        return urls_content

    def __download_url_content(self, url: str) -> str:
        url = self.__clear__url(url=url)
        try:
            req = urllib.request.Request(
                url=url, headers=WWWContentGrabberExecutor.MOZILLA_HEADER
            )
            with urllib.request.urlopen(req) as response:
                return response.read()
        except Exception as e:
            print(f"Error downloading content from {url}: {e}")
            return ""

    @staticmethod
    def __clear__url(url: str) -> str:
        remove_from_end = [".", ",", ";", "?", "!", " "]
        for r in remove_from_end:
            url = url.strip(r)
        return url

    @staticmethod
    def __parse_url_content(url_content: str) -> str:
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

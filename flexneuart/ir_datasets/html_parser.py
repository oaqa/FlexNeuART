#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
    This processor parses HTML using the following fields (IR datasets specific):
    body, body_content_type, http_headers


    You need to call configure_classpath() before using this functionality.
"""
from jnius import autoclass

from flexneuart.config import MAX_DOC_SIZE
from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.ir_datasets import register

from flexneuart.io.json import get_val_err_msg_miss

JHtmlParser = autoclass('edu.cmu.lti.oaqa.flexneuart.utils.HTMLParser')

OUTPUT_FIELD_BODY = 'body'
OUTPUT_FIELD_TITLE = 'title'

DECODE_ERROR_HANDLING = 'replace'

@register('html_parser')
class HtmlParserProcessor(BaseTextProcessor):

    def __init__(self, max_doc_size=MAX_DOC_SIZE):
        self.max_doc_size = max_doc_size

    def __call__(self, input_dict: dict):
        output_dict = {OUTPUT_FIELD_BODY : '', OUTPUT_FIELD_TITLE : ''}

        body = get_val_err_msg_miss(input_dict, 'body', [bytes])
        body_content_type = get_val_err_msg_miss(input_dict, 'body_content_type', [str])
        http_headers = get_val_err_msg_miss(input_dict, 'http_headers', [bytes, str])
        # A bit hacky b/c IR datasets is not very consistent regarding the type of HTTP headers produced
        if type(http_headers) == bytes:
            http_headers = http_headers.decode(errors=DECODE_ERROR_HANDLING)

        # TODO how this decoding would work for non-English data?
        body_to_proc = body.decode(errors=DECODE_ERROR_HANDLING)
        body_to_proc = body_to_proc[0: self.max_doc_size]

        if body_content_type in ('text/html', 'application/xhtml+xml'):
            encoding = None # Will become null in Java

            # TODO use something off-the shelf to do encoding extraction
            # Yet, so far Leo couldn't find an easy-to-use library to parse content response
            for resp1 in http_headers.split('\r\n'):
                if encoding is not None:
                    break
                for resp2 in resp1.split('; '):
                    fields = resp2.split('=')
                    if len(fields) == 2:
                        key, val = fields
                        if key == 'charset':
                            encoding = val
                            break

            res = JHtmlParser.parse(encoding, '', body_to_proc)
            output_dict[OUTPUT_FIELD_BODY] = res.mBodyText
            output_dict[OUTPUT_FIELD_TITLE] = res.mTitle

        elif body_content_type == 'text/plain':
            output_dict[OUTPUT_FIELD_BODY] = body_to_proc

        return output_dict





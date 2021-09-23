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

from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.ir_datasets import register

from flexneuart.io.json import get_val_err_msg_miss

JHtmlParser = autoclass('edu.cmu.lti.oaqa.flexneuart.utils.HTMLParser')

OUTPUT_FIELD_BODY = 'body'
OUTPUT_FIELD_TITLE = 'title'

@register('html_parser')
class HtmlParserProcessor(BaseTextProcessor):

    def __call__(self, input_dict: dict):
        output_dict = {OUTPUT_FIELD_BODY : '', OUTPUT_FIELD_TITLE : ''}

        body = get_val_err_msg_miss(input_dict, 'body', [str])
        body_content_type = get_val_err_msg_miss(input_dict, 'body_content_type', [str])
        http_headers = get_val_err_msg_miss(input_dict, 'http_headers', [str])

        if body_content_type in ('text/html', 'application/xhtml+xml'):
            encoding = None # Will become null in Java
            for resp in http_headers.decode().split('; '):
                fields = resp.split('=')
                if len(fields) == 2:
                    key, val = fields
                    if key == 'charset':
                        encoding = val
                        break

            # TODO Body encoding is in bytes. Does this actually work proeprly for non-ASCII text?
            res = JHtmlParser.parse(encoding, '', body)
            output_dict[OUTPUT_FIELD_BODY] = res.mBodyText
            output_dict[OUTPUT_FIELD_TITLE] = res.mTitle

        elif body_content_type == 'text/plain':
            output_dict[OUTPUT_FIELD_BODY] = body.decode()

        return output_dict





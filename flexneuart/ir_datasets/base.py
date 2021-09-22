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
    Base class for configurable processing components. Processing components are
    designed to be pipelined.
"""

class BaseTextProcessor:

    def __call__(self, input_dict : dict):
        """Process all input components to produce one or more outputs.

            :param input_dict:  input data, keys are names and values are string


            :return: the processor can produce more than one output piece, which need
                     to be represented as a dictionary. For example, the HTML parser
                     can generate a body and a title field. The naming conventions
                     of output depends on the component, but two common approaches would be:
                     1. Use the input field name
                     2. <input field name> . <operation type>
        """
        raise NotImplementedError

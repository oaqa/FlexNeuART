/*
 *  Copyright 2016 Carnegie Mellon University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package edu.cmu.lti.oaqa.knn4qa.squad;

public class SQuADEntry {
  public String            title;
  public SQuADParagraph[]  paragraphs;
  /*
   *  This constructor creates an entry with a single paragraph.
   */
  public SQuADEntry(String title, SQuADParagraph onePara) {
    this.title = title;
    paragraphs = new SQuADParagraph[1];
    paragraphs[0] = onePara;
  }
}

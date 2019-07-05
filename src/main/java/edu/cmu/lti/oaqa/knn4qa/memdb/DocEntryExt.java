/*
 *  Copyright 2015 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.memdb;

/**
 * One document entry of an in-memory forward index.
 * 
 * @author Leonid Boytsov
 *
 */
public class DocEntryExt implements Comparable<DocEntryExt> {
  public final DocEntry  mDocEntry;
  public final String    mId;

  DocEntryExt(String id, DocEntry docEntry) {
    mDocEntry = docEntry;
    mId = id;
  }
  
  @Override
  public int compareTo(DocEntryExt o) {
    return mId.compareTo(o.mId);
  }
}

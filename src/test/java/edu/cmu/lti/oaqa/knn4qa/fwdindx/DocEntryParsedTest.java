/*
 *  Copyright 2014+ Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.fwdindx;

import org.junit.Test;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;

public class DocEntryParsedTest {
	
	void doTest(DocEntryParsed src) throws Exception {
		byte [] bin = src.toBinary();
		
		DocEntryParsed dst = DocEntryParsed.fromBinary(bin);
		if (!src.equals(dst)) {
			System.out.println("Decoded is different from source!");
			System.out.println("Source: " + src);
			System.out.println("Decoded: " + dst);
			System.out.println("Source == null " + (src == null));
			System.out.println("Decoded == null " + (dst == null));
		}
		assertTrue(src.equals(dst));
	}
	
	ArrayList<Integer> getArray(int [] arr) {
		ArrayList<Integer> res = new ArrayList<Integer>();
		for (int e : arr) {
			res.add(e);
		}
		return res;
	}
	
	
	@Test
  public void test1() {
		int wordIds[] = {1, 2, 3, 4, 5, 101, 1001};
		int qtys[] = {2, 3, 5, 7, 2, 4, 5};
		int wordIdsSeq[] = {1, 10, 15, 3, 4, -1, 22, 128, -1, 1024, 1025, 1027, 1029};
		
		try {
			doTest(new DocEntryParsed(getArray(wordIds), getArray(qtys), getArray(wordIdsSeq), true));
			doTest(new DocEntryParsed(getArray(wordIds), getArray(qtys), getArray(wordIdsSeq), false));

		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}
	}
	
	@Test
  public void test2() {
		
		try {
			ArrayList<Integer> wordIds = new ArrayList<Integer>();
			ArrayList<Integer> qtys = new ArrayList<Integer>();
			ArrayList<Integer> wordIdsSeq = new ArrayList<Integer>();
			
			for (int uniqQty = 0; uniqQty < 10; ++uniqQty) {
				for (int docLen = 0; docLen < 10; ++docLen) {
					wordIds.clear();
					qtys.clear();
					wordIdsSeq.clear();
					
					for (int i = 0; i < uniqQty; ++i) {
						wordIds.add(i);
						qtys.add(i * 5);
					}
					for (int i = 0; i < docLen; ++i) {
						wordIdsSeq.add(i*10);
					}
					
					doTest(new DocEntryParsed(wordIds, qtys, wordIdsSeq, true));
					doTest(new DocEntryParsed(wordIds, qtys, wordIdsSeq, false));
				}
			}
			
			
		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}
	}

}

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
package edu.cmu.lti.oaqa.knn4qa.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;

import org.junit.Test;

public class CompressUtilsTest {

	void doTestGZIP(String src) {
		try {
			byte [] compr = CompressUtils.gzipStr(src);
			float srcLen = src.getBytes(Const.ENCODING).length;
			System.out.println("GZIP String byte length: " + srcLen + " compressed len: " + compr.length + 
												 " ratio: " + (srcLen/compr.length));
			assertEquals(src, CompressUtils.ungzipStr(compr));
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		}
	}
	
	void doTestBZIP2(String src) {
		try {
			byte [] compr = CompressUtils.bzip2Str(src);
			float srcLen = src.getBytes(Const.ENCODING).length;
			System.out.println("BZIP2 String byte length: " + srcLen + " compressed len: " + compr.length + 
												 " ratio: " + (srcLen/compr.length));
			assertEquals(src, CompressUtils.unbzip2Str(compr));
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		}
	}
	
	void doTestCompr(String src) {
		try {
			byte [] compr = CompressUtils.comprStr(src);
			float srcLen = src.getBytes(Const.ENCODING).length;
			System.out.println("COMPR String byte length: " + srcLen + " compressed len: " + compr.length + 
												 " ratio: " + (srcLen/compr.length));
			assertEquals(src, CompressUtils.decomprStr(compr));
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		}
	}
	
	String mTestStr[] = {
		"",
		"This is a simple string.",
		"This is a simple string. Но оно потребует использование utf8 (it will need to use utf8)",
		"\"i'm no expert, but generally speaking, a finer-toothed saw will make cleaner cuts (and slower ones ;-) ).\n" + 
		"  among hand tools, i'd try a hack saw; for handheld power tools, a fine-toothed circle saw or perhaps one \n\r" + 
		" of those \\\"super disks\\\"; and for table-based power tools, a band saw.  either way, take your time.\n\n" + 
		"by the way, if you've got significant excess length, you can always practice a couple of different approaches " + 
		"before making the final cut.\"",
		"manhattan project. the manhattan project was a research and development undertaking during world war ii that " + 
	  "produced the first nuclear weapons. it was led by the united states with the support of the united kingdom and canada." +
		" from 1942 to 1946, the project was under the direction of major general leslie groves of the u.s. army corps of engineers. " +
	  " nuclear physicist robert oppenheimer was the director of the los alamos laboratory that designed the actual bombs. " +
		"the army component of the project was designated the\n"
	};
	
	@Test
	public void test() {
		for (String s : mTestStr) {
			doTestGZIP(s);
			doTestBZIP2(s);
			doTestCompr(s);
		}
	}
	
}

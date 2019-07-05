package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.*;

import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;

class Diff {
  public Diff(int mLineNum, int mCharNum, char c1, char c2) {
    this.mLineNum = mLineNum;
    this.mCharNum = mCharNum;
    mC1 = c1;
    mC2 = c2;
  }
  int mLineNum;
  int mCharNum;
  char mC1;
  char mC2;
}

/**
 *  Just a simple class to test if the XML iterator works as it's supposed
 *  to work. In addition, we test if the XML parser works as well. The main
 *  assumptions are:
 *  1) Each XML entry in the file starts in the beginning
 *  of the line and is terminated by a new line. In other words, each line
 *  contains content of exactly one entry;
 *  2) the enclosing tag is not a prefix of another tag; 
 *  3) There are no empty entries. 
 *  This heuristics allows us to determine easily the start/end of the
 *  entry. In turn, the tested XML iterator works without relying on this
 *  heuristics.
 */
public class TextXmlIterAndParserApp {
  
  private static final String NL = System.getProperty("line.separator");

  public static void usage(String err) {
    System.err.println("Error: " + err);
    System.err.println("Usage: <input file> <enclosing tag>");
    System.exit(1);
  }

  public static void main(String[] args) {
    if (args.length != 2) usage("Wrong number of arguments");
    String inputFileName = args[0];
    String enclTag       = args[1];

    try {
      XmlIterator xmlIter = new XmlIterator(CompressUtils.createInputStream(inputFileName), enclTag);
      BufferedReader rd = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(inputFileName)));
      
      String xmlIterStr; 
      int recNum = 0;
      int lineNum = 0;
      
      String startTag = "<"+enclTag;
      String endTag   = "</"+enclTag+">";
      
      while (true) {
        xmlIterStr = xmlIter.readNext();
        ++recNum;
        StringBuffer sb = new StringBuffer();
        String s;
        boolean bStart = false;
        while ((s=rd.readLine()) != null) {
          s = s + NL;
          ++lineNum;
          if (!bStart) {
            int pos = s.indexOf(startTag);
            if (pos >= 0) {
              sb.append(s.substring(pos));
              bStart = true;
            }
          } else {
            int pos = s.indexOf(endTag);
            if (pos >= 0) {
              sb.append(s.substring(0, pos + endTag.length()));
              break;
            } else { sb.append(s); }
          }
        }
        String testStr = sb.toString();
        if (xmlIterStr.isEmpty() && testStr.isEmpty()) break;
        if (xmlIterStr.isEmpty() && !testStr.isEmpty()) {
          System.err.println(String.format(
"Mismatch approximately in record %d line %d, xmlIterStr is empty, but testStr is NOT!",
              recNum, lineNum));
          System.exit(1);
        }
        if (!xmlIterStr.isEmpty() && testStr.isEmpty()) {
          System.err.println(String.format(
"Mismatch approximately in record %d line %d, xmlIterStr is NOT empty, but testStr is!",
              recNum, lineNum));
          System.exit(1);
        }
        xmlIterStr = xmlIterStr.trim().replaceAll("\r+", ""); // "\r" may come from a file in DOS encoding
        testStr = testStr.trim();
        if (!xmlIterStr.equals(testStr)) {
          Diff d = getFirstDiffLine(xmlIterStr, testStr);
          
          System.err.println(String.format("Mismatch in record %d",  recNum, lineNum));
          System.err.println(String.format("The first diff line %d char %d c1=%d c2=%d", 
                                            d.mLineNum, d.mCharNum, (int)d.mC1, (int)d.mC2));          
          System.err.println(String.format("xmlIterStr:\n%s", xmlIterStr));
          System.err.println(String.format("testStr:\n%s", testStr));


          System.exit(1);
        }
        if (recNum % 10000 == 0) System.out.println(String.format("# of rec processed: %d", recNum));
      }
      System.out.println(String.format("# of rec processed: %d", recNum));
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
      System.exit(1);
    }
  }

  private static Diff getFirstDiffLine(String s1, String s2) {
    String [] p1 = s1.split(NL);
    String [] p2 = s2.split(NL);
    for (int i = 0; i < Math.min(p1.length, p2.length); ++i) {
      String d1 = p1[i], d2 = p2[i];
      if (!d1.equals(d2)) {        
        for (int k = 0; k < Math.min(d1.length(), d2.length()); ++k) {
          if (d1.charAt(k) != d2.charAt(k)) return new Diff(i+1, k+1, d1.charAt(k), d2.charAt(k));
        }
        int cind = Math.min(d1.length(), d2.length());
        return new Diff(i+1, cind, cind < d1.length() ? d1.charAt(cind) : (char)0, 
                                   cind < d2.length() ? d2.charAt(cind) : (char)0);
      }
    }
    return new Diff(Math.min(p1.length, p2.length) + 1, 1, (char)0, (char)0);
  }
}

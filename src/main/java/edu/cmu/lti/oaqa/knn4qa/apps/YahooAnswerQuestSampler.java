package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.*;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

import edu.cmu.lti.oaqa.knn4qa.collection_reader.ParsedQuestion;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersParser;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersReader;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;

public class YahooAnswerQuestSampler {  

  static void Usage() {
    System.err.println("Usage <input file> <output file> <qty>");
    System.exit(1);
  }
  
  public static void main(String[] args) {
    try {
      if (args.length != 3)
        Usage();
      BufferedWriter output = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(args[1])));
      
      int qty = Integer.parseInt(args[2]);

      XmlIterator inpIter = new XmlIterator(CompressUtils.createInputStream(args[0]), 
                                            YahooAnswersReader.DOCUMENT_TAG);
      
      String docText = null;
      
      int num = 0;
      
      // This going to be a reservoir sampling
      ArrayList<ParsedQuestion> res = new ArrayList<ParsedQuestion>();
      
      while (!(docText = inpIter.readNext()).isEmpty()) {
        ++num;
        int iReplace = ThreadLocalRandom.current().nextInt(0, num);
        if (num <= qty || iReplace < qty) {
          ParsedQuestion parsed = null;
          try {
            parsed = YahooAnswersParser.parse(docText, true /*do clean up */);
          } catch (Exception e) {      
          // If <bestanswer>...</bestanswer> is missing we may end up here...
          // This is a bit funny, because this element is supposed to be mandatory,
          // but it's not.
            System.err.println("Ignoring... invalid item, exception: " + e);
            continue;
          }
          if (parsed != null) {
            if (num <= qty) {
              res.add(parsed);
              if (res.size() != num)
                throw new Exception("Size mismatch bug (1)!");
            } else {
              if (res.size() != qty)
                throw new Exception("Size mismatch bug (2)!");
              res.set(iReplace, parsed);
            }
          }
        }        
      }
      
      for (ParsedQuestion parsed : res) 
      if (parsed.mBestAnswId >= 0) {
        output.write((replaceNL(parsed.mQuestion + " / " + parsed.mQuestDetail)) + "\t" + 
                            replaceNL(parsed.mAnswers.get(parsed.mBestAnswId)));
        output.newLine();
      }
      
      output.close();
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
  }

  private static String replaceNL(String s) {
    return s.replace('\n', ' ').replace('\r', ' ');
  }

}

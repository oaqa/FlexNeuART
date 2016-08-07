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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.File;
import java.io.IOException;
import java.util.*;

import edu.mit.jwi.*;
import edu.mit.jwi.data.*;
import edu.mit.jwi.item.*;
import edu.mit.jwi.morph.WordnetStemmer;


/**
 * 
 * Various WordNet utils.
 * 
 * @author Leonid Boytsov
 *
 */
public class WordNetUtil {   
  // Should be different from UtilConst.VALUE_SEPARATOR!!!
  private static final String NAME_PART_SEP = ".";
  
  WordnetStemmer  mStemmer = null;
  IDictionary     mDict = null;  
  boolean         mUseLexFileName = false;
  POSUtil         mPosUtil = new POSUtil();
  
  /**
   * 
   * 
   * @param wordNetDir          A directory with WordNet files.
   * @param bUseLexFileName     Do we want to use lexical file numbers in sense representations?
   * @throws IOException
   */
  public WordNetUtil(String wordNetDir, boolean bUseLexFileName) throws IOException {
    mUseLexFileName = bUseLexFileName;
    mDict = new RAMDictionary (new File(wordNetDir), ILoadPolicy.NO_LOAD
        //ILoadPolicy.BACKGROUND_LOAD /* all load options fails due to NullPointerException! */
        );
    
    mDict.open();
    mStemmer = new WordnetStemmer(mDict);
  }
  
  public static String printableName(IWord word, boolean bUseLexFileName) {
    ISynset syn = word.getSynset();
    StringBuilder sb = new StringBuilder();
    sb.append(word.getLemma());
    sb.append(NAME_PART_SEP);
    sb.append(syn.getPOS().getTag());
    sb.append(NAME_PART_SEP);
    // This is the sense number or name
    if (bUseLexFileName)
      sb.append(syn.getLexicalFile().getName().toLowerCase());
    else 
      sb.append(syn.getLexicalFile().getNumber());
    sb.append(NAME_PART_SEP);
    sb.append(word.getLexicalID());
    String res = sb.toString();
    return res.replaceAll("[_]", NAME_PART_SEP);
  }
  
  public HashSet<String> getRelatedWords(String origStr, 
                                         POS pos,
                                         IPointer type) {
    HashSet<String>   seen = new HashSet<String>();
  
    for (String lemma: mStemmer.findStems(origStr, pos)) {    
      IIndexWord        idxWord = mDict.getIndexWord(lemma, pos);
    
      if (idxWord == null) continue; // For some reason this happens sometimes
      
      seen.addAll(getRelated(idxWord, type));
    }
    
    return seen;
  }

  public HashSet<String> getRelated(IIndexWord idxWord, IPointer type) {
    HashSet<String>   seen = new HashSet<String>();

    
    for (IWordID wordID: idxWord.getWordIDs()) { 
      IWord word = mDict.getWord(wordID);
      
      List<ISynsetID> relList = word.getSynset().getRelatedSynsets(type);
      
      for( ISynsetID sid : relList){
        List <IWord> hw = mDict.getSynset(sid).getWords();      

        for(Iterator <IWord > i = hw.iterator (); i. hasNext () ;){
          IWord relWord = i.next();
          seen.add(relWord.getLemma());
        }
      }

    }
    return seen;
  }  
  
  /**
   * Return lexical file name (WordNet class) corresponding
   * to the most frequent sense.
   * 
   * @param origStr input word
   * @param pos     POS tag
   * @return        a lexical file name or an empty string.
   */
  public String getLexName(String origStr, POS pos) {
    String res = "";
    
    for (String lemma: mStemmer.findStems(origStr, pos)) {    
      IIndexWord        idxWord = mDict.getIndexWord(lemma, pos);
      
      if (idxWord == null) continue; // For some reason this happens sometimes
      
      for (IWordID wordID: idxWord.getWordIDs()) {
        IWord   word = mDict.getWord(wordID);
        ISynset syn = word.getSynset();
        
        //System.out.println(printableName(word, mUseLexFileName));
        
        // This is the sense number or name
        res = mUseLexFileName ? 
          syn.getLexicalFile().getName().toLowerCase() :
          syn.getLexicalFile().getNumber() + "";
        
        break;
      }
      if (!res.isEmpty()) break;
    }
    
    return res;
  }
  
  /**
   * Return lexical file name (WordNet class) corresponding
   * to the most frequent sense, use Penn Treebank POS tag name
   * to specify the part of speech.
   * 
   * @param origStr input word
   * @param tagStr  Penn Treebank POS tag
   * @return        a lexical file name or an empty string.
   */
  public String getLexName(String origStr, String tagStr) {
    Character p = mPosUtil.get(tagStr);
    if (p == null) return "";
    POS pos = POS.getPartOfSpeech(p);
    if (null == pos) return "";
    return getLexName(origStr, pos);
  }
  
  /**
   * @param origStr       a word/string
   * @param pos           POS tag 
   * @param maxSenseQty   maximum # of most frequent senses
   * @param hyperDepth    0 for the word, 1 for 1st direct hypernym, 
                          2 for the hypernym of hypernym, and so on.
                          
   * @return a list of most common WordNet senses and their respective hypernyms.
   */
  public HashSet<String> getSenses(String origStr, 
                                     POS pos, 
                                     int maxSenseQty, 
                                     int hyperDepth) {    
    HashSet<String>   seen = new HashSet<String>();

    int senseId = 0;
    
    for (String lemma: mStemmer.findStems(origStr, pos)) {    
      IIndexWord        idxWord = mDict.getIndexWord(lemma, pos);
      
      if (idxWord == null) continue; // For some reason this happens sometimes
      
      for (IWordID wordID: idxWord.getWordIDs()) {
        if (++senseId > maxSenseQty) break; 
        IWord word = mDict.getWord(wordID);
        
        seen.add(printableName(word, mUseLexFileName));
        getHyperRecursive(seen, word, hyperDepth, 1);
      }
    }
    
    return seen;
  }
  
  private void getHyperRecursive(HashSet<String> seen,
                                 IWord word, int hyperDepth, int dist) {
    if (dist > hyperDepth) return;
    
    List<ISynsetID> hypernyms =
        word.getSynset().getRelatedSynsets ( Pointer.HYPERNYM );
    
    List <IWord> hw;
    
    for( ISynsetID sid : hypernyms ){
      hw = mDict.getSynset(sid).getWords();      
      
      for(Iterator <IWord > i = hw.iterator (); i. hasNext () ;){
        IWord relWord = i.next();
        String wstr = printableName(relWord, mUseLexFileName);
        if (seen.contains(wstr)) continue;
        seen.add(wstr);
        getHyperRecursive(seen, relWord, hyperDepth,dist+1);
      }
    }    
  }
  
  

  public static void main (String [] args) throws IOException {
    if (args.length != 6) {
      System.err.println("Arguments: <wordnet directory> " + 
          "<use lexical file name/number true/false> <surface form> <POS symbol (n|v|a|r)> " +
          "<# of common senses> <# hypernym depth>");
      System.exit(1);
    }
    WordNetUtil wn = new WordNetUtil(args[0],
                                     Boolean.parseBoolean(args[1]));

    System.out.println("\n\n\n==============================");
    
    String  origStr = args[2];
    POS     pos = POS.getPartOfSpeech(args[3].charAt(0));
    
    int maxSenseQty = Integer.parseInt(args[4]);
    for (String sense: wn.getSenses(origStr, 
                                  pos,
                                  maxSenseQty,
                                  Integer.parseInt(args[5]))) {
      System.out.println(sense);
    }

    System.out.println("==============================\n\n\n");
    
    System.out.println("Lexical file name/number: " + wn.getLexName(origStr, pos));
    
    System.out.println("==============================\n\n\n");
    
    Scanner scanner = new Scanner(System.in);
    
    while (true) {
      System.out.println("Enter the sentence, where each string is followed by '/' and POS tag");
      String line = scanner.nextLine().trim();
      for (String s : line.split("\\s+")) {
        s = s.trim();
        String parts[] = s.split("/");
        if (parts.length != 2) {
          System.err.println("Wrong format for the string: '" + s + "', should be word/POS");
          continue;
        }
        
        String res = "?";
        String tmp = wn.getLexName(parts[0], parts[1]);
        if (!tmp.isEmpty()) res = tmp;
        
        System.out.println(parts[0] + "\t" + res);
      }
    }
  }  
  
}

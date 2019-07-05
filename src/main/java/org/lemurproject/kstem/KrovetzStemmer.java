// BSD License (http://lemurproject.org/galago-license)
/*
Copyright 2003,
Center for Intelligent Information Retrieval,
University of Massachusetts, Amherst.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. The names "Center for Intelligent Information Retrieval" and
"University of Massachusetts" must not be used to endorse or promote products
derived from this software without prior written permission. To obtain
permission, contact info@ciir.cs.umass.edu.

THIS SOFTWARE IS PROVIDED BY UNIVERSITY OF MASSACHUSETTS AND OTHER CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
 */
package org.lemurproject.kstem;

import java.io.FileReader;
import java.io.LineNumberReader;
import java.util.HashMap;

/**
 * <p>Title: Kstemmer</p>
 * <p>Description: This is a java version of Bob Krovetz' kstem stemmer</p>
 * <p>Copyright: Copyright (c) 2003</p>
 * <p>Company: CIIR University of Massachusetts Amherst (http://ciir.cs.umass.edu) </p>
 * @author Sergio Guzman-Lara
 * @version 1.1
 */
/**
This class implements the Kstem algorithm
 */
public class KrovetzStemmer extends Stemmer {

  /** Default size of the cache that stores <code>(word,stem)</code> pairs.
  <p>This speeds up processing since Kstem works by
  sucessive "transformations" to the input word until a
  suitable stem is found.
   */
  //static public int DEFAULT_CACHE_SIZE = 20000;
  static private final int MaxWordLen = 100;
  static private final String[] exceptionWords = {"aide",
    "bathe", "caste", "cute", "dame", "dime", "doge", "done", "dune",
    "envelope", "gage", "grille", "grippe", "lobe", "mane", "mare",
    "nape", "node", "pane", "pate", "plane", "pope", "programme",
    "quite", "ripe", "rote", "rune", "sage", "severe", "shoppe",
    "sine", "slime", "snipe", "steppe", "suite", "swinge", "tare",
    "tine", "tope", "tripe", "twine"};
  static private final String[][] directConflations = {
    {"aging", "age"}, {"going", "go"}, {"goes", "go"}, {"lying", "lie"},
    {"using", "use"}, {"owing", "owe"}, {"suing", "sue"}, {"dying", "die"},
    {"tying", "tie"}, {"vying", "vie"}, {"aged", "age"}, {"used", "use"},
    {"vied", "vie"}, {"cued", "cue"}, {"died", "die"}, {"eyed", "eye"},
    {"hued", "hue"}, {"iced", "ice"}, {"lied", "lie"}, {"owed", "owe"},
    {"sued", "sue"}, {"toed", "toe"}, {"tied", "tie"}, {"does", "do"},
    {"doing", "do"}, {"aeronautical", "aeronautics"},
    {"mathematical", "mathematics"}, {"political", "politics"},
    {"metaphysical", "metaphysics"}, {"cylindrical", "cylinder"},
    {"nazism", "nazi"}, {"ambiguity", "ambiguous"}, {"barbarity", "barbarous"},
    {"credulity", "credulous"}, {"generosity", "generous"}, {"spontaneity", "spontaneous"},
    {"unanimity", "unanimous"}, {"voracity", "voracious"}, {"fled", "flee"},
    {"miscarriage", "miscarry"},
    {"appendices", "appendix"},
    {"babysitting", "babysit"},
    {"bater", "bate"},
    {"belying", "belie"},
    {"bookshelves", "bookshelf"},
    {"bootstrapped", "bootstrap"},
    {"bootstrapping", "bootstrap"},
    {"checksummed", "checksum"},
    {"checksumming", "checksum"},
    {"crises", "crisis"},
    {"dwarves", "dwarf"},
    {"eerily", "eerie"},
    {"housewives", "housewife"},
    {"midwives", "midwife"},
    {"oases", "oasis"},
    {"parentheses", "parenthesis"},
    {"scarves", "scarf"},
    {"synopses", "synopsis"},
    {"syntheses", "synthesis"},
    {"taxied", "taxi"},
    {"testes", "testicle"},
    {"theses", "thesis"},
    {"thieves", "thief"},
    {"vortices", "vortex"},
    {"wharves", "wharf"},
    {"wolves", "wolf"},
    {"yourselves", "yourself"}};
  static private final String[][] countryNationality = {
    {"afghan", "afghanistan"}, {"african", "africa"}, {"albanian", "albania"},
    {"algerian", "algeria"}, {"american", "america"}, {"andorran", "andorra"},
    {"angolan", "angola"}, {"arabian", "arabia"}, {"argentine", "argentina"},
    {"armenian", "armenia"}, {"asian", "asia"}, {"australian", "australia"},
    {"austrian", "austria"}, {"azerbaijani", "azerbaijan"},
    {"azeri", "azerbaijan"}, {"bangladeshi", "bangladesh"},
    {"belgian", "belgium"}, {"bermudan", "bermuda"},
    {"bolivian", "bolivia"}, {"bosnian", "bosnia"}, {"botswanan", "botswana"},
    {"brazilian", "brazil"}, {"british", "britain"}, {"bulgarian", "bulgaria"},
    {"burmese", "burma"}, {"californian", "california"}, {"cambodian", "cambodia"},
    {"canadian", "canada"}, {"chadian", "chad"}, {"chilean", "chile"},
    {"chinese", "china"}, {"colombian", "colombia"}, {"croat", "croatia"},
    {"croatian", "croatia"}, {"cuban", "cuba"}, {"cypriot", "cyprus"},
    {"czechoslovakian", "czechoslovakia"}, {"danish", "denmark"}, {"egyptian", "egypt"},
    {"equadorian", "equador"}, {"eritrean", "eritrea"}, {"estonian", "estonia"},
    {"ethiopian", "ethiopia"}, {"european", "europe"}, {"fijian", "fiji"},
    {"filipino", "philippines"}, {"finnish", "finland"}, {"french", "france"},
    {"gambian", "gambia"}, {"georgian", "georgia"}, {"german", "germany"},
    {"ghanian", "ghana"}, {"greek", "greece"}, {"grenadan", "grenada"},
    {"guamian", "guam"}, {"guatemalan", "guatemala"}, {"guinean", "guinea"},
    {"guyanan", "guyana"}, {"haitian", "haiti"}, {"hawaiian", "hawaii"},
    {"holland", "dutch"}, {"honduran", "honduras"}, {"hungarian", "hungary"},
    {"icelandic", "iceland"}, {"indonesian", "indonesia"},
    {"iranian", "iran"}, {"iraqi", "iraq"}, {"iraqui", "iraq"}, {"irish", "ireland"},
    {"israeli", "israel"}, {"italian", "italy"}, {"jamaican", "jamaica"}, {"japanese", "japan"},
    {"jordanian", "jordan"}, {"kampuchean", "cambodia"}, {"kenyan", "kenya"},
    {"korean", "korea"}, {"kuwaiti", "kuwait"}, {"lankan", "lanka"},
    {"laotian", "laos"}, {"latvian", "latvia"}, {"lebanese", "lebanon"},
    {"liberian", "liberia"}, {"libyan", "libya"}, {"lithuanian", "lithuania"},
    {"macedonian", "macedonia"}, {"madagascan", "madagascar"}, {"malaysian", "malaysia"},
    {"maltese", "malta"}, {"mauritanian", "mauritania"}, {"mexican", "mexico"},
    {"micronesian", "micronesia"}, {"moldovan", "moldova"}, {"monacan", "monaco"},
    {"mongolian", "mongolia"}, {"montenegran", "montenegro"}, {"moroccan", "morocco"},
    {"myanmar", "burma"}, {"namibian", "namibia"}, {"nepalese", "nepal"},
    //{"netherlands", "dutch"},
    {"nicaraguan", "nicaragua"}, {"nigerian", "nigeria"},
    {"norwegian", "norway"}, {"omani", "oman"}, {"pakistani", "pakistan"},
    {"panamanian", "panama"}, {"papuan", "papua"}, {"paraguayan", "paraguay"},
    {"peruvian", "peru"}, {"portuguese", "portugal"}, {"romanian", "romania"},
    {"rumania", "romania"}, {"rumanian", "romania"}, {"russian", "russia"},
    {"rwandan", "rwanda"}, {"samoan", "samoa"}, {"scottish", "scotland"},
    {"serb", "serbia"}, {"serbian", "serbia"}, {"siam", "thailand"},
    {"siamese", "thailand"}, {"slovakia", "slovak"}, {"slovakian", "slovak"},
    {"slovenian", "slovenia"}, {"somali", "somalia"}, {"somalian", "somalia"},
    {"spanish", "spain"}, {"swedish", "sweden"}, {"swiss", "switzerland"},
    {"syrian", "syria"}, {"taiwanese", "taiwan"}, {"tanzanian", "tanzania"},
    {"texan", "texas"}, {"thai", "thailand"}, {"tunisian", "tunisia"},
    {"turkish", "turkey"}, {"ugandan", "uganda"}, {"ukrainian", "ukraine"},
    {"uruguayan", "uruguay"}, {"uzbek", "uzbekistan"}, {"venezuelan", "venezuela"},
    {"vietnamese", "viet"}, {"virginian", "virginia"}, {"yemeni", "yemen"},
    {"yugoslav", "yugoslavia"}, {"yugoslavian", "yugoslavia"}, {"zambian", "zambia"},
    {"zealander", "zealand"}, {"zimbabwean", "zimbabwe"}};
  static private final String[] supplementDict = {
    "aids",
    "applicator", "capacitor", "digitize", "electromagnet",
    "ellipsoid", "exosphere", "extensible", "ferromagnet",
    "graphics", "hydromagnet", "polygraph", "toroid", "superconduct",
    "backscatter", "connectionism"
  };
  static private final String[] properNouns = {
    "abrams", "achilles", "acropolis", "adams", "agnes",
    "aires", "alexander", "alexis", "alfred", "algiers",
    "alps", "amadeus", "ames", "amos", "andes",
    "angeles", "annapolis", "antilles", "aquarius", "archimedes",
    "arkansas", "asher", "ashly", "athens", "atkins",
    "atlantis", "avis", "bahamas", "bangor", "barbados",
    "barger", "bering", "brahms", "brandeis", "brussels",
    "bruxelles", "cairns", "camoros", "camus", "carlos",
    "celts", "chalker", "charles", "cheops", "ching",
    "christmas", "cocos", "collins", "columbus", "confucius",
    "conners", "connolly", "copernicus", "cramer", "cyclops",
    "cygnus", "cyprus", "dallas", "damascus", "daniels",
    "davies", "davis", "decker", "denning", "dennis",
    "descartes", "dickens", "doris", "douglas", "downs",
    "dreyfus", "dukakis", "dulles", "dumfries", "ecclesiastes",
    "edwards", "emily", "erasmus", "euphrates", "evans",
    "everglades", "fairbanks", "federales", "fisher", "fitzsimmons",
    "fleming", "forbes", "fowler",
    "france",
    "francis",
    "goering", "goodling", "goths", "grenadines", "guiness",
    "hades", "harding", "harris", "hastings", "hawkes",
    "hawking", "hayes", "heights", "hercules", "himalayas",
    "hippocrates", "hobbs", "holmes", "honduras", "hopkins",
    "hughes", "humphreys", "illinois", "indianapolis", "inverness",
    "iris", "iroquois", "irving", "isaacs", "italy",
    "james", "jarvis", "jeffreys", "jesus", "jones",
    "josephus", "judas", "julius", "kansas", "keynes",
    "kipling", "kiwanis", "lansing", "laos", "leeds",
    "levis", "leviticus", "lewis", "louis", "maccabees",
    "madras", "maimonides", "maldive", "massachusetts", "matthews",
    "mauritius", "memphis", "mercedes", "midas", "mingus",
    "minneapolis", "mohammed", "moines", "morris", "moses",
    "myers", "myknos", "nablus", "nanjing", "nantes",
    "naples", "neal", "netherlands", "nevis", "nostradamus",
    "oedipus", "olympus", "orleans", "orly", "papas",
    "paris", "parker", "pauling", "peking", "pershing",
    "peter", "peters", "philippines", "phineas", "pisces",
    "pryor", "pythagoras", "queens", "rabelais", "ramses",
    "reynolds", "rhesus", "rhodes", "richards", "robins",
    "rodgers", "rogers", "rubens", "sagittarius", "seychelles",
    "socrates", "texas", "thames", "thomas", "tiberias",
    "tunis", "venus", "vilnius", "wales", "warner",
    "wilkins", "williams", "wyoming", "xmas", "yonkers",
    "zeus", "frances", "aarhus", "adonis", "andrews", "angus",
    "antares", "aquinas", "arcturus", "ares", "artemis", "augustus",
    "ayers", "barnabas", "barnes", "becker", "bejing", "biggs",
    "billings", "boeing", "boris", "borroughs", "briggs", "buenos",
    "calais", "caracas", "cassius", "cerberus", "ceres", "cervantes",
    "chantilly", "chartres", "chester", "connally",
    "conner", "coors", "cummings", "curtis", "daedalus", "dionysus",
    "dobbs", "dolores", "edmonds"};

  private static class DictEntry {

    boolean exception;
    String root;

    public DictEntry(String root, boolean isException) {
      this.root = root;
      this.exception = isException;
    }
  }
  private static HashMap dict_ht = null;
  //private int MaxCacheSize;
  //private HashMap stem_ht = null;
  private StringBuffer word;
  private int j; /* index of final letter in stem (within word) */

  private int k; /* INDEX of final letter in word.
  You must add 1 to k to get the current length of word.
  When you want the length of word, use the method
  wordLength, which returns (k+1). */


//  private void initializeStemHash() {
//    stem_ht = new HashMap();
//  }
  private char finalChar() {
    return word.charAt(k);
  }

  private char penultChar() {
    return word.charAt(k - 1);
  }

  private boolean isVowel(int index) {
    return !isCons(index);
  }

  private boolean isCons(int index) {
    char ch;

    ch = word.charAt(index);

    if ((ch == 'a') || (ch == 'e') || (ch == 'i') || (ch == 'o') || (ch == 'u')) {
      return false;
    }
    if ((ch != 'y') || (index == 0)) {
      return true;
    } else {
      return (!isCons(index - 1));
    }
  }

  private static synchronized void initializeDictHash() {
    DictEntry defaultEntry;
    DictEntry entry;

    if (dict_ht != null) {
      return;
    }

    dict_ht = new HashMap();
    for (int i = 0; i < exceptionWords.length; i++) {
      if (!dict_ht.containsKey(exceptionWords[i])) {
        entry = new DictEntry(exceptionWords[i], true);
        dict_ht.put(exceptionWords[i], entry);
      } else {
        System.out.println("Warning: Entry [" + exceptionWords[i]
                + "] already in dictionary 1");
      }
    }

    for (int i = 0; i < directConflations.length; i++) {
      if (!dict_ht.containsKey(directConflations[i][0])) {
        entry = new DictEntry(directConflations[i][1], false);
        dict_ht.put(directConflations[i][0], entry);
      } else {
        System.out.println("Warning: Entry [" + directConflations[i][0]
                + "] already in dictionary 2");
      }
    }

    for (int i = 0; i < countryNationality.length; i++) {
      if (!dict_ht.containsKey(countryNationality[i][0])) {
        entry = new DictEntry(countryNationality[i][1], false);
        dict_ht.put(countryNationality[i][0], entry);
      } else {
        System.out.println("Warning: Entry ["
                + countryNationality[i][0]
                + "] already in dictionary 3");
      }
    }

    defaultEntry = new DictEntry(null, false);

    String[] array;
    array = KStemData1.data;

    for (int i = 0; i < array.length; i++) {
      if (!dict_ht.containsKey(array[i])) {
        dict_ht.put(array[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + array[i]
                + "] already in dictionary 4");
      }
    }


    array = KStemData2.data;
    for (int i = 0; i < array.length; i++) {
      if (!dict_ht.containsKey(array[i])) {
        dict_ht.put(array[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + array[i]
                + "] already in dictionary 4");
      }
    }

    array = KStemData3.data;
    for (int i = 0; i < array.length; i++) {
      if (!dict_ht.containsKey(array[i])) {
        dict_ht.put(array[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + array[i]
                + "] already in dictionary 4");
      }
    }

    array = KStemData4.data;
    for (int i = 0; i < array.length; i++) {
      if (!dict_ht.containsKey(array[i])) {
        dict_ht.put(array[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + array[i]
                + "] already in dictionary 4");
      }
    }


    array = KStemData5.data;
    for (int i = 0; i < array.length; i++) {
      if (!dict_ht.containsKey(array[i])) {
        dict_ht.put(array[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + array[i]
                + "] already in dictionary 4");
      }
    }


    array = KStemData6.data;
    for (int i = 0; i < array.length; i++) {
      if (!dict_ht.containsKey(array[i])) {
        dict_ht.put(array[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + array[i]
                + "] already in dictionary 4");
      }
    }

    array = KStemData7.data;
    for (int i = 0; i < array.length; i++) {
      if (!dict_ht.containsKey(array[i])) {
        dict_ht.put(array[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + array[i]
                + "] already in dictionary 4");
      }
    }

    for (int i = 0; i < KStemData8.data.length; i++) {
      if (!dict_ht.containsKey(KStemData8.data[i])) {
        dict_ht.put(KStemData8.data[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry [" + KStemData8.data[i]
                + "] already in dictionary 4");
      }
    }

    for (int i = 0; i < supplementDict.length; i++) {
      if (!dict_ht.containsKey(supplementDict[i])) {
        dict_ht.put(supplementDict[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry ["
                + supplementDict[i]
                + "] already in dictionary 5");
      }
    }

    for (int i = 0; i < properNouns.length; i++) {
      if (!dict_ht.containsKey(properNouns[i])) {
        dict_ht.put(properNouns[i], defaultEntry);
      } else {
        System.out.println("Warning: Entry ["
                + properNouns[i]
                + "] already in dictionary 6");
      }
    }
  }

  private boolean isAlpha(char ch) {
    if ((ch >= 'a') && (ch <= 'z')) {
      return true;
    }
    if ((ch >= 'A') && (ch <= 'Z')) {
      return true;
    }
    return false;
  }

  /* length of stem within word */
  private int stemLength() {
    return j + 1;
  }

  private boolean endsIn(String s) {
    boolean match;
    int sufflength = s.length();

    int r = word.length() - sufflength; /* length of word before this suffix */
    if (sufflength > k) {
      return false;
    }

    match = true;
    for (int r1 = r, i = 0; (i < sufflength) && (match); i++, r1++) {
      if (s.charAt(i) != word.charAt(r1)) {
        match = false;
      }
    }

    if (match) {
      j = r - 1;  /* index of the character BEFORE the posfix */
    } else {
      j = k;
    }
    return match;
  }

  private DictEntry wordInDict() {
    String s = word.toString();
    return (DictEntry) dict_ht.get(s);
  }


  /* Convert plurals to singular form, and '-ies' to 'y' */
  private void plural() {
    if (finalChar() == 's') {
      if (endsIn("ies")) {
        word.setLength(j + 3);
        k--;
        if (lookup(word.toString())) /* ensure calories -> calorie */ {
          return;
        }
        k++;
        word.append('s');
        setSuffix("y");
      } else if (endsIn("es")) {
        /* try just removing the "s" */
        word.setLength(j + 2);
        k--;

        /* note: don't check for exceptions here.  So, `aides' -> `aide',
        but `aided' -> `aid'.  The exception for double s is used to prevent
        crosses -> crosse.  This is actually correct if crosses is a plural
        noun (a type of racket used in lacrosse), but the verb is much more
        common */

        if ((j > 0) && (lookup(word.toString()))
                && !((word.charAt(j) == 's')
                && (word.charAt(j - 1) == 's'))) {
          return;
        }

        /* try removing the "es" */

        word.setLength(j + 1);
        k--;
        if (lookup(word.toString())) {
          return;
        }

        /* the default is to retain the "e" */
        word.append('e');
        k++;
        return;
      } else {
        if (word.length() > 3 && penultChar() != 's'
                && !endsIn("ous")) {
          /* unless the word ends in "ous" or a double "s", remove the final "s" */

          word.setLength(k);
          k--;
        }
      }
    }
  }

  private void setSuffix(String s) {
    setSuff(s, s.length());
  }

  /* replace old suffix with s */
  private void setSuff(String s, int len) {
    word.setLength(j + 1);
    for (int l = 0; l < len; l++) {
      word.append(s.charAt(l));
    }
    k = j + len;
  }

  /* Returns true if s is found in the dictionary */
  private boolean lookup(String s) {
    if (dict_ht.containsKey(s)) {
      return true;
    } else {
      return false;
    }
  }

  /* convert past tense (-ed) to present, and `-ied' to `y' */
  private void pastTense() {
    /* Handle words less than 5 letters with a direct mapping
    This prevents (fled -> fl).  */

    if (word.length() <= 4) {
      return;
    }

    if (endsIn("ied")) {
      word.setLength(j + 3);
      k--;
      if (lookup(word.toString())) /* we almost always want to convert -ied to -y, but */ {
        return;                  /* this isn't true for short words (died->die)      */
      }
      k++;                         /* I don't know any long words that this applies to, */
      word.append('d');            /* but just in case...                              */
      setSuffix("y");
      return;
    }

    /* the vowelInStem() is necessary so we don't stem acronyms */
    if (endsIn("ed") && vowelInStem()) {
      /* see if the root ends in `e' */
      word.setLength(j + 2);
      k = j + 1;

      DictEntry entry = wordInDict();
      if (entry != null) {
        if (!entry.exception) /* if it's in the dictionary and not an exception */ {
          return;
        }
      }

      /* try removing the "ed" */
      word.setLength(j + 1);
      k = j;
      if (lookup(word.toString())) {
        return;
      }


      /* try removing a doubled consonant.  if the root isn't found in
      the dictionary, the default is to leave it doubled.  This will
      correctly capture `backfilled' -> `backfill' instead of
      `backfill' -> `backfille', and seems correct most of the time  */

      if (doubleC(k)) {
        word.setLength(k);
        k--;
        if (lookup(word.toString())) {
          return;
        }
        word.append(word.charAt(k));
        k++;
        return;
      }

      /* if we have a `un-' prefix, then leave the word alone  */
      /* (this will sometimes screw up with `under-', but we   */
      /*  will take care of that later)                        */

      if ((word.charAt(0) == 'u') && (word.charAt(1) == 'n')) {
        word.append('e');
        word.append('d');
        k = k + 2;
        return;
      }


      /* it wasn't found by just removing the `d' or the `ed', so prefer to
      end with an `e' (e.g., `microcoded' -> `microcode'). */

      word.setLength(j + 1);
      word.append('e');
      k = j + 1;
      return;
    }
  }

  /* return TRUE if word ends with a double consonant */
  private boolean doubleC(int i) {
    if (i < 1) {
      return false;
    }

    if (word.charAt(i) != word.charAt(i - 1)) {
      return false;
    }
    return (isCons(i));
  }

  private boolean vowelInStem() {
    for (int i = 0; i < stemLength(); i++) {
      if (isVowel(i)) {
        return true;
      }
    }
    return false;
  }

  /* handle `-ing' endings */
  private void aspect() {
    /* handle short words (aging -> age) via a direct mapping.  This
    prevents (thing -> the) in the version of this routine that
    ignores inflectional variants that are mentioned in the dictionary
    (when the root is also present) */

    if (word.length() <= 5) {
      return;
    }

    /* the vowelinstem() is necessary so we don't stem acronyms */
    if (endsIn("ing") && vowelInStem()) {

      /* try adding an `e' to the stem and check against the dictionary */
      word.setCharAt(j + 1, 'e');
      word.setLength(j + 2);
      k = j + 1;

      DictEntry entry = wordInDict();
      if (entry != null) {
        if (!entry.exception) /* if it's in the dictionary and not an exception */ {
          return;
        }
      }

      /* adding on the `e' didn't work, so remove it */
      word.setLength(k);
      k--;           /* note that `ing' has also been removed */

      if (lookup(word.toString())) {
        return;
      }

      /* if I can remove a doubled consonant and get a word, then do so */
      if (doubleC(k)) {
        k--;
        word.setLength(k + 1);
        if (lookup(word.toString())) {
          return;
        }
        word.append(word.charAt(k)); /* restore the doubled consonant */

        /* the default is to leave the consonant doubled            */
        /*  (e.g.,`fingerspelling' -> `fingerspell').  Unfortunately */
        /*  `bookselling' -> `booksell' and `mislabelling' -> `mislabell'). */
        /*  Without making the algorithm significantly more complicated, this */
        /*  is the best I can do */
        k++;
        return;
      }

      /* the word wasn't in the dictionary after removing the stem, and then
      checking with and without a final `e'.  The default is to add an `e'
      unless the word ends in two consonants, so `microcoding' -> `microcode'.
      The two consonants restriction wouldn't normally be necessary, but is
      needed because we don't try to deal with prefixes and compounds, and
      most of the time it is correct (e.g., footstamping -> footstamp, not
      footstampe; however, decoupled -> decoupl).  We can prevent almost all
      of the incorrect stems if we try to do some prefix analysis first */

      if ((j > 0) && isCons(j) && isCons(j - 1)) {
        k = j;
        word.setLength(k + 1);
        return;
      }

      word.setLength(j + 1);
      word.append('e');
      k = j + 1;
      return;
    }
  }

  /* this routine deals with -ity endings.  It accepts -ability, -ibility,
  and -ality, even without checking the dictionary because they are so
  productive.  The first two are mapped to -ble, and the -ity is remove
  for the latter */
  private void ityEndings() {
    int old_k = k;

    if (endsIn("ity")) {
      word.setLength(j + 1);          /* try just removing -ity */
      k = j;
      if (lookup(word.toString())) {
        return;
      }
      word.append('e');             /* try removing -ity and adding -e */
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }
      word.setCharAt(j + 1, 'i');
      word.append("ty");
      k = old_k;
      /* the -ability and -ibility endings are highly productive, so just accept them */
      if ((j > 0) && (word.charAt(j - 1) == 'i') && (word.charAt(j) == 'l')) {
        word.setLength(j - 1);
        word.append("le");   /* convert to -ble */
        k = j;
        return;
      }


      /* ditto for -ivity */
      if ((j > 0) && (word.charAt(j - 1) == 'i') && (word.charAt(j) == 'v')) {
        word.setLength(j + 1);
        word.append('e');         /* convert to -ive */
        k = j + 1;
        return;
      }
      /* ditto for -ality */
      if ((j > 0) && (word.charAt(j - 1) == 'a') && (word.charAt(j) == 'l')) {
        word.setLength(j + 1);
        k = j;
        return;
      }

      /* if the root isn't in the dictionary, and the variant *is*
      there, then use the variant.  This allows `immunity'->`immune',
      but prevents `capacity'->`capac'.  If neither the variant nor
      the root form are in the dictionary, then remove the ending
      as a default */

      if (lookup(word.toString())) {
        return;
      }

      /* the default is to remove -ity altogether */
      word.setLength(j + 1);
      k = j;
      return;
    }
  }

  /* handle -ence and -ance */
  private void nceEndings() {
    int old_k = k;
    char word_char;

    if (endsIn("nce")) {
      if (!((word.charAt(j) == 'e') || (word.charAt(j) == 'a'))) {
        return;
      }
      word_char = word.charAt(j);
      word.setLength(j);
      word.append('e');     /* try converting -e/ance to -e (adherance/adhere) */
      k = j;
      if (lookup(word.toString())) {
        return;
      }
      word.setLength(j); /* try removing -e/ance altogether (disappearance/disappear) */
      k = j - 1;
      if (lookup(word.toString())) {
        return;
      }
      word.append(word_char);  /* restore the original ending */
      word.append("nce");
      k = old_k;
    }
    return;
  }

  /* handle -ness */
  private void nessEndings() {
    if (endsIn("ness")) {     /* this is a very productive endings, so just accept it */
      word.setLength(j + 1);
      k = j;
      if (word.charAt(j) == 'i') {
        word.setCharAt(j, 'y');
      }
    }
    return;
  }

  /* handle -ism */
  private void ismEndings() {
    if (endsIn("ism")) {    /* this is a very productive ending, so just accept it */
      word.setLength(j + 1);
      k = j;
    }
    return;
  }

  /* this routine deals with -ment endings. */
  private void mentEndings() {
    int old_k = k;

    if (endsIn("ment")) {
      word.setLength(j + 1);
      k = j;
      if (lookup(word.toString())) {
        return;
      }
      word.append("ment");
      k = old_k;
    }
    return;
  }

  /* this routine deals with -ize endings. */
  private void izeEndings() {
    int old_k = k;

    if (endsIn("ize")) {
      word.setLength(j + 1);       /* try removing -ize entirely */
      k = j;
      if (lookup(word.toString())) {
        return;
      }
      word.append('i');

      if (doubleC(j)) {      /* allow for a doubled consonant */
        word.setLength(j);
        k = j - 1;
        if (lookup(word.toString())) {
          return;
        }
        word.append(word.charAt(j - 1));
      }

      word.setLength(j + 1);
      word.append('e');        /* try removing -ize and adding -e */
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }
      word.setLength(j + 1);
      word.append("ize");
      k = old_k;
    }
    return;
  }

  /* handle -ency and -ancy */
  private void ncyEndings() {
    if (endsIn("ncy")) {
      if (!((word.charAt(j) == 'e') || (word.charAt(j) == 'a'))) {
        return;
      }
      word.setCharAt(j + 2, 't');  /* try converting -ncy to -nt */
      word.setLength(j + 3);
      k = j + 2;

      if (lookup(word.toString())) {
        return;
      }

      word.setCharAt(j + 2, 'c');  /* the default is to convert it to -nce */
      word.append('e');
      k = j + 3;
    }
    return;
  }

  /* handle -able and -ible */
  private void bleEndings() {
    int old_k = k;
    char word_char;

    if (endsIn("ble")) {
      if (!((word.charAt(j) == 'a') || (word.charAt(j) == 'i'))) {
        return;
      }
      word_char = word.charAt(j);
      word.setLength(j);         /* try just removing the ending */
      k = j - 1;
      if (lookup(word.toString())) {
        return;
      }
      if (doubleC(k)) {          /* allow for a doubled consonant */
        word.setLength(k);
        k--;
        if (lookup(word.toString())) {
          return;
        }
        k++;
        word.append(word.charAt(k - 1));
      }
      word.setLength(j);
      word.append('e');   /* try removing -a/ible and adding -e */
      k = j;
      if (lookup(word.toString())) {
        return;
      }
      word.setLength(j);
      word.append("ate"); /* try removing -able and adding -ate */
      /* (e.g., compensable/compensate)     */
      k = j + 2;
      if (lookup(word.toString())) {
        return;
      }
      word.setLength(j);
      word.append(word_char);        /* restore the original values */
      word.append("ble");
      k = old_k;
    }
    return;
  }

  /* handle -ic endings.   This is fairly straightforward, but this is
  also the only place we try *expanding* an ending, -ic -> -ical.
  This is to handle cases like `canonic' -> `canonical' */
  private void icEndings() {
    if (endsIn("ic")) {
      word.setLength(j + 3);
      word.append("al");        /* try converting -ic to -ical */
      k = j + 4;
      if (lookup(word.toString())) {
        return;
      }

      word.setCharAt(j + 1, 'y');        /* try converting -ic to -y */
      word.setLength(j + 2);
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }

      word.setCharAt(j + 1, 'e');        /* try converting -ic to -e */
      if (lookup(word.toString())) {
        return;
      }

      word.setLength(j + 1); /* try removing -ic altogether */
      k = j;
      if (lookup(word.toString())) {
        return;
      }
      word.append("ic"); /* restore the original ending */
      k = j + 2;
    }
    return;
  }

  /* handle some derivational endings */
  /* this routine deals with -ion, -ition, -ation, -ization, and -ication.  The
  -ization ending is always converted to -ize */
  private void ionEndings() {
    int old_k = k;

    if (endsIn("ization")) {   /* the -ize ending is very productive, so simply accept it as the root */
      word.setLength(j + 3);
      word.append('e');
      k = j + 3;
      return;
    }


    if (endsIn("ition")) {
      word.setLength(j + 1);
      word.append('e');
      k = j + 1;
      if (lookup(word.toString())) /* remove -ition and add `e', and check against the dictionary */ {
        return;                    /* (e.g., definition->define, opposition->oppose) */
      }

      /* restore original values */
      word.setLength(j + 1);
      word.append("ition");
      k = old_k;
    }


    if (endsIn("ation")) {
      word.setLength(j + 3);
      word.append('e');
      k = j + 3;
      if (lookup(word.toString())) /* remove -ion and add `e', and check against the dictionary */ {
        return;                  /* (elmination -> eliminate)  */
      }

      word.setLength(j + 1);
      word.append('e');   /* remove -ation and add `e', and check against the dictionary */
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }

      word.setLength(j + 1);/* just remove -ation (resignation->resign) and check dictionary */
      k = j;
      if (lookup(word.toString())) {
        return;
      }

      /* restore original values */
      word.setLength(j + 1);
      word.append("ation");
      k = old_k;
    }


    /* test -ication after -ation is attempted (e.g., `complication->complicate'
    rather than `complication->comply') */

    if (endsIn("ication")) {
      word.setLength(j + 1);
      word.append('y');
      k = j + 1;
      if (lookup(word.toString())) /* remove -ication and add `y', and check against the dictionary */ {
        return;                 /* (e.g., amplification -> amplify) */
      }

      /* restore original values */
      word.setLength(j + 1);
      word.append("ication");
      k = old_k;
    }


    if (endsIn("ion")) {
      word.setLength(j + 1);
      word.append('e');
      k = j + 1;
      if (lookup(word.toString())) /* remove -ion and add `e', and check against the dictionary */ {
        return;
      }

      word.setLength(j + 1);
      k = j;
      if (lookup(word.toString())) /* remove -ion, and if it's found, treat that as the root */ {
        return;
      }

      /* restore original values */
      word.setLength(j + 1);
      word.append("ion");
      k = old_k;
    }

    return;
  }

  /* this routine deals with -er, -or, -ier, and -eer.  The -izer ending is always converted to
  -ize */
  private void erAndOrEndings() {
    int old_k = k;

    char word_char;                 /* so we can remember if it was -er or -or */

    if (endsIn("izer")) {          /* -ize is very productive, so accept it as the root */
      word.setLength(j + 4);
      k = j + 3;
      return;
    }

    if (endsIn("er") || endsIn("or")) {
      word_char = word.charAt(j + 1);
      if (doubleC(j)) {
        word.setLength(j);
        k = j - 1;
        if (lookup(word.toString())) {
          return;
        }
        word.append(word.charAt(j - 1));       /* restore the doubled consonant */
      }


      if (word.charAt(j) == 'i') {         /* do we have a -ier ending? */
        word.setCharAt(j, 'y');
        word.setLength(j + 1);
        k = j;
        if (lookup(word.toString())) /* yes, so check against the dictionary */ {
          return;
        }
        word.setCharAt(j, 'i');             /* restore the endings */
        word.append('e');
      }


      if (word.charAt(j) == 'e') {         /* handle -eer */
        word.setLength(j);
        k = j - 1;
        if (lookup(word.toString())) {
          return;
        }
        word.append('e');
      }

      word.setLength(j + 2); /* remove the -r ending */
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }
      word.setLength(j + 1); /* try removing -er/-or */
      k = j;
      if (lookup(word.toString())) {
        return;
      }
      word.append('e');    /* try removing -or and adding -e */
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }
      word.setLength(j + 1);
      word.append(word_char);
      word.append('r');    /* restore the word to the way it was */
      k = old_k;
    }

  }

  /* this routine deals with -ly endings.  The -ally ending is always converted to -al
  Sometimes this will temporarily leave us with a non-word (e.g., heuristically
  maps to heuristical), but then the -al is removed in the next step.  */
  private void lyEndings() {
    int old_k = k;

    if (endsIn("ly")) {

      word.setCharAt(j + 2, 'e');             /* try converting -ly to -le */

      if (lookup(word.toString())) {
        return;
      }
      word.setCharAt(j + 2, 'y');

      word.setLength(j + 1);         /* try just removing the -ly */
      k = j;

      if (lookup(word.toString())) {
        return;
      }

      if ((j > 0) && (word.charAt(j - 1) == 'a') && (word.charAt(j) == 'l')) /* always convert -ally to -al */ {
        return;
      }
      word.append("ly");
      k = old_k;

      if ((j > 0) && (word.charAt(j - 1) == 'a') && (word.charAt(j) == 'b')) {  /* always convert -ably to -able */
        word.setCharAt(j + 2, 'e');
        k = j + 2;
        return;
      }

      if (word.charAt(j) == 'i') {        /* e.g., militarily -> military */
        word.setLength(j);
        word.append('y');
        k = j;
        if (lookup(word.toString())) {
          return;
        }
        word.setLength(j);
        word.append("ily");
        k = old_k;
      }

      word.setLength(j + 1); /* the default is to remove -ly */

      k = j;
    }
    return;
  }

  /* this routine deals with -al endings.  Some of the endings from the previous routine
  are finished up here.  */
  private void alEndings() {
    int old_k = k;

    if (word.length() < 4) {
      return;
    }
    if (endsIn("al")) {
      word.setLength(j + 1);
      k = j;
      if (lookup(word.toString())) /* try just removing the -al */ {
        return;
      }

      if (doubleC(j)) {            /* allow for a doubled consonant */
        word.setLength(j);
        k = j - 1;
        if (lookup(word.toString())) {
          return;
        }
        word.append(word.charAt(j - 1));
      }

      word.setLength(j + 1);
      word.append('e');              /* try removing the -al and adding -e */
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }

      word.setLength(j + 1);
      word.append("um");    /* try converting -al to -um */
      /* (e.g., optimal - > optimum ) */
      k = j + 2;
      if (lookup(word.toString())) {
        return;
      }

      word.setLength(j + 1);
      word.append("al");    /* restore the ending to the way it was */
      k = old_k;

      if ((j > 0) && (word.charAt(j - 1) == 'i') && (word.charAt(j) == 'c')) {
        word.setLength(j - 1); /* try removing -ical  */
        k = j - 2;
        if (lookup(word.toString())) {
          return;
        }

        word.setLength(j - 1);
        word.append('y');/* try turning -ical to -y (e.g., bibliographical) */
        k = j - 1;
        if (lookup(word.toString())) {
          return;
        }

        word.setLength(j - 1);
        word.append("ic"); /* the default is to convert -ical to -ic */
        k = j;
        return;
      }

      if (word.charAt(j) == 'i') {        /* sometimes -ial endings should be removed */
        word.setLength(j); /* (sometimes it gets turned into -y, but we */
        k = j - 1;                  /* aren't dealing with that case for now) */
        if (lookup(word.toString())) {
          return;
        }
        word.append("ial");
        k = old_k;
      }

    }
    return;
  }

  /* this routine deals with -ive endings.  It normalizes some of the
  -ative endings directly, and also maps some -ive endings to -ion. */
  private void iveEndings() {
    int old_k = k;

    if (endsIn("ive")) {
      word.setLength(j + 1);     /* try removing -ive entirely */
      k = j;
      if (lookup(word.toString())) {
        return;
      }

      word.append('e');          /* try removing -ive and adding -e */
      k = j + 1;
      if (lookup(word.toString())) {
        return;
      }
      word.setLength(j + 1);
      word.append("ive");
      if ((j > 0) && (word.charAt(j - 1) == 'a') && (word.charAt(j) == 't')) {
        word.setCharAt(j - 1, 'e');       /* try removing -ative and adding -e */
        word.setLength(j);        /* (e.g., determinative -> determine) */
        k = j - 1;
        if (lookup(word.toString())) {
          return;
        }
        word.setLength(j - 1); /* try just removing -ative */
        if (lookup(word.toString())) {
          return;
        }

        word.append("ative");
        k = old_k;
      }

      /* try mapping -ive to -ion (e.g., injunctive/injunction) */
      word.setCharAt(j + 2, 'o');
      word.setCharAt(j + 3, 'n');
      if (lookup(word.toString())) {
        return;
      }

      word.setCharAt(j + 2, 'v');       /* restore the original values */
      word.setCharAt(j + 3, 'e');
      k = old_k;
    }
    return;
  }

  /** Create a KrovetzStemmer 
   *
   */
  public KrovetzStemmer() {
//    MaxCacheSize = DEFAULT_CACHE_SIZE;
    if (dict_ht == null) {
      initializeDictHash();
    }
  }

  /** Returns the stem of a word.
   *  @param term The word to be stemmed.
   *  @return The stem form of the term.
   */
  @Override
  protected String stemTerm(String term) {
    boolean stemIt;
    String result;
    String original;

//    if (stem_ht == null) {
//      initializeStemHash();
//    }

    k = term.length() - 1;

    /* If the word is too long or too short, or not
    entirely alphabetic, just lowercase copy it
    into stem and return */
    stemIt = true;
    if ((k <= 1) || (k >= MaxWordLen - 1)) {
      stemIt = false;
    } else {
      word = new StringBuffer(term.length());
      for (int i = 0; i < term.length(); i++) {
        char ch = Character.toLowerCase(term.charAt(i));
        word.append(ch);
        if (!isAlpha(ch)) {
          stemIt = false;
          break;
        }
      }
    }
    if (!stemIt) {
      return term.toLowerCase();
    }
    /* Check to see if it's in the cache */
    original = word.toString();
//    if (stem_ht.containsKey(original)) {
//      return (String) stem_ht.get(original);
//    }

    result = original; /* default response */

    /* This while loop will never be executed more than one time;
    it is here only to allow the break statement to be used to escape
    as soon as a word is recognized */

    DictEntry entry = null;

    while (true) {
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      plural();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      pastTense();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      aspect();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      ityEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      nessEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      ionEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      erAndOrEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      lyEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      alEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      iveEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      izeEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      mentEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      bleEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      ismEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      icEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      ncyEndings();
      entry = wordInDict();
      if (entry != null) {
        break;
      }
      nceEndings();
      entry = wordInDict();
      break;
    }


    /* try for a direct mapping (allows for cases like `Italian'->`Italy' and
    `Italians'->`Italy')
     */
    if (entry != null) {
      if (entry.root != null) {
        result = entry.root;
      } else {
        result = word.toString();
      }
    } else {
      result = word.toString();
    }

    /* Enter into cache, at the place not used by the last cache hit */
    //   if (stem_ht.size() < MaxCacheSize) {
      /* Add term to cache */
    //    stem_ht.put(original, result);
    //  }

    return result;
  }

  // DEPRECATED....
  static private void usage() {
    System.out.println("Usage:");
    System.out.println("  KStemmer <inputFile>");
    System.out.println("    or");
    System.out.println("  KStemmer -w <word>");
    System.exit(1);
  }

  /** For testing only.
  <p>Usage:
  <ul>
  <li><code><B>KrovetzStemmer &lt;inputFile&gt;</B></code>
  <p> Will stem all words
  in <code>&lt;inputFile&gt;</code> (one word per line).
  <p>
  <li><code><B>KrovetzStemmer -w &lt;word&gt;</B></code>
  <p> Will stem a single
  <code>&lt;word&gt;</code>
  <p>
  </ul>
  In either case, the output is sent to <code>System.out</code>
   */
  static public void main(String[] args) {
    KrovetzStemmer stemmer = new KrovetzStemmer();
    String line = null;

    if ((args.length == 0) || (args.length > 2)) {
      usage();
    }

    if (args.length == 2) {
      if (!args[0].equals("-w")) {
        usage();
      }
      System.out.println(args[1] + " " + stemmer.stem(args[1]));
      return;
    }

    // If we get here, we are about to process a file

    try {
      LineNumberReader reader = new LineNumberReader(new FileReader(args[0]));

      line = reader.readLine();
      while (line != null) {
        line = line.trim();
        System.out.println(line + " " + stemmer.stem(line));
        line = reader.readLine();
      }
      reader.close();

    } catch (Exception e) {
      System.out.println("Exception while processing term [" + line + "]");
      e.printStackTrace();
    }
  }
}

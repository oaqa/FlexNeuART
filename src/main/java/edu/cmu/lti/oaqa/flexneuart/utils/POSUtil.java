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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.util.HashMap;

/**
 * 
 * Provides mapping of PennTree bank tags to WordNet tags.
 * This mapping makes sense only for a subset of WordNet tags.
 * 
 * @author Leonid Boytsov
 *
 */
public class POSUtil {
  HashMap<String, Character> mMap = new HashMap<String, Character>();
  
  public POSUtil() {
    //    1.  CC  Coordinating conjunction
    //    2.  CD  Cardinal number
    //    3.  DT  Determiner
    //    4.  EX  Existential there
    //    5.  FW  Foreign word
    //    6.  IN  Preposition or subordinating conjunction
    mMap.put("JJ", 'A');  // 7.  JJ  Adjective
    mMap.put("JJR", 'A'); // 8.  JJR Adjective, comparative
    mMap.put("JJS", 'A'); // 9.  JJS Adjective, superlative
    //    10. LS  List item marker
    //    11. MD  Modal
    mMap.put("NN", 'N');  // 12. NN  Noun, singular or mass
    mMap.put("NNS", 'N'); // 13. NNS Noun, plural
    mMap.put("NNP", 'N'); // 14. NNP Proper noun, singular
    mMap.put("NNPS", 'N');// 15. NNPS  Proper noun, plural
    //    16. PDT Predeterminer
    //    17. POS Possessive ending
    //    18. PRP Personal pronoun
    //    19. PRP$  Possessive pronoun
    mMap.put("RB", 'R');  // 20. RB  Adverb
    mMap.put("RBR", 'R'); // 21. RBR Adverb, comparative
    mMap.put("RBS", 'R'); // 22. RBS Adverb, superlative
    //    23. RP  Particle
    //    24. SYM Symbol
    //    25. TO  to
    //    26. UH  Interjection
    mMap.put("VB", 'V'); //   27. VB  Verb, base form
    mMap.put("VBD", 'V'); //  28. VBD Verb, past tense
    mMap.put("VBG", 'V'); //  29. VBG Verb, gerund or present participle
    mMap.put("VBN", 'V'); //  30. VBN Verb, past participle
    mMap.put("VBP", 'V'); //  31. VBP Verb, non-3rd person singular present
    mMap.put("VBZ", 'V'); //  32. VBZ Verb, 3rd person singular present
    //    33. WDT Wh-determiner
    //    34. WP  Wh-pronoun
    //    35. WP$ Possessive wh-pronoun
    //    36. WRB Wh-adverb
    
    String keys[] = new String[mMap.size()];
    
    int n = 0;
    for (String k:mMap.keySet()) {
      keys[n++] = k;
    }
    for (int i = 0; i < n; ++i) {
      String k = keys[i];
      mMap.put(k.toLowerCase(), mMap.get(k));
    }
  }
  
  public Character get(String pos) {
    return mMap.get(pos);
  }

}

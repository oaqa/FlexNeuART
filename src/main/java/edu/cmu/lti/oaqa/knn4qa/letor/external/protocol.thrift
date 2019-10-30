namespace java edu.cmu.lti.oaqa.knn4qa.letor.external

struct WordEntryInfo {
  1: required string      word; // a word
  2: required double      IDF; // the value of the word IDF
  3: required i32         qty; // the number of times the word repeats
}

struct TextEntryParsed {
  1: required string                id;
  2: required list<WordEntryInfo>   entries; 
}

struct TextEntryRaw {
  1: required string id;
  2: required string text;
}

exception ScoringException {
    1: string message;
}

service ExternalScorer {
  // Process parsed document entries
  map<string, list<double>> getScoresFromParsed(1: required TextEntryParsed query, // a parsed query entry
                                                2: required list<TextEntryParsed> docs) // an array of parsed documents 
  map<string, list<double>> getScoresFromRaw(1: required TextEntryRaw query, // a raw-text query entry
                                             2: required list<TextEntryRaw> docs) // an array of raw-text documents 
  throws (1: ScoringException err)
}

namespace java edu.cmu.lti.oaqa.flexneuart.embed.external

exception EmbeddingException {
    1: string message;
}

service ExternalEmbedder {
  // Process parsed document entries
  list<list<double>> embedDocuments(1: required list<string> docs) // raw-text documents
  list<double>       embedQuery(1: required string query) // a raw-text query entry
  throws (1: EmbeddingException err)
}
